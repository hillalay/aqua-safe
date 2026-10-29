from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .binary_drowning import CLASSES, read_csv, write_csv
from .evaluate import binary_video_metrics
from .model import build_r3d18, load_checkpoint, save_checkpoint
from .preprocessing import ClipRecord, build_clip_records, decode_clip, probe_video, transform_clip


ROOT = Path(__file__).resolve().parents[2]
WEAK_EVALUATION_LABEL = "experimental_weak_video_labels_not_event_ground_truth"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def load_training_rows(path: Path) -> list[dict[str, str]]:
    rows = [row for row in read_csv(path) if is_true(row["initial_training_eligible"])]
    if any(is_true(row["is_exact_duplicate"]) or row["canonical_video_id"] != row["video_id"] for row in rows):
        raise ValueError("Exact duplicates cannot enter the MIL pool")
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("MIL metadata contains duplicate video IDs")
    if set(row["binary_label"] for row in rows) != set(CLASSES):
        raise ValueError("MIL metadata must contain both binary classes")
    group_folds: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_folds[row["relation_group"]].add(row["fold"])
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A relation group crosses folds")
    return rows


def fold_rows(rows: list[dict[str, str]], validation_fold: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    validation = [row for row in rows if int(row["fold"]) == validation_fold]
    train = [row for row in rows if int(row["fold"]) != validation_fold]
    if {row["video_id"] for row in train} & {row["video_id"] for row in validation}:
        raise AssertionError("A video crosses train and validation")
    if set(row["binary_label"] for row in train) != set(CLASSES) or set(row["binary_label"] for row in validation) != set(CLASSES):
        raise ValueError(f"Fold {validation_fold} must contain both classes in train and validation")
    return train, validation


def clips_for_rows(rows: list[dict[str, str]], config: dict[str, object]) -> dict[str, list[ClipRecord]]:
    clips = {}
    for row in rows:
        records = build_clip_records(
            {**row, "style": row["binary_label"]}, probe_video(ROOT / Path(row["path"])),
            float(config["sampling_fps"]), int(config["clip_length"]), int(config["clip_stride"]),
        )
        if not records:
            raise ValueError(f"Video has no complete clips: {row['video_id']}")
        clips[row["video_id"]] = records
    return clips


def sampled_clips(clips: list[ClipRecord], maximum: int, seed: int, epoch: int,
                  video_id: str, training: bool) -> list[ClipRecord]:
    if maximum <= 0 or len(clips) <= maximum:
        return clips
    if training:
        indices = sorted(random.Random(f"{seed}:{epoch}:{video_id}").sample(range(len(clips)), maximum))
    else:
        indices = np.linspace(0, len(clips) - 1, maximum, dtype=int).tolist()
    return [clips[index] for index in indices]


class VideoBagDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], clips: dict[str, list[ClipRecord]], maximum: int,
                 seed: int, training: bool):
        self.rows, self.clips, self.maximum = rows, clips, maximum
        self.seed, self.training, self.epoch = seed, training, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, object]]:
        row = self.rows[index]
        selected = sampled_clips(self.clips[row["video_id"]], self.maximum, self.seed, self.epoch,
                                 row["video_id"], self.training)
        bag = torch.stack([transform_clip(decode_clip(ROOT / Path(record.source_path), record)) for record in selected])
        return bag, CLASSES.index(row["binary_label"]), {
            "video_id": row["video_id"], "records": [record.to_dict() for record in selected],
        }


def collate_bags(batch: list[tuple[torch.Tensor, int, dict[str, object]]]):
    bags, labels, metadata = zip(*batch)
    return list(bags), torch.tensor(labels, dtype=torch.float32), list(metadata)


def aggregate_bag_logits(clip_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if clip_logits.ndim != 2 or clip_logits.shape[1] != 2:
        raise ValueError("Expected clip logits with shape (clips, 2)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    clip_drowning_logits = clip_logits[:, 1] - clip_logits[:, 0]
    selected = torch.topk(clip_drowning_logits, min(top_k, len(clip_drowning_logits)))
    return selected.values.mean(), selected.indices


def bag_loss(clip_logits: torch.Tensor, target: torch.Tensor, top_k: int,
             positive_weight: torch.Tensor | None = None) -> torch.Tensor:
    bag_logit, _ = aggregate_bag_logits(clip_logits, top_k)
    return functional.binary_cross_entropy_with_logits(
        bag_logit.unsqueeze(0), target.float().reshape(1), pos_weight=positive_weight,
    )


def model_bag_logits(model: nn.Module, bag: torch.Tensor, device: torch.device,
                     clip_batch_size: int) -> torch.Tensor:
    return torch.cat([model(chunk.to(device, non_blocking=True)) for chunk in bag.split(clip_batch_size)])


def validate_no_clip_leakage(train: list[ClipRecord], validation: list[ClipRecord]) -> None:
    if {clip.video_id for clip in train} & {clip.video_id for clip in validation}:
        raise ValueError("Video leakage between train and validation")
    if {clip.clip_id for clip in train} & {clip.clip_id for clip in validation}:
        raise ValueError("Clip leakage between train and validation")


def make_loader(dataset: VideoBagDataset, config: dict[str, object], shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(dataset, batch_size=int(config["bag_batch_size"]), shuffle=shuffle,
                      num_workers=int(config["num_workers"]), collate_fn=collate_bags,
                      pin_memory=str(config["device"]).startswith("cuda"),
                      generator=torch.Generator().manual_seed(seed))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: dict[str, object],
             positive_weight: torch.Tensor) -> tuple[float, dict[str, object], list[dict[str, object]]]:
    model.eval()
    losses, labels, scores, predictions = [], [], [], []
    with torch.no_grad():
        for bags, targets, metadata in loader:
            for bag, target, details in zip(bags, targets, metadata):
                clip_logits = model_bag_logits(model, bag, device, int(config["clip_forward_batch_size"]))
                bag_logit, top_indices = aggregate_bag_logits(clip_logits, int(config["top_k"]))
                losses.append(float(functional.binary_cross_entropy_with_logits(
                    bag_logit.reshape(1), target.to(device).reshape(1), pos_weight=positive_weight,
                )))
                score = float(bag_logit.sigmoid())
                records = details["records"]
                ranked = sorted(top_indices.tolist(),
                                key=lambda index: float(clip_logits[index, 1] - clip_logits[index, 0]), reverse=True)
                predictions.append({
                    "video_id": details["video_id"], "weak_label": CLASSES[int(target)],
                    "drowning_behavior_score": score, "predicted_label": CLASSES[int(score >= 0.5)],
                    "top_clip_ids": "|".join(str(records[index]["clip_id"]) for index in ranked),
                    "top_clip_start_times_sec": "|".join(f"{float(records[index]['start_time']):.3f}" for index in ranked),
                    "top_clip_end_times_sec": "|".join(f"{float(records[index]['end_time']):.3f}" for index in ranked),
                })
                labels.append(int(target))
                scores.append(score)
    metrics = binary_video_metrics(labels, scores)
    metrics.update({"evaluation_label": WEAK_EVALUATION_LABEL,
                    "recall_warning": "Recall against weak candidate labels is not real drowning detection sensitivity."})
    return float(np.mean(losses)), metrics, predictions


def write_manifests(output: Path, rows: list[dict[str, str]], clips: dict[str, list[ClipRecord]],
                    fold_count: int) -> None:
    folds = []
    for fold in range(fold_count):
        train, validation = fold_rows(rows, fold)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_ids = {row["video_id"] for row in train}
        video_rows = [{"video_id": row["video_id"], "binary_label": row["binary_label"],
                       "label_strength": row["label_strength"], "relation_group": row["relation_group"],
                       "source_id": row["source_id"], "session_id": row["session_id"],
                       "split": "train" if row["video_id"] in train_ids else "validation"} for row in rows]
        clip_rows = [{**clip.to_dict(), "split": "train" if video_id in train_ids else "validation"}
                     for video_id, records in clips.items() for clip in records]
        train_clips = [clip for video_id, records in clips.items() if video_id in train_ids for clip in records]
        validation_clips = [clip for video_id, records in clips.items() if video_id not in train_ids for clip in records]
        validate_no_clip_leakage(train_clips, validation_clips)
        write_csv(fold_dir / "video_manifest.csv", video_rows)
        write_csv(fold_dir / "clip_manifest.csv", clip_rows)
        folds.append({"fold": fold, "train_videos": len(train), "validation_videos": len(validation),
                      "train_class_counts": dict(Counter(row["binary_label"] for row in train)),
                      "validation_class_counts": dict(Counter(row["binary_label"] for row in validation))})
    (output / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")


def run_fold(fold: int, rows: list[dict[str, str]], clips: dict[str, list[ClipRecord]],
             config: dict[str, object], output: Path, device: torch.device) -> dict[str, object]:
    train_rows, validation_rows = fold_rows(rows, fold)
    train_dataset = VideoBagDataset(train_rows, clips, int(config["max_clips_per_train_bag"]),
                                    int(config["seed"]) + fold, True)
    validation_dataset = VideoBagDataset(validation_rows, clips, int(config["max_clips_per_validation_bag"]),
                                         int(config["seed"]) + fold, False)
    train_loader = make_loader(train_dataset, config, True, int(config["seed"]) + fold)
    validation_loader = make_loader(validation_dataset, config, False, int(config["seed"]) + fold)
    seed_everything(int(config["seed"]) + fold)
    model = build_r3d18(2, bool(config["pretrained"]), bool(config["freeze_backbone"])).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                      lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    counts = Counter(row["binary_label"] for row in train_rows)
    positive_weight = torch.tensor([counts[CLASSES[0]] / counts[CLASSES[1]]], device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    fold_dir, history, best_loss = output / f"fold_{fold}", [], float("inf")
    checkpoint = fold_dir / "best_checkpoint.pt"
    for epoch in range(1, int(config["epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        if config["freeze_backbone"]:
            for module in model.children():
                if module is not model.fc:
                    module.eval()
        total_loss, video_count = 0.0, 0
        for bags, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                bag_logits = torch.stack([
                    aggregate_bag_logits(model_bag_logits(model, bag, device, int(config["clip_forward_batch_size"])),
                                         int(config["top_k"]))[0] for bag in bags
                ])
                loss = functional.binary_cross_entropy_with_logits(
                    bag_logits, targets.to(device), pos_weight=positive_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(bags)
            video_count += len(bags)
        validation_loss, metrics, predictions = evaluate(
            model, validation_loader, device, config, positive_weight,
        )
        record = {"epoch": epoch, "train_loss": total_loss / video_count, "validation_loss": validation_loss}
        history.append(record)
        print(json.dumps({"fold": fold, **record}), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(str(checkpoint), model, optimizer, epoch, CLASSES, config)
    best = load_checkpoint(str(checkpoint), model)
    validation_loss, metrics, predictions = evaluate(model, validation_loader, device, config, positive_weight)
    metrics.update({"fold": fold, "best_epoch": best["epoch"], "best_validation_loss": best_loss,
                    "final_validation_loss": validation_loss, "train_videos": len(train_rows),
                    "validation_videos": len(validation_rows), "positive_class_weight": float(positive_weight)})
    (fold_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(fold_dir / "training_log.csv", history)
    write_csv(fold_dir / "video_predictions.csv", predictions)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary R3D-18 with video-bag top-k MIL.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "r3d18_binary_drowning.yaml")
    parser.add_argument("--prepare-only", action="store_true", help="Write manifests without training or weights.")
    parser.add_argument("--fold", type=int, help="Train only one validation fold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = load_training_rows(ROOT / config["binary_metadata"])
    clips = clips_for_rows(rows, config)
    output = ROOT / config["mil_output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    write_manifests(output, rows, clips, int(config["cv_folds"]))
    plan = {"videos": len(rows), "classes": dict(Counter(row["binary_label"] for row in rows)),
            "evaluation_label": WEAK_EVALUATION_LABEL, "config": config}
    (output / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use device: cpu only for diagnostics.")
    device = torch.device(str(config["device"]))
    folds = [args.fold] if args.fold is not None else list(range(int(config["cv_folds"])))
    if any(fold not in range(int(config["cv_folds"])) for fold in folds):
        raise ValueError("Requested fold is outside configured cv_folds")
    results = [run_fold(fold, rows, clips, config, output, device) for fold in folds]
    report = {"evaluation_label": WEAK_EVALUATION_LABEL, "folds": results,
              "mean_accuracy": float(np.mean([result["accuracy"] for result in results])),
              "mean_macro_f1": float(np.mean([result["macro_f1"] for result in results])),
              "warning": "Scores are not calibrated risk probabilities; weak-label recall is not real drowning sensitivity."}
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
