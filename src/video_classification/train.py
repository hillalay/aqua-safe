from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import VideoClipDataset, class_counts, duplicate_groups, exclusion_reason, load_metadata, validate_split_leakage
from .evaluate import aggregate_video_predictions, classification_metrics
from .model import build_r3d18, load_checkpoint, save_checkpoint
from .preprocessing import build_clip_records, probe_video


ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_inventory(rows: list[dict[str, str]], output: Path) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    inventory, eligible = [], []
    for row in rows:
        path = ROOT / Path(row["path"])
        reason = exclusion_reason(row)
        try:
            info = probe_video(path)
            record = {**row, "fps": info.fps, "frame_count": info.frame_count, "width": info.width,
                      "height": info.height, "decoded_duration_sec": info.duration_sec,
                      "duration_delta_sec": abs(info.duration_sec - float(row["duration_sec"])),
                      "status": "ok", "training_eligible": reason is None, "exclusion_reason": reason or ""}
            if reason is None:
                eligible.append(row)
        except RuntimeError as error:
            record = {**row, "status": "error", "training_eligible": False, "exclusion_reason": str(error)}
        inventory.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(inventory[0])
    with output.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    return inventory, eligible


def one_batch_smoke(config: dict[str, object], rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    classes = sorted({row["style"] for row in rows})
    row = rows[0]
    info = probe_video(ROOT / Path(row["path"]))
    clips = build_clip_records(row, info, config["sampling_fps"], config["clip_length"], config["stride"])
    if not clips:
        raise RuntimeError("Smoke video is too short for one clip")
    dataset = VideoClipDataset(ROOT, clips[:1], classes)
    write_clip_manifest(clips[:1], output_dir / "smoke_clip_manifest.csv")
    inputs, labels, metadata = next(iter(DataLoader(dataset, batch_size=1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_r3d18(len(classes), pretrained=True, freeze_backbone=True).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config["learning_rate"])
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(inputs.to(device))
        loss = nn.CrossEntropyLoss()(logits, labels.to(device))
    loss.backward()
    optimizer.step()
    checkpoint = output_dir / "smoke_checkpoint.pt"
    save_checkpoint(str(checkpoint), model, optimizer, 1, classes, config)
    return {"mode": "smoke_test", "device": str(device), "clip_count": 1,
            "tensor_shape": list(inputs.shape), "logits_shape": list(logits.shape),
            "loss": float(loss.detach().cpu()), "video_id": metadata["video_id"][0],
            "checkpoint": str(checkpoint.relative_to(ROOT))}


def write_clip_manifest(clips: list, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(clips[0].to_dict()))
        writer.writeheader()
        writer.writerows(clip.to_dict() for clip in clips)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], list[str], np.ndarray]:
    labels, predictions, video_ids, probabilities = [], [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets, metadata in loader:
            logits = model(inputs.to(device))
            batch_probabilities = logits.softmax(dim=1).cpu().numpy()
            labels.extend(targets.tolist())
            predictions.extend(batch_probabilities.argmax(axis=1).tolist())
            probabilities.extend(batch_probabilities)
            video_ids.extend(metadata["video_id"])
    return labels, predictions, video_ids, np.asarray(probabilities)


def train_model(config: dict[str, object], rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    classes = sorted({row["style"] for row in rows})
    clips = []
    for row in rows:
        clips.extend(build_clip_records(row, probe_video(ROOT / Path(row["path"])), config["sampling_fps"],
                                        config["clip_length"], config["stride"]))
    write_clip_manifest(clips, output_dir / "clip_manifest.csv")
    by_split = {split: [clip for clip in clips if clip.split == split] for split in ("train", "validation", "test")}
    if not by_split["train"] or not by_split["validation"]:
        raise RuntimeError("Training requires non-empty train and validation clips")
    datasets = {split: VideoClipDataset(ROOT, split_clips, classes) for split, split_clips in by_split.items()}
    loaders = {
        split: DataLoader(dataset, batch_size=config["batch_size"], shuffle=split == "train", num_workers=0)
        for split, dataset in datasets.items() if len(dataset)
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_r3d18(len(classes), pretrained=True, freeze_backbone=config["freeze_backbone"]).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss, stale_epochs, history = float("inf"), 0, []
    checkpoint = output_dir / "best_checkpoint.pt"
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        if config["freeze_backbone"]:
            for module in model.children():
                if module is not model.fc:
                    module.eval()
        train_loss = 0.0
        for inputs, targets, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(inputs.to(device)), targets.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * len(inputs)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in loaders["validation"]:
                validation_loss += float(criterion(model(inputs.to(device)), targets.to(device))) * len(inputs)
        train_loss /= len(datasets["train"])
        validation_loss /= len(datasets["validation"])
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            save_checkpoint(str(checkpoint), model, optimizer, epoch, classes, config)
        else:
            stale_epochs += 1
            if stale_epochs >= config["early_stopping_patience"]:
                break
    load_checkpoint(str(checkpoint), model)
    evaluation_split = "test" if "test" in loaders else "validation"
    labels, predictions, video_ids, probabilities = predict(model, loaders[evaluation_split], device)
    metrics = classification_metrics(labels, predictions, len(classes))
    video_predictions = aggregate_video_predictions(video_ids, probabilities)
    video_labels = {}
    for video_id, label in zip(video_ids, labels):
        video_labels.setdefault(video_id, label)
    metrics["video_predictions"] = video_predictions
    metrics["video_level"] = classification_metrics(
        [video_labels[video_id] for video_id in sorted(video_predictions)],
        [video_predictions[video_id] for video_id in sorted(video_predictions)], len(classes)
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"mode": "trained", "device": str(device), "clip_count": len(clips),
            "clips_by_split": {key: len(value) for key, value in by_split.items()},
            "epochs_completed": len(history), "evaluation_split": evaluation_split, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "r3d18_pilot.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_metadata(ROOT / config["metadata"])
    validate_split_leakage(rows)
    inventory, eligible = prepare_inventory(rows, output_dir / "video_inventory.csv")
    duplicates = duplicate_groups([ROOT / Path(row["path"]) for row in rows])
    valid_splits = {row["split"] for row in eligible}
    can_train = bool(eligible) and {"train", "validation"}.issubset(valid_splits)
    result = one_batch_smoke(config, rows, output_dir) if not can_train else train_model(config, eligible, output_dir)
    raw_paths = list((ROOT / "data" / "raw").glob("*/*.mp4"))
    report = {
        "raw_videos_found": len(raw_paths), "raw_class_counts": class_counts([{"style": path.parent.name} for path in raw_paths]),
        "metadata_videos": len(rows), "decoded_videos": sum(row["status"] == "ok" for row in inventory),
        "training_eligible_videos": len(eligible), "class_counts": class_counts(rows),
        "eligible_class_counts": class_counts(eligible), "exact_duplicate_groups": duplicates,
        "split_available": can_train, "result": result,
        "limitations": ["All current labels are pending", "Swimmer/session identities are unknown",
                        "No independent train/validation/test claim is possible",
                        "Different cuts of the same source cannot be ruled out without source/session IDs"],
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = f"""# AquaMed R3D-18 pilot report

- Raw videos: **{report['raw_videos_found']}** ({report['raw_class_counts']})
- Videos in selected metadata: **{report['metadata_videos']}**
- Successfully decoded: **{report['decoded_videos']}**
- Training-eligible: **{report['training_eligible_videos']}**
- Selected class distribution: **{report['class_counts']}**
- Exact duplicate groups: **{len(report['exact_duplicate_groups'])}**
- Valid train/validation split: **{report['split_available']}**
- Execution mode: **{report['result']['mode']}**
- Device: **{report['result'].get('device', 'not run')}**
- Generated clips: **{report['result'].get('clip_count', 0)}**
- Tensor shape: **{report['result'].get('tensor_shape', 'not run')}**
- Checkpoint: **{report['result'].get('checkpoint', 'not run')}**

No classification accuracy is reported: every current style label is pending,
swimmer/session IDs are unknown, and no leakage-safe validation split exists.
This is an infrastructure smoke test, not a trained model result.

Before drowning-behavior classification, person-level labels, normal/abnormal
temporal annotations, source/session grouping, multi-swimmer identity tracking,
and a leakage-safe independent evaluation set are required.
"""
    (output_dir / "pilot_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
