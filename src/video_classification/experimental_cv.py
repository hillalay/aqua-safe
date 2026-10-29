from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.video_review.reviews import merge_reviews

from .dataset import VideoClipDataset
from .evaluate import aggregate_video_predictions, classification_metrics
from .model import build_r3d18, load_checkpoint, save_checkpoint
from .preprocessing import ClipRecord, build_clip_records, probe_video


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["backstroke", "breaststroke", "butterfly", "freestyle"]
CANDIDATE_GROUPS = {
    "G1": {"V009", "V011", "V012", "V018", "V020"},
    "G2": {"V007", "V008"},
    "G3": {"V030", "V032"},
    "N1020": {"R3D1EA2AC4B", "RD2D067EF06", "RCB1E3886CD"},
    "N1022_DHASH1": {"V005", "R375838EEF1", "R41D162F0AA", "RC02CD33BA5", "R077828F5CD", "RDEC0D28E9D"},
    "N1026": {"R4836BE4024", "R61131A34C7", "R1E340E9BEC", "RB4F689CD33", "R72B067E520",
              "RD778B333B7", "R9D94E1F1CF", "R1C1672B327", "R4E00C441C0", "R27F591CA08",
              "R3A4373760E", "R37C5C0048E", "R935221AE57", "R42D619016F", "R6A351D6B04"},
}
RESULT_LABEL = "experimental_not_independent_test"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    kept, duplicates = [], []
    for digest, group in sorted(by_hash.items()):
        ordered = sorted(group, key=lambda row: row["video_id"])
        kept.append(ordered[0])
        if len(ordered) > 1:
            duplicates.append({"sha256": digest, "kept_video_id": ordered[0]["video_id"],
                               "dropped_video_ids": [row["video_id"] for row in ordered[1:]]})
    return sorted(kept, key=lambda row: row["video_id"]), duplicates


def load_verified_rows(config: dict[str, object]) -> tuple[list[dict[str, str]], str, list[dict[str, object]]]:
    state_path = ROOT / str(config["review_state"])
    before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    master = read_csv(ROOT / str(config["master_metadata"]))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = [row for row in merge_reviews(master, state)
            if (row.get("video_style_verification") == "verified" or row.get("review_status") == "verified")
            and (not config.get("require_single_swimmer") or row.get("swimmer_count") == "single")
            and row.get("mixed_styles") == "no"
            and row.get("style") in CLASSES]
    if hashlib.sha256(state_path.read_bytes()).hexdigest() != before:
        raise RuntimeError("review_state.json changed while metadata was read")
    rows, duplicates = deduplicate_rows(rows)
    expected = int(config.get("expected_videos", len(rows)))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} verified deduplicated videos, found {len(rows)}")
    for row in rows:
        row["style_verified"] = "verified"
        row["swimmer_id"] = "unknown"
        row["split"] = "experimental_cv"
        row["evaluation_label"] = RESULT_LABEL
    return rows, before, duplicates


def cv_group(video_id: str) -> str:
    for name, members in CANDIDATE_GROUPS.items():
        if video_id in members:
            return name
    return f"video:{video_id}"


def make_folds(rows: list[dict[str, str]], fold_count: int, seed: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[cv_group(row["video_id"])].append(row)
    minimum_groups = min(sum(any(row["style"] == style for row in bundle) for bundle in grouped.values())
                         for style in CLASSES)
    actual_folds = min(fold_count, minimum_groups)
    if actual_folds < 2:
        raise ValueError("At least two class-preserving folds are required")

    rng = random.Random(seed)
    validation: list[list[str]] = [[] for _ in range(actual_folds)]
    counts = [Counter() for _ in range(actual_folds)]
    totals = Counter(row["style"] for row in rows)
    bundles = list(grouped.values())
    rng.shuffle(bundles)
    bundles.sort(key=len, reverse=True)
    for bundle in bundles:
        bundle_counts = Counter(row["style"] for row in bundle)
        scores = [sum((counts[index][style] + amount) / totals[style]
                      for style, amount in bundle_counts.items()) for index in range(actual_folds)]
        choices = [index for index, score in enumerate(scores) if score == min(scores)]
        fold = rng.choice(choices)
        validation[fold].extend(row["video_id"] for row in bundle)
        counts[fold].update(bundle_counts)

    all_ids = {row["video_id"] for row in rows}
    folds = []
    for index, validation_ids in enumerate(validation, 1):
        validation_set = set(validation_ids)
        folds.append({"fold": index, "train_ids": sorted(all_ids - validation_set),
                      "validation_ids": sorted(validation_set)})
    return folds


def validate_folds(rows: list[dict[str, str]], folds: list[dict[str, object]]) -> dict[str, object]:
    all_ids = {row["video_id"] for row in rows}
    style_by_id = {row["video_id"]: row["style"] for row in rows}
    validation_seen: list[str] = []
    distributions = []
    for fold in folds:
        train, validation = set(fold["train_ids"]), set(fold["validation_ids"])
        if train & validation or train | validation != all_ids:
            raise AssertionError(f"Fold {fold['fold']} has video leakage or missing videos")
        validation_seen.extend(validation)
        counts = Counter(style_by_id[video_id] for video_id in validation)
        if set(counts) != set(CLASSES):
            raise AssertionError(f"Fold {fold['fold']} does not contain every class: {counts}")
        distributions.append({"fold": fold["fold"], **{name: counts[name] for name in CLASSES}})
    if Counter(validation_seen) != Counter({video_id: 1 for video_id in all_ids}):
        raise AssertionError("Each video must be validation data exactly once")
    for name, members in CANDIDATE_GROUPS.items():
        active_members = members & all_ids
        if not active_members:
            continue
        assigned = [fold["fold"] for fold in folds if active_members & set(fold["validation_ids"])]
        if len(assigned) != 1 or not active_members <= set(folds[assigned[0] - 1]["validation_ids"]):
            raise AssertionError(f"Candidate group {name} crosses folds")
    return {"status": "passed", "videos": len(rows), "folds": len(folds),
            "validation_class_distribution": distributions,
            "candidate_groups": {name: sorted(members & all_ids) for name, members in CANDIDATE_GROUPS.items()
                                 if members & all_ids},
            "independent_test": False, "evaluation_label": RESULT_LABEL}


def build_capped_clips(rows: list[dict[str, str]], config: dict[str, object]) -> tuple[list[ClipRecord], dict[str, int]]:
    selected, available = [], {}
    maximum = int(config["max_clips_per_video"])
    seed = int(config["seed"])
    for row in rows:
        clips = build_clip_records(row, probe_video(ROOT / Path(row["path"])), float(config["sampling_fps"]),
                                   int(config["clip_length"]), int(config["stride"]))
        available[row["video_id"]] = len(clips)
        if len(clips) > maximum:
            clips = sorted(random.Random(f"{seed}:{row['video_id']}").sample(clips, maximum),
                           key=lambda clip: clip.start_frame)
        selected.extend(clips)
    return selected, available


def metrics_with_classes(labels: list[int], predictions: list[int]) -> dict[str, object]:
    metrics = classification_metrics(labels, predictions, len(CLASSES))
    metrics["per_class"] = {
        name: {"precision": metrics["precision"][index], "recall": metrics["recall"][index],
               "support": metrics["support"][index]}
        for index, name in enumerate(CLASSES)
    }
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             train_clips: list[ClipRecord], seed: int) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    labels, predictions, video_ids, probabilities, clip_ids = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets, metadata in loader:
            batch_probabilities = model(inputs.to(device)).softmax(dim=1).cpu().numpy()
            labels.extend(targets.tolist())
            predictions.extend(batch_probabilities.argmax(axis=1).tolist())
            probabilities.extend(batch_probabilities)
            video_ids.extend(metadata["video_id"])
            clip_ids.extend(metadata["clip_id"])
    probabilities_array = np.asarray(probabilities)
    clip_rows = []
    for clip_id, video_id, label, prediction, probability in zip(clip_ids, video_ids, labels, predictions, probabilities):
        clip_rows.append({"clip_id": clip_id, "video_id": video_id, "actual": CLASSES[label],
                          "predicted": CLASSES[prediction],
                          **{f"probability_{name}": float(probability[index]) for index, name in enumerate(CLASSES)}})

    video_predictions = aggregate_video_predictions(video_ids, probabilities_array)
    video_labels = {video_id: label for video_id, label in zip(video_ids, labels)}
    sorted_ids = sorted(video_predictions)
    video_actual = [video_labels[video_id] for video_id in sorted_ids]
    video_predicted = [video_predictions[video_id] for video_id in sorted_ids]
    video_rows = []
    for video_id in sorted_ids:
        mask = [index for index, value in enumerate(video_ids) if value == video_id]
        mean_probability = probabilities_array[mask].mean(axis=0)
        video_rows.append({"video_id": video_id, "actual": CLASSES[video_labels[video_id]],
                           "predicted": CLASSES[video_predictions[video_id]], "clip_count": len(mask),
                           **{f"probability_{name}": float(mean_probability[index]) for index, name in enumerate(CLASSES)}})

    train_majority = Counter(clip.style for clip in train_clips).most_common(1)[0][0]
    majority_index = CLASSES.index(train_majority)
    random_predictions = np.random.default_rng(seed).integers(0, len(CLASSES), size=len(video_actual)).tolist()
    metrics = {
        "evaluation_label": RESULT_LABEL,
        "clip_level": metrics_with_classes(labels, predictions),
        "video_level": metrics_with_classes(video_actual, video_predicted),
        "video_level_baselines": {
            "majority_class": {"predicted_class": train_majority,
                               **metrics_with_classes(video_actual, [majority_index] * len(video_actual))},
            "uniform_random_seeded": {"seed": seed, **metrics_with_classes(video_actual, random_predictions)},
        },
    }
    return metrics, clip_rows, video_rows


def confusion_rows(matrix: list[list[int]]) -> list[dict[str, object]]:
    return [{"actual": name, **{predicted: matrix[row][column] for column, predicted in enumerate(CLASSES)}}
            for row, name in enumerate(CLASSES)]


def run_fold(fold: dict[str, object], rows: list[dict[str, str]], clips: list[ClipRecord],
             config: dict[str, object], output_dir: Path, device: torch.device,
             tensor_cache: dict[str, torch.Tensor]) -> dict[str, object]:
    fold_number = int(fold["fold"])
    fold_dir = output_dir / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ids, validation_ids = set(fold["train_ids"]), set(fold["validation_ids"])
    train_clips = [clip for clip in clips if clip.video_id in train_ids]
    validation_clips = [clip for clip in clips if clip.video_id in validation_ids]
    manifest = [{**clip.to_dict(), "fold": fold_number,
                 "split": "train" if clip.video_id in train_ids else "validation"} for clip in train_clips + validation_clips]
    write_csv(fold_dir / "clip_manifest.csv", manifest)
    video_manifest = [{"fold": fold_number, "video_id": row["video_id"], "style": row["style"],
                       "cv_constraint_group": cv_group(row["video_id"]),
                       "split": "train" if row["video_id"] in train_ids else "validation"}
                      for row in rows]
    write_csv(fold_dir / "video_manifest.csv", video_manifest)
    if {clip.video_id for clip in train_clips} & {clip.video_id for clip in validation_clips}:
        raise AssertionError("Clip split leaked a video")

    datasets = {"train": VideoClipDataset(ROOT, train_clips, CLASSES, tensor_cache),
                "validation": VideoClipDataset(ROOT, validation_clips, CLASSES, tensor_cache)}
    generator = torch.Generator().manual_seed(int(config["seed"]) + fold_number)
    sampler = None
    if config.get("balance_training"):
        counts = Counter(clip.style for clip in train_clips)
        weights = [1.0 / counts[clip.style] for clip in train_clips]
        sampler = WeightedRandomSampler(weights, len(train_clips), replacement=True, generator=generator)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=int(config["batch_size"]), sampler=sampler,
                            shuffle=sampler is None, num_workers=0, pin_memory=device.type == "cuda", generator=generator),
        "validation": DataLoader(datasets["validation"], batch_size=int(config["batch_size"]), shuffle=False,
                                 num_workers=0, pin_memory=device.type == "cuda"),
    }

    seed_everything(int(config["seed"]) + fold_number)
    model = build_r3d18(len(CLASSES), pretrained=bool(config.get("pretrained", True)),
                        freeze_backbone=bool(config["freeze_backbone"])).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                      lr=float(config["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = fold_dir / "best_checkpoint.pt"
    best_loss, history = float("inf"), []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        if config["freeze_backbone"]:
            for module in model.children():
                if module is not model.fc:
                    module.eval()
        train_loss = 0.0
        for inputs, targets, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(inputs.to(device, non_blocking=True)), targets.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * len(inputs)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in loaders["validation"]:
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(inputs.to(device, non_blocking=True)), targets.to(device, non_blocking=True))
                validation_loss += float(loss.detach()) * len(inputs)
        record = {"epoch": epoch, "training_loss": train_loss / len(train_clips),
                  "validation_loss": validation_loss / len(validation_clips)}
        history.append(record)
        print(json.dumps({"fold": fold_number, **record}), flush=True)
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            save_checkpoint(str(checkpoint), model, optimizer, epoch, CLASSES, config)

    best = load_checkpoint(str(checkpoint), model)
    metrics, clip_predictions, video_predictions = evaluate(
        model, loaders["validation"], device, train_clips, int(config["seed"]) + 1000 + fold_number)
    elapsed = time.perf_counter() - started
    metrics.update({"fold": fold_number, "epochs_completed": len(history), "best_epoch": best["epoch"],
                    "final_training_loss": history[-1]["training_loss"],
                    "final_validation_loss": history[-1]["validation_loss"],
                    "best_validation_loss": min(row["validation_loss"] for row in history),
                    "elapsed_seconds": elapsed, "device": str(device),
                    "cuda_peak_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0,
                    "train_clips": len(train_clips), "validation_clips": len(validation_clips)})
    (fold_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_csv(fold_dir / "training_log.csv", history)
    write_csv(fold_dir / "clip_predictions.csv", clip_predictions)
    write_csv(fold_dir / "video_predictions.csv", video_predictions)
    write_csv(fold_dir / "confusion_matrix_clip.csv", confusion_rows(metrics["clip_level"]["confusion_matrix"]))
    write_csv(fold_dir / "confusion_matrix_video.csv", confusion_rows(metrics["video_level"]["confusion_matrix"]))
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def mean_metric(folds: list[dict[str, object]], level: str, name: str) -> float:
    return float(np.mean([fold[level][name] for fold in folds]))


def mean_per_class(folds: list[dict[str, object]], level: str) -> dict[str, dict[str, float]]:
    return {name: {
        metric: float(np.mean([fold[level]["per_class"][name][metric] for fold in folds]))
        for metric in ("precision", "recall")
    } for name in CLASSES}


def summed_confusion(folds: list[dict[str, object]], level: str) -> list[list[int]]:
    return np.sum([fold[level]["confusion_matrix"] for fold in folds], axis=0).astype(int).tolist()


def write_report(output: Path, config: dict[str, object], checks: dict[str, object],
                 fold_metrics: list[dict[str, object]], available: dict[str, int], selected: list[ClipRecord],
                 state_hash: str, elapsed: float, device: torch.device) -> None:
    failed_attempts_path = output / "failed_attempts.json"
    failed_attempts = json.loads(failed_attempts_path.read_text(encoding="utf-8")) if failed_attempts_path.exists() else []
    clip_confusion = summed_confusion(fold_metrics, "clip_level")
    video_confusion = summed_confusion(fold_metrics, "video_level")
    write_csv(output / "confusion_matrix_clip_all_folds.csv", confusion_rows(clip_confusion))
    write_csv(output / "confusion_matrix_video_all_folds.csv", confusion_rows(video_confusion))
    summary = {
        "evaluation_label": RESULT_LABEL, "independent_test": False, "folds_trained": len(fold_metrics),
        "epochs_completed": sum(int(fold["epochs_completed"]) for fold in fold_metrics),
        "epochs_per_fold": [fold["epochs_completed"] for fold in fold_metrics], "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "elapsed_seconds": elapsed, "videos": checks["videos"], "available_clips": sum(available.values()),
        "selected_capped_clips": len(selected), "clip_selection_seed": config["seed"],
        "training_sampling_seed_rule": "seed + fold", "review_state_sha256": state_hash,
        "mean_clip_level": {name: mean_metric(fold_metrics, "clip_level", name) for name in ("accuracy", "macro_f1")},
        "mean_video_level": {name: mean_metric(fold_metrics, "video_level", name) for name in ("accuracy", "macro_f1")},
        "mean_per_class_clip_level": mean_per_class(fold_metrics, "clip_level"),
        "mean_per_class_video_level": mean_per_class(fold_metrics, "video_level"),
        "summed_confusion_matrix_clip_level": clip_confusion,
        "summed_confusion_matrix_video_level": video_confusion,
        "mean_video_level_baselines": {
            baseline: {metric: float(np.mean([fold["video_level_baselines"][baseline][metric]
                                              for fold in fold_metrics]))
                       for metric in ("accuracy", "macro_f1")}
            for baseline in ("majority_class", "uniform_random_seeded")
        },
        "mean_losses": {
            "final_training": float(np.mean([fold["final_training_loss"] for fold in fold_metrics])),
            "final_validation": float(np.mean([fold["final_validation_loss"] for fold in fold_metrics])),
            "best_validation": float(np.mean([fold["best_validation_loss"] for fold in fold_metrics])),
        },
        "fold_metrics": fold_metrics,
        "failed_attempts": failed_attempts,
        "limitations": ["Source and session identities are unknown.",
                        "This is experimental video-level cross-validation, not an independent test.",
                        "Candidate groups are visual grouping constraints, not verified source identities.",
                        "The results do not measure real-world drowning detection."],
    }
    (output / "experiment_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fold_lines = "\n".join(
        f"| {fold['fold']} | {fold['epochs_completed']} | {fold['final_training_loss']:.4f} | "
        f"{fold['final_validation_loss']:.4f} | {fold['clip_level']['accuracy']:.4f} | "
        f"{fold['clip_level']['macro_f1']:.4f} | {fold['video_level']['accuracy']:.4f} | "
        f"{fold['video_level']['macro_f1']:.4f} | {fold['elapsed_seconds']:.1f} |"
        for fold in fold_metrics)
    class_lines = "\n".join(
        f"| {name} | {summary['mean_per_class_clip_level'][name]['precision']:.4f} | "
        f"{summary['mean_per_class_clip_level'][name]['recall']:.4f} | "
        f"{summary['mean_per_class_video_level'][name]['precision']:.4f} | "
        f"{summary['mean_per_class_video_level'][name]['recall']:.4f} |" for name in CLASSES)
    report = f"""# AquaMed R3D-18 experimental video-level CV

**Evaluation label: `{RESULT_LABEL}`. This is not an independent test.**

- Videos: **{checks['videos']}**; folds trained: **{len(fold_metrics)}**
- Epochs completed: **{summary['epochs_completed']}** ({config['epochs']} per fold)
- Device: **{summary['gpu'] or device}**; CUDA used: **{device.type == 'cuda'}**
- Runtime: **{elapsed:.1f} seconds**
- Clips: **{sum(available.values())} available**, **{len(selected)} selected** after a per-video cap of {config['max_clips_per_video']}
- Clip selection seed: **{config['seed']}**; balanced training sampler seed: **seed + fold**
- Mean clip accuracy / macro-F1: **{summary['mean_clip_level']['accuracy']:.4f} / {summary['mean_clip_level']['macro_f1']:.4f}**
- Mean video accuracy / macro-F1: **{summary['mean_video_level']['accuracy']:.4f} / {summary['mean_video_level']['macro_f1']:.4f}**
- Mean final train / validation loss: **{summary['mean_losses']['final_training']:.4f} / {summary['mean_losses']['final_validation']:.4f}**
- Mean video majority baseline accuracy / macro-F1: **{summary['mean_video_level_baselines']['majority_class']['accuracy']:.4f} / {summary['mean_video_level_baselines']['majority_class']['macro_f1']:.4f}**
- Mean video seeded-random baseline accuracy / macro-F1: **{summary['mean_video_level_baselines']['uniform_random_seeded']['accuracy']:.4f} / {summary['mean_video_level_baselines']['uniform_random_seeded']['macro_f1']:.4f}**

| Fold | Epochs | Final train loss | Final val loss | Clip accuracy | Clip macro-F1 | Video accuracy | Video macro-F1 | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{fold_lines}

| Class | Mean clip precision | Mean clip recall | Mean video precision | Mean video recall |
|---|---:|---:|---:|---:|
{class_lines}

Per-class precision/recall, confusion matrices, seeded random and training-majority baselines are in each fold's `metrics.json` and CSV files. Validation clips keep the capped natural distribution; only training uses balanced sampling.

## Failed attempts

{chr(10).join(f"- {attempt['reason']}" for attempt in failed_attempts) or '- None'}

## Limitations

Real `source_id` and `session_id` values remain unknown and were not invented. G1/G2/G3 are only conservative visual grouping constraints. These results are experimental video-level cross-validation and must not be presented as independent generalization or real-world drowning-detection performance.
"""
    (output / "experiment_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "r3d18_verified.yaml")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    output = ROOT / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows, state_hash, duplicates = load_verified_rows(config)
    for row in rows:
        row["cv_constraint_group"] = cv_group(row["video_id"])
    write_csv(ROOT / str(config["metadata"]), rows)
    folds = make_folds(rows, int(config["cv_folds"]), int(config["seed"]))
    checks = validate_folds(rows, folds)
    checks["requested_folds"] = int(config["cv_folds"])
    checks["fold_reduction_reason"] = None if len(folds) == int(config["cv_folds"]) else "Insufficient groups in the rarest class"
    (output / "split_leakage_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (output / "deduplication.json").write_text(json.dumps({"groups": duplicates,
                                                             "dropped_videos": sum(len(group["dropped_video_ids"])
                                                                                   for group in duplicates)}, indent=2),
                                                   encoding="utf-8")
    (output / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    (output / "model_config.json").write_text(json.dumps({**config, "model": "torchvision.models.video.r3d_18",
                                                           "weights": "R3D_18_Weights.DEFAULT",
                                                           "optimizer": "AdamW", "input_size": [112, 112],
                                                           "fold_initialization": "independent_pretrained_model"}, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)
    if args.check_only:
        return
    if args.summarize_existing:
        fold_metrics = []
        for fold in folds:
            fold_dir = output / f"fold_{fold['fold']}"
            metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
            history = json.loads((fold_dir / "training_log.json").read_text(encoding="utf-8"))
            metrics.update({"final_training_loss": history[-1]["training_loss"],
                            "final_validation_loss": history[-1]["validation_loss"],
                            "best_validation_loss": min(row["validation_loss"] for row in history)})
            (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            fold_metrics.append(metrics)
        selection = read_csv(output / "clip_selection.csv")
        available = {row["video_id"]: int(row["available_clips"]) for row in selection}
        selected_count = sum(int(row["selected_clips"]) for row in selection)
        elapsed = sum(float(fold["elapsed_seconds"]) for fold in fold_metrics)
        write_report(output, config, checks, fold_metrics, available, [None] * selected_count,
                     state_hash, elapsed, torch.device(str(fold_metrics[0]["device"])))
        return
    clips, available = build_capped_clips(rows, config)
    clip_selection = [{"video_id": row["video_id"], "style": row["style"],
                       "available_clips": available[row["video_id"]],
                       "selected_clips": sum(clip.video_id == row["video_id"] for clip in clips)} for row in rows]
    write_csv(output / "clip_selection.csv", clip_selection)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    tensor_cache: dict[str, torch.Tensor] = {}
    metrics = [run_fold(fold, rows, clips, config, output, device, tensor_cache) for fold in folds]
    elapsed = time.perf_counter() - started
    write_report(output, config, checks, metrics, available, clips, state_hash, elapsed, device)


if __name__ == "__main__":
    main()
