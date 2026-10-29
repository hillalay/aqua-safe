import json
import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
import time
from tqdm.auto import tqdm
from .model import build_r3d18
from .yolo_dataset import make_loaders
from .evaluate import classification_metrics
from src.video_review.inventory import sha256_file


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/aquasafe_yolo_r3d18_v1/training_weight2_epoch5"

EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Continue the existing checkpoint; no existing experiment is overwritten.")
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--drowning-weight', type=float, default=2 * (55 / 22))
    parser.add_argument('--init-checkpoint', type=Path,
                        default=ROOT / 'results/aquasafe_yolo_r3d18_v1/training/best_model.pt')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.epochs < 1 or not math.isfinite(args.drowning_weight) or args.drowning_weight <= 0:
        parser.error('Epochs and drowning weight must be positive and finite')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('This continuation experiment requires CUDA')
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = ROOT / "results/aquasafe_yolo_r3d18_v1/clip_manifest.csv"

    loaders = make_loaders(
        ROOT,
        manifest,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_r3d18(
        num_classes=2,
        pretrained=False,
        freeze_backbone=True,
    ).to(device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu', weights_only=True), strict=True)
    (args.output_dir / 'experiment_config.json').write_text(json.dumps({
        'epochs': args.epochs, 'class_weights': [1.0, args.drowning_weight],
        'previous_class_weights': [1.0, 55 / 22],
        'init_checkpoint': str(args.init_checkpoint), 'init_checkpoint_sha256': sha256_file(args.init_checkpoint),
        'manifest_sha256': sha256_file(manifest), 'optimizer': 'new AdamW; previous optimizer state was not saved',
        'learning_rate': LEARNING_RATE, 'checkpoint_selection': 'minimum weighted validation loss',
        'supervision': 'weak_video_label; clip_event_label=unknown',
        'training_inputs': 'annotated full-video clips; not person crops',
        'freeze_backbone': True, 'batchnorm_policy': 'unchanged from original model.train() behavior',
    }, indent=2), encoding='utf-8')

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, args.drowning_weight], device=device)
    )

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0

        for clips, labels, _ in tqdm(
            loaders["train"],
            desc=f"Epoch {epoch}/{args.epochs} - Training",
            unit="batch",
        ):
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(clips)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_targets, val_predictions = [], []

        with torch.no_grad():
            for clips, labels, _ in tqdm(
                loaders["validation"],
                desc=f"Epoch {epoch}/{args.epochs} - Validation",
                unit="batch",
            ):
                clips = clips.to(device)
                labels = labels.to(device)

                logits = model(clips)
                loss = criterion(logits, labels)

                val_loss += loss.item()

                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                val_targets.extend(labels.cpu().tolist())
                val_predictions.extend(predictions.cpu().tolist())

        train_loss /= len(loaders["train"])
        val_loss /= len(loaders["validation"])
        val_accuracy = correct / total
        metrics = classification_metrics(val_targets, val_predictions, 2)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Drowning Recall: {metrics['recall'][1]:.4f} | Drowning F1: {metrics['f1'][1]:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_drowning_recall": metrics['recall'][1],
            "val_drowning_precision": metrics['precision'][1],
            "val_drowning_f1": metrics['f1'][1],
            "val_confusion_matrix": metrics['confusion_matrix'],
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                args.output_dir / "best_model.pt",
            )

    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("Eğitim tamamlandı!")
    print("Model:", args.output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
