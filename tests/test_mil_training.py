import csv
import tempfile
import unittest
from pathlib import Path

import torch

from src.video_classification.mil_train import (
    aggregate_bag_logits,
    bag_loss,
    load_training_rows,
    model_bag_logits,
    sampled_clips,
    seed_everything,
    validate_no_clip_leakage,
)
from src.video_classification.model import build_r3d18, load_checkpoint, save_checkpoint
from src.video_classification.preprocessing import ClipRecord


def clip(video_id: str, index: int) -> ClipRecord:
    return ClipRecord(video_id, f"{video_id}_C{index:04d}", "unused.mp4", index, index + 15,
                      float(index), float(index + 1), "normal_swimming", "train", 1)


class MilTrainingTests(unittest.TestCase):
    def test_top_k_bag_aggregation(self):
        bag_logit, indices = aggregate_bag_logits(torch.tensor([[0.0, -1.0], [0.0, 3.0], [0.0, 1.0]]), 2)
        self.assertEqual(indices.tolist(), [1, 2])
        self.assertEqual(float(bag_logit), 2.0)

    def test_negative_and_positive_bag_loss(self):
        low = torch.tensor([[0.0, -3.0], [0.0, -2.0]])
        high = torch.tensor([[0.0, 3.0], [0.0, 2.0]])
        self.assertGreater(bag_loss(high, torch.tensor(0), 1), bag_loss(low, torch.tensor(0), 1))
        self.assertLess(bag_loss(high, torch.tensor(1), 1), bag_loss(low, torch.tensor(1), 1))

    def test_video_and_clip_leakage_is_rejected(self):
        validate_no_clip_leakage([clip("A", 0)], [clip("B", 0)])
        with self.assertRaises(ValueError):
            validate_no_clip_leakage([clip("A", 0)], [clip("A", 1)])

    def test_exact_duplicate_cannot_enter_pool(self):
        rows = [
            {"video_id": "N", "initial_training_eligible": "True", "is_exact_duplicate": "False",
             "canonical_video_id": "N", "binary_label": "normal_swimming", "relation_group": "N", "fold": "0"},
            {"video_id": "D", "initial_training_eligible": "True", "is_exact_duplicate": "True",
             "canonical_video_id": "OTHER", "binary_label": "drowning_behavior", "relation_group": "D", "fold": "1"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.csv"
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                load_training_rows(path)

    def test_sampling_is_deterministic_for_seed_and_epoch(self):
        clips = [clip("V", index) for index in range(20)]
        first = sampled_clips(clips, 5, 42, 3, "V", True)
        second = sampled_clips(clips, 5, 42, 3, "V", True)
        self.assertEqual([item.clip_id for item in first], [item.clip_id for item in second])
        self.assertNotEqual([item.clip_id for item in first],
                            [item.clip_id for item in sampled_clips(clips, 5, 42, 4, "V", True)])

    def test_synthetic_forward_backward_cpu_and_available_cuda(self):
        for device in [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []):
            seed_everything(42)
            model = torch.nn.Linear(4, 2).to(device)
            logits = model_bag_logits(model, torch.randn(5, 4), device, 2)
            loss = bag_loss(logits, torch.tensor(1, device=device), 2)
            loss.backward()
            self.assertIsNotNone(model.weight.grad)

    def test_r3d18_two_class_mil_forward_backward_smoke(self):
        model = build_r3d18(2, pretrained=False, freeze_backbone=True)
        model.eval()
        logits = model_bag_logits(model, torch.randn(2, 3, 16, 32, 32), torch.device("cpu"), 1)
        loss = bag_loss(logits, torch.tensor(1), 2)
        loss.backward()
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertIsNotNone(model.fc.weight.grad)

    def test_mil_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(str(path), model, optimizer, 1, ["normal_swimming", "drowning_behavior"], {"seed": 42})
            loaded_model = torch.nn.Linear(4, 2)
            self.assertEqual(load_checkpoint(str(path), loaded_model)["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
