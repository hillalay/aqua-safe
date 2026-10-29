import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.video_classification.binary_drowning import prepare_rows
from src.video_classification.dataset import VideoClipDataset, exclusion_reason, load_metadata, validate_split_leakage
from src.video_classification.evaluate import binary_video_metrics
from src.video_classification.experimental_cv import CANDIDATE_GROUPS, CLASSES, deduplicate_rows, make_folds, validate_folds
from src.video_classification.model import build_r3d18, load_checkpoint, save_checkpoint
from src.video_classification.preprocessing import R3D_MEAN, build_clip_records, decode_clip, probe_video, transform_clip


class VideoClassificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "sample.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 48))
        for index in range(20):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 2] = index
            writer.write(frame)
        writer.release()

    def tearDown(self):
        self.temp.cleanup()

    def row(self):
        return {"video_id": "V1", "path": "sample.mp4", "style": "serbest", "duration_sec": "2.5",
                "swimmer_id": "S1", "session_id": "A", "style_verified": "verified", "split": "train"}

    def test_metadata_and_ambiguous_labels(self):
        path = self.root / "metadata.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self.row())
            writer.writeheader()
            writer.writerow(self.row())
        self.assertEqual(load_metadata(path)[0]["video_id"], "V1")
        pending = {**self.row(), "style_verified": "pending"}
        self.assertEqual(exclusion_reason(pending), "style_not_verified")
        multi = {**self.row(), "multi_swimmer": "true"}
        self.assertIsNone(exclusion_reason(multi))

    def test_decode_shape_rgb_and_normalization(self):
        info = probe_video(self.video)
        record = build_clip_records(self.row(), info, sampling_fps=8, clip_length=16, stride=16)[0]
        frames = decode_clip(self.video, record)
        tensor = transform_clip(frames)
        self.assertEqual(frames.shape, (16, 48, 64, 3))
        self.assertEqual(tuple(tensor.shape), (3, 16, 112, 112))
        expected_red = (0 / 255 - R3D_MEAN[0]) / 0.22803
        self.assertAlmostEqual(float(tensor[0, 0].mean()), expected_red, places=3)

    def test_tensor_cache_avoids_redecoding(self):
        record = build_clip_records(self.row(), probe_video(self.video), sampling_fps=8, clip_length=16, stride=16)[0]
        cache = {}
        dataset = VideoClipDataset(self.root, [record], ["serbest"], cache)
        first = dataset[0][0]
        self.video.unlink()
        self.assertTrue(torch.equal(dataset[0][0], first))

    def test_split_leakage(self):
        with self.assertRaises(ValueError):
            validate_split_leakage([self.row(), {**self.row(), "split": "test"}])

    def test_model_forward_training_and_checkpoint(self):
        model = build_r3d18(4, pretrained=False, freeze_backbone=True)
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-4)
        inputs = torch.randn(1, 3, 16, 112, 112)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1, 4))
        torch.nn.CrossEntropyLoss()(logits, torch.tensor([1])).backward()
        optimizer.step()
        checkpoint = self.root / "checkpoint.pt"
        save_checkpoint(str(checkpoint), model, optimizer, 1, ["a", "b", "c", "d"], {"seed": 42})
        loaded = build_r3d18(4, pretrained=False, freeze_backbone=True)
        self.assertEqual(load_checkpoint(str(checkpoint), loaded)["epoch"], 1)

    def test_experimental_cv_is_video_level_grouped_and_stratified(self):
        rows = []
        for style_index, style in enumerate(CLASSES):
            for index in range(6):
                rows.append({"video_id": f"X{style_index}{index}", "style": style})
        rows.extend({"video_id": video_id, "style": "breaststroke"} for video_id in CANDIDATE_GROUPS["G1"])
        rows.extend({"video_id": video_id, "style": "breaststroke"} for video_id in CANDIDATE_GROUPS["G2"])
        rows.extend({"video_id": video_id, "style": "backstroke"} for video_id in CANDIDATE_GROUPS["G3"])
        folds = make_folds(rows, 5, 42)
        checks = validate_folds(rows, folds)
        self.assertEqual(checks["status"], "passed")
        self.assertFalse(checks["independent_test"])

    def test_experimental_cv_keeps_cross_class_candidate_group_together(self):
        rows = [{"video_id": video_id, "style": "breaststroke"} for video_id in CANDIDATE_GROUPS["G1"]]
        rows[0]["style"] = "freestyle"
        rows.extend({"video_id": f"X{style}{index}", "style": style}
                    for style in CLASSES for index in range(2))
        folds = make_folds(rows, 2, 42)
        assignments = [fold["fold"] for fold in folds if CANDIDATE_GROUPS["G1"] & set(fold["validation_ids"])]
        self.assertEqual(len(assignments), 1)

    def test_experimental_cv_deduplicates_sha256(self):
        rows = [{"video_id": "B", "sha256": "same"}, {"video_id": "A", "sha256": "same"},
                {"video_id": "C", "sha256": "other"}]
        unique, groups = deduplicate_rows(rows)
        self.assertEqual([row["video_id"] for row in unique], ["A", "C"])
        self.assertEqual(groups[0]["dropped_video_ids"], ["B"])

    def test_binary_metadata_keeps_weak_labels_and_related_folds(self):
        master = [
            {"video_id": "D", "path": "data\\raw\\bogulma\\d.mp4", "sha256": "d",
             "behavior": "unknown", "review_status": "unreviewed", "source_id": "unknown", "session_id": "unknown"},
            {"video_id": "N", "path": "data\\raw\\serbest\\n.mp4", "sha256": "n",
             "behavior": "unknown", "review_status": "unreviewed", "source_id": "unknown", "session_id": "unknown"},
        ]
        state = {
            "D": {"behavior": "drowning_candidate", "review_status": "ambiguous"},
            "N": {"behavior": "normal", "review_status": "verified", "mixed_styles": "no"},
        }
        rows, _ = prepare_rows(master, state, [{"video_id_a": "D", "video_id_b": "N"}], 2)
        by_id = {row["video_id"]: row for row in rows}
        self.assertEqual(by_id["D"]["label_strength"], "weak_video_label")
        self.assertEqual(by_id["D"]["clip_label"], "unlabeled")
        self.assertEqual(by_id["D"]["fold"], by_id["N"]["fold"])
        self.assertTrue(by_id["N"]["initial_training_eligible"])

    def test_binary_video_metrics_include_f1_fpr_and_score_warning(self):
        metrics = binary_video_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2])
        self.assertEqual(metrics["f1"], [0.5, 0.5])
        self.assertEqual(metrics["false_positive_rate"], [0.5, 0.5])
        self.assertFalse(metrics["score_is_calibrated_risk_probability"])


if __name__ == "__main__":
    unittest.main()
