import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pose_pipeline.extractors import PoseExtractor
from src.pose_pipeline.quality import landmark_is_usable, pose_is_usable, roi_to_original, window_quality
from src.pose_pipeline.schema import COMMON_LANDMARKS, Landmark, PoseFrame
from src.pose_pipeline.temporal import save_windows, validate_split_groups
from src.pose_pipeline.video import calculate_metrics, sampled_frame_numbers


def frame(number: int, point: Landmark | None = None) -> PoseFrame:
    common = {name: None for name in COMMON_LANDMARKS}
    if point:
        common[point.name] = point
    return PoseFrame("V001", number, number / 10, "fake", "1", point is not None, (point,) if point else (), common)


class PosePipelineTests(unittest.TestCase):
    def test_extractor_interface(self):
        class FakeExtractor(PoseExtractor):
            name, version = "fake", "1"

            def extract(self, bgr_frame, timestamp_ms):
                self.asserted_shape = bgr_frame.shape
                return (), {name: None for name in COMMON_LANDMARKS}

        extractor = FakeExtractor()
        native, common = extractor.extract(np.zeros((2, 3, 3), dtype=np.uint8), 0)
        self.assertEqual(native, ())
        self.assertEqual(set(common), set(COMMON_LANDMARKS))

    def test_sampling_and_timestamps(self):
        self.assertEqual(sampled_frame_numbers(30, 30, 10), list(range(0, 30, 3)))
        self.assertEqual(sampled_frame_numbers(5, 5, 10), list(range(5)))

    def test_missing_pose_has_no_fake_coordinates_and_valid_schema(self):
        missing = frame(0)
        missing.validate()
        data = missing.to_dict()
        self.assertEqual(data["pose_status"], "not_detected")
        self.assertEqual(data["native_landmarks"], [])
        self.assertTrue(all(value is None for value in data["common_landmarks"].values()))
        metrics = calculate_metrics([missing], 1)
        self.assertEqual(metrics["missing_common_landmark_rate"], 1.0)
        self.assertEqual(metrics["raw_pose_detection_rate"], 0.0)
        self.assertEqual(metrics["usable_pose_detection_rate"], 0.0)

    def test_windows_keep_mask_and_video_identity(self):
        records = [frame(i, Landmark("nose", 0.2, 0.3, confidence=0.9)) for i in range(30)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "windows.npz"
            self.assertEqual(save_windows(records, path, "serbest", "train"), 1)
            with np.load(path) as saved:
                self.assertEqual(saved["features"].shape, (1, 30, 13, 3))
                self.assertTrue(saved["mask"][0, 0, 0])
                self.assertFalse(saved["training_usable"][0])
                self.assertGreater(float(saved["missing_rate"][0]), 0.9)
                self.assertEqual(str(saved["video_id"]), "V001")

    def test_presence_and_pose_usability_are_not_hidden(self):
        low_presence = Landmark("nose", 0.2, 0.3, visibility=0.9, presence=0.1)
        self.assertFalse(landmark_is_usable(low_presence))
        common = {name: Landmark(name, 0.2, 0.3, confidence=0.9) for name in COMMON_LANDMARKS}
        self.assertTrue(pose_is_usable(common))
        complete = [PoseFrame("V", i, i / 10, "fake", "1", True, tuple(common.values()), common) for i in range(30)]
        self.assertEqual(window_quality(complete), (0.0, True))

    def test_roi_coordinates_return_to_original_frame(self):
        point = roi_to_original(Landmark("nose", 0.5, 0.5, confidence=0.9), (20, 10, 120, 60), 200, 100)
        self.assertAlmostEqual(point.x, 0.35)
        self.assertAlmostEqual(point.y, 0.35)

    def test_same_group_cannot_cross_splits(self):
        rows = [
            {"video_id": "V1", "swimmer_id": "S1", "session_id": "A", "split": "train"},
            {"video_id": "V2", "swimmer_id": "S1", "session_id": "A", "split": "test"},
        ]
        with self.assertRaises(ValueError):
            validate_split_groups(rows)
        with self.assertRaises(ValueError):
            validate_split_groups([
                {"video_id": "V1", "swimmer_id": "S1", "session_id": "unknown", "split": "train"},
                {"video_id": "V2", "swimmer_id": "S1", "session_id": "unknown", "split": "test"},
            ])
        validate_split_groups([
            {"video_id": "V1", "swimmer_id": "unknown", "session_id": "unknown", "split": "train"},
            {"video_id": "V1", "swimmer_id": "unknown", "session_id": "unknown", "split": "train"},
        ])


if __name__ == "__main__":
    unittest.main()
