import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.drowning_demo.analysis import inspect_video
from src.drowning_demo.swimmers import analyze_swimmers, collect_tracks, crop_person, draw_people, score_clip
from src.drowning_demo.detection import attach_tracks, detection_gaps, make_tracker
from src.drowning_demo.detection_review import match_boxes
from ultralytics.engine.results import Boxes
from tests.test_drowning_demo import make_video


class MeanModel(torch.nn.Module):
    def forward(self, tensor):
        value = tensor.mean(dim=(1, 2, 3, 4))
        return torch.stack((-value, value), dim=1)


class Detector:
    def __init__(self):
        self.index = -1

    def track(self, *args, **kwargs):
        self.index += 1
        ids = [1, 2] if self.index != 9 else [2]
        boxes = [[0, 0, 70, 110] if i == 1 else [80, 0, 159, 110] for i in ids]
        return [SimpleNamespace(boxes=SimpleNamespace(id=torch.tensor(ids), xyxy=torch.tensor(boxes)))]


class SwimmerTests(unittest.TestCase):
    def test_raw_boxes_survive_bytetrack_confirmation_and_low_confidence(self):
        tracker = make_tracker()
        boxes = Boxes(torch.tensor([[5, 30, 35, 80, .15, 0], [90, 30, 140, 100, .9, 0]]), (120, 160))
        people = attach_tracks(boxes, tracker)
        self.assertEqual(len(people), 2)
        self.assertIsNone(people[0]['track_id'])
        self.assertIsNotNone(people[1]['track_id'])
        from PIL import Image
        frame = SimpleNamespace(to_image=lambda: Image.new('RGB', (160, 120), 'black'))
        rendered = draw_people(frame, people, None, 0)
        self.assertEqual(rendered.getpixel((5, 60)), (255, 200, 87))
        self.assertEqual(rendered.getpixel((90, 60)), (255, 200, 87))
        stats = detection_gaps([[], [], people, [], people])
        self.assertEqual(stats['empty_frame_runs'], [2, 1])
        self.assertEqual(stats['detections_without_confirmed_id'], 2)

    def test_detection_only_exports_real_frames_without_loading_behavior_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, weight, output = root/'input.mp4', root/'detector.pt', root/'output'
            make_video(source, count=3)
            weight.write_bytes(b'test weight')
            output.mkdir()
            class RawDetector:
                def predict(self, image, **kwargs):
                    return [SimpleNamespace(boxes=Boxes(torch.tensor([[10, 30, 60, 100, .15, 0]]), image.shape[:2]))]
            with patch('src.drowning_demo.swimmers.YOLO', return_value=RawDetector()), \
                 patch('src.drowning_demo.swimmers.new_output_dir', return_value=output), \
                 patch('src.drowning_demo.swimmers.load_model') as behavior:
                exported = analyze_swimmers(source, root/'missing_checkpoint.pt', 'cpu', weight, detection_only=True)
            behavior.assert_not_called()
            self.assertEqual(inspect_video(exported)['frame_count'], 3)
            import json
            report = json.loads((output/'analysis_summary.json').read_text(encoding='utf-8'))
            self.assertIsNone(report['checkpoint'])
            self.assertEqual(report['frames_with_detections'], 3)
            self.assertEqual(report['scored_windows'], 0)

    def test_box_evaluation_counts_duplicate_and_missed_people(self):
        result = match_boxes([[0, 0, 10, 10], [20, 0, 30, 10]], [[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertEqual(match_boxes([], [])['tp'], 0)

    def test_distinct_people_and_reused_id_after_gap_have_independent_scores(self):
        class Frame:
            def __init__(self, index):
                self.index = index

            def to_ndarray(self, format):
                rgb = np.zeros((120, 320, 3), np.uint8)
                rgb[:, 160:] = 255
                if self.index >= 10:
                    rgb[:, :160] = 255
                return rgb

        class TwoPeople:
            def __init__(self):
                self.index = -1

            def track(self, *args, **kwargs):
                self.index += 1
                ids = [1, 2] if self.index != 9 else [2]
                boxes = [[10, 10, 100, 110] if i == 1 else [210, 10, 310, 110] for i in ids]
                return [SimpleNamespace(boxes=SimpleNamespace(id=torch.tensor(ids), xyxy=torch.tensor(boxes)))]

        with patch('src.drowning_demo.swimmers.decoded_frames', return_value=iter(Frame(i) for i in range(32))):
            frames, windows = collect_tracks(Path('synthetic'), {'fps': 8, 'frame_count': 32},
                                             TwoPeople(), MeanModel(), 'cpu')
        before, other = frames[0]
        after = frames[10][0]
        self.assertEqual(before['track_id'], after['track_id'])
        self.assertNotEqual(before['segment'], after['segment'])
        self.assertTrue(all(w['drowning_behavior_score'] is None for w in windows[before['segment']]))
        self.assertTrue(all(w['drowning_behavior_score'] > 0.5 for w in windows[other['segment']]))
        self.assertTrue(all(w['drowning_behavior_score'] > 0.5 for w in windows[after['segment']]))
        self.assertGreater(min(w['start_frame'] for w in windows[after['segment']]), 9)

    def test_documents_undetected_identity_swap_limitation(self):
        class Frame:
            def __init__(self, index):
                self.index = index

            def to_ndarray(self, format):
                return np.full((120, 160, 3), 0 if self.index < 8 else 255, np.uint8)

        class SameID:
            def track(self, *args, **kwargs):
                return [SimpleNamespace(boxes=SimpleNamespace(id=torch.tensor([1]),
                        xyxy=torch.tensor([[10, 10, 100, 110]])))]

        class RecordingModel(MeanModel):
            def forward(self, tensor):
                self.seen = tensor.clone()
                return super().forward(tensor)

        model = RecordingModel()
        with patch('src.drowning_demo.swimmers.decoded_frames', return_value=iter(Frame(i) for i in range(16))):
            frames, _ = collect_tracks(Path('synthetic'), {'fps': 8, 'frame_count': 16},
                                       SameID(), model, 'cpu')
        # A continuous tracker ID swap is invisible to this scorer; this reproduces the limitation.
        self.assertEqual(frames[0][0]['segment'], frames[-1][0]['segment'])
        self.assertLess(float(model.seen[:, :, :8].mean()), 0)
        self.assertGreater(float(model.seen[:, :, 8:].mean()), 0)

    def test_person_crops_have_independent_scores_and_short_tracks_are_not_scored(self):
        rgb = np.zeros((120, 320, 3), dtype=np.uint8)
        rgb[:, 160:] = 255
        dark = crop_person(rgb, [10, 10, 100, 110])
        light = crop_person(rgb, [210, 10, 310, 110])
        model = MeanModel()
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            short, padded = score_clip(model, [dark], 'cpu')
            forward.assert_not_called()
        normal, _ = score_clip(model, [dark] * 16, 'cpu')
        drowning, _ = score_clip(MeanModel(), [light] * 16, 'cpu')
        self.assertLess(normal, 0.5)
        self.assertGreater(drowning, 0.5)
        self.assertIsNone(short)
        self.assertEqual(padded, 0)
        with self.assertRaises(ValueError):
            crop_person(rgb, [0, 0, 0, 0])

    def test_two_tracks_and_gap_never_share_temporal_buffers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.mp4'
            make_video(path, count=32)
            frames, windows = collect_tracks(path, inspect_video(path), Detector(), MeanModel(), 'cpu')
            self.assertEqual(len(frames), 32)
            self.assertEqual(len(frames[0]), 2)
            first, second = frames[0]
            self.assertNotEqual(first['segment'], second['segment'])
            self.assertNotEqual(frames[10][0]['segment'], first['segment'])
            self.assertEqual(frames[10][1]['segment'], second['segment'])
            self.assertEqual(windows[first['segment']][-1]['end_frame'], 8)
            self.assertTrue(all(windows.values()))


if __name__ == '__main__':
    unittest.main()
