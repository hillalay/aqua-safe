import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.video_classification.preprocessing import probe_video
from src.video_classification.visualize_yolo_r3d18 import (
    error_group, infer, overlay_lines, prediction_at_frame, render_video, summarize_predictions,
    run, validate_cached_predictions, verify_render,
)
from src.video_classification.prepare_binary_v1 import write_csv
from src.video_classification.yolo_dataset import read_csv
from src.video_review.inventory import sha256_file


def prediction(index, target, output):
    return {'video_id': f'V{target}', 'clip_id': f'C{index}', 'video_label': str(target),
            'source_path': f'results/detect_all/bogulma/v/v.avi',
            'predicted_label': output, 'predicted_class': ('normal', 'drowning')[output],
            'normal_probability': 0.9 if output == 0 else 0.1,
            'drowning_probability': 0.1 if output == 0 else 0.9,
            'error_group': error_group(target, output), 'label_strength': 'weak_video_label',
            'clip_event_label': 'unknown', 'start_frame': '0', 'end_frame': '15',
            'start_time': '0', 'end_time': '1.875'}


class VisualizationTests(unittest.TestCase):
    def test_all_error_groups_and_verified_reference(self):
        rows = []
        for target, output, count in ((0, 0, 151), (0, 1, 14), (1, 0, 31), (1, 1, 13)):
            for _ in range(count):
                rows.append(prediction(len(rows), target, output))
        result = summarize_predictions(rows)
        self.assertEqual(result['confusion_matrix'], [[151, 14], [31, 13]])
        self.assertTrue(result['matches_user_reported_reference'])
        self.assertEqual(result['weak_label_groups']['FN']['count'], 31)
        self.assertEqual(result['weak_label_groups']['TN']['count'], 151)
        self.assertFalse(result['independent_test'])

    def test_duplicate_predictions_and_invalid_labels_fail(self):
        row = prediction(0, 1, 0)
        with self.assertRaises(ValueError):
            summarize_predictions([row, row])
        with self.assertRaises(ValueError):
            error_group(2, 0)

    def test_intervals_are_inclusive_and_gaps_never_get_predictions(self):
        rows = [dict(prediction(0, 1, 0), start_frame='4', end_frame='19'),
                dict(prediction(1, 1, 1), start_frame='24', end_frame='39')]
        starts = [4, 24]
        for index in (0, 3, 20, 23, 40, 50):
            self.assertIsNone(prediction_at_frame(rows, starts, index))
        for index in (4, 19):
            self.assertIs(prediction_at_frame(rows, starts, index), rows[0])
        for index in (24, 39):
            self.assertIs(prediction_at_frame(rows, starts, index), rows[1])

    def test_overlay_uses_prediction_and_never_parent_target(self):
        row = prediction(0, 1, 0)
        lines = overlay_lines(row, 0.5)
        self.assertIn('PRED: normal', lines[1])
        self.assertNotIn('PRED: drowning', '\n'.join(lines))
        self.assertNotIn('video_label', '\n'.join(lines))
        self.assertIn('NO PREDICTION', overlay_lines(None, 2.5)[1])
        self.assertNotIn('P(normal)', '\n'.join(overlay_lines(None, 2.5)))

    def test_inference_once_per_batch_without_gradients_or_label_copying(self):
        class Clips(Dataset):
            def __init__(self):
                self.rows = [prediction(i, 1, 0) for i in range(3)]

            def __len__(self):
                return 3

            def __getitem__(self, index):
                return torch.zeros(3, 16, 8, 8), 1, self.rows[index]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                if torch.is_grad_enabled() or self.training:
                    raise AssertionError('Inference must use eval and no gradients')
                self.calls += 1
                return torch.tensor([[2., -1.]]).repeat(len(x), 1)

        model = Model()
        result = infer(model, DataLoader(Clips(), batch_size=2, shuffle=False), torch.device('cpu'))
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r['predicted_label'] == 0 and r['error_group'] == 'FN' for r in result))
        self.assertTrue(all(abs(r['normal_probability'] + r['drowning_probability'] - 1) < 1e-6 for r in result))

    def test_real_h264_render_keeps_frames_gaps_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = 'results/detect_all/bogulma/v/v.avi'
            source = root / relative
            source.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 8, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(40):
                writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
            writer.release()
            digest = sha256_file(source)
            rows = [dict(prediction(0, 1, 0), source_path=relative),
                    dict(prediction(1, 1, 1), source_path=relative, start_frame='20',
                         end_frame='35', start_time='2.5', end_time='4.375')]
            out = root / 'output.mp4'
            with patch('src.video_classification.visualize_yolo_r3d18.overlay_lines', wraps=overlay_lines) as spy:
                result = render_video(root, rows, out)
            self.assertEqual(result['frames'], 40)
            self.assertEqual(probe_video(out).frame_count, 40)
            self.assertEqual(spy.call_count, 40)
            for i in (16, 19, 36, 39):
                self.assertIsNone(spy.call_args_list[i].args[0])
            with av.open(str(out)) as container:
                self.assertEqual(container.streams.video[0].codec_context.name, 'h264')
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 40)
            self.assertEqual(sha256_file(source), digest)
            with self.assertRaises(FileExistsError):
                render_video(root, rows, out)

    def test_cached_predictions_reject_missing_duplicate_stale_or_invalid_scores(self):
        original = prediction(0, 1, 0)
        parent = {k: v for k, v in original.items() if k not in
                  {'predicted_label', 'predicted_class', 'normal_probability', 'drowning_probability', 'error_group'}}
        validate_cached_predictions([original], [parent])
        for rows in ([], [original, original], [{**original, 'start_time': '0.1'}],
                     [{**original, 'drowning_probability': float('nan')}],
                     [{**original, 'drowning_probability': 0.8}],
                     [{**original, 'predicted_label': 1}], [{**original, 'error_group': 'TP'}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_cached_predictions(rows, [parent])

    def test_all_videos_uses_cache_reuses_existing_and_never_loads_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'results/aquasafe_yolo_r3d18_v1/visualization'
            output.mkdir(parents=True)
            rows, parents = [], []
            for label in (0, 1):
                folder = 'bogulma' if label else 'serbest'
                relative = f'results/detect_all/{folder}/v{label}/v{label}.avi'
                path = root / relative
                path.parent.mkdir(parents=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, (64, 48))
                self.assertTrue(writer.isOpened())
                for _ in range(16):
                    writer.write(np.full((48, 64, 3), 40 + label * 60, dtype=np.uint8))
                writer.release()
                row = {**prediction(label, label, 0), 'source_path': relative, 'yolo_sha256': sha256_file(path)}
                rows.append(row)
                parents.append({k: str(v) for k, v in row.items() if k not in
                                {'predicted_label', 'predicted_class', 'normal_probability', 'drowning_probability', 'error_group'}})
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'not a model: must never be loaded')
            trainer = root / 'src/video_classification/train_yolo_r3d18.py'
            trainer.parent.mkdir(parents=True)
            trainer.write_text('original training settings', encoding='utf-8')
            historical_trainer_hash = sha256_file(trainer)
            trainer.write_text('new training settings for a separate experiment', encoding='utf-8')
            csv_path = output / 'predictions.csv'
            write_csv(csv_path, rows, list(rows[0]))
            metrics_path = output / 'metrics.json'
            metrics_path.write_text(json.dumps({
                'prediction_csv_sha256': sha256_file(csv_path),
                'input_sha256': {'checkpoint.pt': sha256_file(checkpoint),
                                 'src/video_classification/train_yolo_r3d18.py': historical_trainer_hash},
                'source_yolo_sha256': {r['source_path']: r['yolo_sha256'] for r in rows},
            }), encoding='utf-8')
            existing = output / 'videos/normal_v0.mp4'
            render_video(root, [rows[0]], existing)
            before = {p: sha256_file(p) for p in (existing, csv_path, metrics_path, checkpoint, trainer)}
            module = 'src.video_classification.visualize_yolo_r3d18.'
            with patch(module + 'YoloClipDataset', return_value=SimpleNamespace(rows=parents)), \
                 patch(module + 'infer', side_effect=AssertionError('Must not rerun inference')), \
                 patch(module + 'build_r3d18', side_effect=AssertionError('Must not load model')), \
                 patch(module + 'render_video', wraps=render_video) as renderer:
                report = run(root, all_videos=True)
            self.assertEqual(renderer.call_count, 1)
            self.assertEqual((report['created'], report['reused'], report['expected_videos']), (1, 1, 2))
            self.assertTrue(report['complete'])
            self.assertFalse(report['inference_rerun'])
            self.assertTrue(report['training_script_changed_since_inference'])
            self.assertEqual(report['failed'], [])
            self.assertTrue((output / 'videos/drowning_v1.mp4').is_file())
            self.assertEqual(before, {p: sha256_file(p) for p in before})
            self.assertTrue(all(v['playability_check'] == 'full_decode_passed' for v in report['videos']))
            # Cached CSV tampering must fail before any render/model operation.
            csv_path.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash'):
                run(root, all_videos=True)


if __name__ == '__main__':
    unittest.main()
