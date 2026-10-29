import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.drowning_demo.swimmers import score_clip
from src.video_classification.train_yolo_r3d18 import parse_args, main
from src.video_classification.visualize_person_r3d18 import person_clip_plan, person_text, score_people


def parent(end=15, step=1):
    return {'clip_id': 'C1', 'video_id': 'V1', 'source_path': 'results/detect_all/bogulma/v/v.avi',
            'start_frame': '0', 'end_frame': str(end), 'frame_step': str(step),
            'start_time': '0', 'end_time': str(end / 8), 'video_label': '1',
            'draft_split': 'validation', 'leakage_group': 'G1'}


def detections(count=16):
    return [[{'track_id': 1, 'box': [20, 20, 100, 100], 'confidence': 0.9},
             {'track_id': 2, 'box': [200, 20, 280, 100], 'confidence': 0.9}]
            for _ in range(count)]


class PersonR3DTests(unittest.TestCase):
    def test_training_defaults_resume_5_epochs_double_positive_weight(self):
        args = parse_args([])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.drowning_weight, 5.0)
        self.assertEqual(args.drowning_weight / (55 / 22), 2)
        self.assertEqual(args.init_checkpoint.name, 'best_model.pt')
        self.assertNotEqual(args.output_dir, args.init_checkpoint.parent)

    def test_training_invalid_settings_rejected(self):
        for values in (['--epochs', '0'], ['--drowning-weight', '0'],
                       ['--drowning-weight', 'nan'], ['--drowning-weight', 'inf']):
            with self.subTest(values=values), self.assertRaises(SystemExit):
                parse_args(values)

    def test_training_cannot_overwrite_existing_experiment_or_start_during_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'best_model.pt'
            checkpoint.write_bytes(b'preserve')
            with patch('src.video_classification.train_yolo_r3d18.torch.cuda.is_available', return_value=True), \
                 patch('src.video_classification.train_yolo_r3d18.build_r3d18') as model:
                with self.assertRaises(FileExistsError):
                    main(['--init-checkpoint', str(checkpoint), '--output-dir', str(root)])
            model.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b'preserve')

    def test_all_real_samples_and_group_parent_preserved(self):
        plans, skipped = person_clip_plan([parent(60, 4)], detections(61))
        self.assertEqual(len(plans), 2)
        self.assertEqual(skipped, [])
        self.assertEqual(plans[0]['sample_frame_indices'], list(range(0, 61, 4)))
        self.assertTrue(all(p['person_event_label'] == 'unknown' and p['leakage_group'] == 'G1' for p in plans))
        self.assertEqual(len({p['person_clip_id'] for p in plans}), 2)

    def test_gap_reused_id_and_unconfirmed_detections_do_not_get_invented_crops(self):
        frames = detections()
        frames[9] = [frames[9][1], {'track_id': None, 'box': [20, 20, 100, 100], 'confidence': 0.15}]
        plans, _ = person_clip_plan([parent()], frames)
        self.assertEqual([p['track_id'] for p in plans], [2])
        plans, skipped = person_clip_plan([parent()], [[] for _ in range(16)])
        self.assertEqual(plans, [])
        self.assertEqual(skipped[0]['reason'], 'no_confirmed_track_in_all_16_sampled_frames')

    def test_user_approved_gaps_only_between_real_sampled_frames(self):
        frames = detections(61)
        frames[9] = [frames[9][1]]
        plans, _ = person_clip_plan([parent(60, 4)], frames)
        self.assertEqual([p['track_id'] for p in plans], [1, 2])
        frames[8] = [frames[8][1]]
        plans, _ = person_clip_plan([parent(60, 4)], frames)
        self.assertEqual([p['track_id'] for p in plans], [2])

    def test_person_scores_depend_on_separate_crops_not_parent_video_target(self):
        rgb = np.zeros((120, 320, 3), np.uint8)
        rgb[:, 160:] = 255
        fake_frames = [SimpleNamespace(to_ndarray=lambda format: rgb) for _ in range(16)]
        frames = detections()
        plans, _ = person_clip_plan([parent()], frames)

        class MeanModel(torch.nn.Module):
            def forward(self, batch):
                mean = batch.mean(dim=(1, 2, 3, 4))
                return torch.stack((-mean, mean), dim=1)

        with patch('src.video_classification.visualize_person_r3d18.decoded_frames', return_value=iter(fake_frames)), \
             patch('src.video_classification.visualize_person_r3d18.score_clip',
                   side_effect=lambda model, samples, device: score_clip(model, samples, 'cpu')):
            results = score_people(Path('unused'), frames, plans, MeanModel().eval())
        self.assertEqual(len(results), 2)
        self.assertLess(results[0]['drowning_probability'], 0.3)
        self.assertGreater(results[1]['drowning_probability'], 0.3)
        self.assertTrue(all(r['sample_count'] == 16 and r['padded_frames'] == 0 for r in results))

    def test_text_threshold_and_missing_score(self):
        person = {'track_id': 1}
        low, color = person_text(person, {'drowning_probability': 0.29}, 0.3)
        self.assertIn('Normal yüzücü', low)
        self.assertEqual(color, '#30EF90')
        high, color = person_text(person, {'drowning_probability': 0.3}, 0.3)
        self.assertIn('Boğulma riski (deneysel)', high)
        self.assertEqual(color, '#FF4444')
        missing, color = person_text(person, None, 0.3)
        self.assertIn('Skor yok', missing)
        self.assertNotIn('Normal', missing)
        self.assertEqual(color, '#FFC857')


if __name__ == '__main__':
    unittest.main()
