import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.video_classification.prepare_binary_v1 import write_csv
from src.video_classification.prepare_yolo_clips import build_clips, experimental_split, select_videos
from src.video_classification.preprocessing import build_clip_records, VideoInfo
from src.video_classification.yolo_dataset import (
    YoloClipDataset, annotated_path, make_loaders, read_csv, validate_manifest,
)


def make_video(path, red):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, (64, 48))
    if not writer.isOpened():
        raise RuntimeError('MJPG fixture unavailable')
    try:
        for _ in range(40):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 2] = red
            writer.write(frame)
    finally:
        writer.release()


def video(index, label, group=None):
    return {'video_id': f'V{index}', 'class_label': label,
            'sha256': hashlib.sha256(f'original-{index}'.encode()).hexdigest(),
            'relation_group': f'R{index}', 'leakage_group': group or f'G{index}',
            'confirmed_source_group': 'unknown', 'source_id': 'unknown', 'session_id': 'unknown',
            'frame_count': '40', 'duration_sec': '5',
            'yolo_output_path': f"results/detect_all/{'bogulma' if label == 'drowning' else 'serbest'}/v{index}/v{index}.avi"}


class YoloClipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.videos = [video(1, 'drowning'), video(2, 'normal')]
        for row, red in zip(self.videos, (220, 70)):
            make_video(self.root / row['yolo_output_path'], red)
        self.clips, self.prepared = build_clips(self.videos, {'V1': 'train', 'V2': 'validation'}, self.root)
        self.manifest = self.root / 'clips.csv'
        write_csv(self.manifest, self.clips, list(self.clips[0]))

    def tearDown(self):
        self.temp.cleanup()

    def test_dataset_shape_rgb_weak_target_and_provenance(self):
        dataset = YoloClipDataset(self.root, self.manifest)
        x, label, meta = dataset[0]
        self.assertEqual(tuple(x.shape), (3, 16, 112, 112))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x).all())
        self.assertGreater(float(x[0].mean()), float(x[2].mean()))
        self.assertEqual(label, 1)
        self.assertEqual(meta['video_id'], 'V1')
        self.assertEqual(meta['label_strength'], 'weak_video_label')
        self.assertEqual(meta['clip_event_label'], 'unknown')
        self.assertTrue(meta['source_path'].startswith('results/detect_all/'))

    def test_dataloaders_batch_and_partition_integrity(self):
        loaders = make_loaders(self.root, self.manifest, batch_size=2)
        self.assertEqual(set(loaders), {'train', 'validation'})
        for split, expected in (('train', 1), ('validation', 0)):
            x, labels, meta = next(iter(loaders[split]))
            self.assertEqual(tuple(x.shape), (2, 3, 16, 112, 112))
            self.assertEqual(labels.dtype, torch.int64)
            self.assertEqual(labels.tolist(), [expected, expected])
            self.assertEqual(set(meta['draft_split']), {split})
        with self.assertRaises(ValueError):
            YoloClipDataset(self.root, self.manifest, 'test')

    def test_exactly_sixteen_frames_no_overlap_or_padding(self):
        self.assertEqual(len(self.clips), 4)
        first, second = self.clips[:2]
        self.assertEqual((first['start_frame'], first['end_frame']), (0, 15))
        self.assertEqual((second['start_frame'], second['end_frame']), (16, 31))
        self.assertLess(second['end_frame'], 40)
        records = build_clip_records({'video_id': 'V', 'path': 'x', 'style': 'normal'},
                                     VideoInfo(30, 128, 128 / 30, 64, 48), 8, 16, 16)
        self.assertEqual([r.frame_step for r in records], [4, 4])
        self.assertEqual([(r.start_frame, r.end_frame) for r in records], [(0, 60), (64, 124)])

    def test_raw_paths_absolute_paths_and_traversal_rejected(self):
        for path in ('data/raw/bogulma/v1.avi', '../outside.avi',
                     str(self.root / self.videos[0]['yolo_output_path'])):
            with self.subTest(path=path), self.assertRaises(ValueError):
                annotated_path(self.root, path)

    def test_folder_label_conflict_and_confirmed_clip_claim_rejected(self):
        for field, value in (('video_label', 0), ('clip_event_label', 'drowning'),
                             ('label_strength', 'verified'), ('draft_split', 'test')):
            rows = [dict(r) for r in self.clips]
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_manifest(self.root, rows)

    def test_clip_parent_and_group_leakage_rejected(self):
        rows = [dict(r) for r in self.clips]
        rows[1]['draft_split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'crosses partitions'):
            validate_manifest(self.root, rows)
        for field in ('relation_group', 'leakage_group', 'source_id', 'confirmed_source_group'):
            rows = [dict(r) for r in self.clips]
            for row in rows:
                row[field] = 'SHARED'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'crosses partitions'):
                validate_manifest(self.root, rows)

    def test_duplicates_overlapping_intervals_and_invalid_frames_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Repeated clip ID'):
            validate_manifest(self.root, self.clips + [self.clips[0]])
        duplicate = {**self.clips[0], 'clip_id': 'different-id-same-frames'}
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            validate_manifest(self.root, self.clips + [duplicate])
        for field, value in (('end_frame', 40), ('frame_step', 0), ('start_time', float('nan'))):
            rows = [dict(r) for r in self.clips]
            rows[0][field] = value
            with self.assertRaises(ValueError):
                validate_manifest(self.root, rows)

    def test_exact_yolo_content_cannot_enter_another_video_or_split(self):
        rows = [dict(r) for r in self.clips]
        for row in rows:
            row['yolo_sha256'] = rows[0]['yolo_sha256']
        with self.assertRaises(ValueError):
            validate_manifest(self.root, rows)

    def test_decode_failure_is_not_silently_padded(self):
        dataset = YoloClipDataset(self.root, self.manifest)
        (self.root / self.videos[0]['yolo_output_path']).write_bytes(b'corrupt avi')
        with self.assertRaises(RuntimeError):
            dataset[0]

    def test_approved_yolo_metadata_drift_rejected(self):
        rows = [dict(r) for r in self.videos]
        rows[0]['frame_count'] = '100'
        with self.assertRaises(ValueError):
            build_clips(rows, {'V1': 'train', 'V2': 'validation'}, self.root)

    def test_fixed_92_cohort_and_group_consistency(self):
        rows = [video(i, 'drowning' if i < 27 else 'normal') for i in range(92)]
        for row in rows:
            row['weak_training_with_yolo_candidate'] = 'True'
        groups = [dict(r) for r in rows]
        selected = select_videos(rows, groups)
        self.assertEqual(len(selected), 92)
        rows.append({**rows[0], 'weak_training_with_yolo_candidate': 'False'})
        self.assertEqual(len(select_videos(rows, groups)), 92)
        with self.assertRaises(ValueError):
            select_videos(rows[1:], groups)
        groups[0]['sha256'] = 'changed'
        with self.assertRaises(ValueError):
            select_videos(rows, groups)

    def test_grouped_draft_reproducible_and_not_a_test_set(self):
        rows = [video(i, 'drowning' if i % 2 else 'normal', group=f'G{i // 2}') for i in range(20)]
        first = experimental_split(rows)
        self.assertEqual(first, experimental_split(rows))
        self.assertEqual(set(first.values()), {'train', 'validation'})
        self.assertEqual(len(first), len(rows))
        for i in range(0, 20, 2):
            self.assertEqual(first[f'V{i}'], first[f'V{i+1}'])
        self.assertEqual(sum(split == 'validation' for split in first.values()), 4)


if __name__ == '__main__':
    unittest.main()
