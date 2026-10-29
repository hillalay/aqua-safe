import csv
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from src.video_classification.prepare_binary_v1 import collect, label_for, prepare, summarize


def make_video(path, frames=24, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, (32, 32))
    if not writer.isOpened():
        raise RuntimeError('MJPG fixture writer unavailable')
    try:
        rng = np.random.default_rng(seed)
        for _ in range(frames):
            writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()


class PreparationTests(unittest.TestCase):
    def test_labels_and_conflicts(self):
        raw = Path('data/raw')
        for folder in ('kelebek', 'kurbagalama', 'serbest', 'sirtustu'):
            self.assertEqual(label_for(raw / folder / 'clip.mp4', raw), ('normal', False))
            self.assertEqual(label_for(raw / folder / 'Boğulma.mp4', raw), ('', True))
            self.assertEqual(label_for(raw / folder / 'drowning.mp4', raw), ('', True))
        self.assertEqual(label_for(raw / 'bogulma/a.mp4', raw), ('drowning', False))
        self.assertEqual(label_for(raw / 'bogulma/serbest.mp4', raw), ('', True))
        self.assertEqual(label_for(raw / 'other/a.mp4', raw), ('', False))

    def test_end_to_end_dedup_missing_yolo_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data/raw/bogulma/a.avi'
            make_video(source)
            shutil.copyfile(source, source.with_name('copy.avi'))
            make_video(root / 'data/raw/serbest/b.avi', seed=2)
            yolo = root / 'results/detect_all/bogulma/a/a.avi'
            yolo.parent.mkdir(parents=True)
            shutil.copyfile(source, yolo)
            original = source.read_bytes(), yolo.read_bytes()
            with redirect_stdout(io.StringIO()):
                summary = prepare(root)
            out = root / 'results/aquasafe_binary_v1'
            self.assertEqual(summary['raw_video_files'], 3)
            self.assertEqual(summary['unique_content_videos'], 2)
            self.assertEqual(summary['weak_training_candidates'], 2)
            self.assertEqual(summary['weak_training_with_yolo_candidates'], 1)
            with (out / 'dataset_manifest.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({r['sha256'] for r in rows}), 2)
            self.assertTrue(all(r['split'] == 'unassigned' and r['person_detection_present'] == 'unknown'
                                and r['label_strength'] == 'weak_video_label' for r in rows))
            duplicates = json.loads((out / 'duplicate_report.json').read_text())
            self.assertEqual(duplicates['redundant_exact_duplicate_files'], 1)
            with (out / 'missing_yolo_outputs.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            readiness = json.loads((out / 'data_readiness_report.json').read_text())
            self.assertFalse(readiness['ready_for_training'])
            with self.assertRaises(FileExistsError):
                prepare(root)
            self.assertEqual(original, (source.read_bytes(), yolo.read_bytes()))

    def test_cross_folder_duplicate_conflict_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data/raw/bogulma/a.avi'
            make_video(source)
            other = root / 'data/raw/serbest/b.avi'
            other.parent.mkdir(parents=True)
            shutil.copyfile(source, other)
            with redirect_stdout(io.StringIO()):
                records, scan = collect(root)
            manifest, _, duplicates, _ = summarize(records, scan)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]['class_label'], '')
            self.assertFalse(manifest[0]['weak_training_candidate'])
            self.assertTrue(duplicates['exact_duplicate_groups'][0]['label_conflict'])

    def test_partial_yolo_and_unreadable_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_video(root / 'data/raw/serbest/good.avi')
            make_video(root / 'results/detect_all/serbest/good/good.avi', frames=4)
            (root / 'data/raw/serbest/bad.mp4').write_bytes(b'not a video')
            with redirect_stdout(io.StringIO()):
                records, scan = collect(root)
            manifest, summary, _, _ = summarize(records, scan)
            self.assertEqual(summary['weak_training_candidates'], 1)
            self.assertEqual(summary['weak_training_with_yolo_candidates'], 0)
            self.assertEqual(next(r for r in manifest if r['filename'] == 'good.avi')['yolo_output_status'],
                             'frame_count_mismatch')

    def test_similar_videos_share_relation_group_and_changes_block_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_video(root / 'data/raw/serbest/a.avi')
            make_video(root / 'data/raw/serbest/b.avi', seed=1)
            with redirect_stdout(io.StringIO()):
                records, scan = collect(root)
            for record in records:
                record['visual_hashes'] = '|'.join(['0' * 16] * 5)
            manifest, _, duplicates, _ = summarize(records, scan)
            self.assertEqual(duplicates['suspicious_similar_pair_count'], 1)
            self.assertEqual(len({r['relation_group'] for r in manifest}), 1)
            scan['files_changed_during_scan']['yolo'] = ['changed.avi']
            manifest, _, _, readiness = summarize(records, scan)
            self.assertTrue(all(not r['weak_training_candidate'] for r in manifest))
            self.assertIn('Files changed', readiness['blocking_issues'][-1])

    def test_ambiguous_output_and_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/raw').mkdir(parents=True)
            with self.assertRaises(ValueError):
                collect(root)
            make_video(root / 'data/raw/serbest/a.avi')
            target = root / 'results/detect_all/serbest/a'
            make_video(target / 'a.avi')
            (target / 'a.mp4').write_bytes(b'invalid second candidate')
            with redirect_stdout(io.StringIO()):
                records, scan = collect(root)
            self.assertEqual(records[0]['yolo_output_status'], 'ambiguous')
            self.assertEqual(summarize(records, scan)[1]['weak_training_with_yolo_candidates'], 0)

    def test_same_stem_sources_cannot_claim_one_yolo_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data/raw/serbest/a.avi'
            make_video(source)
            shutil.copyfile(source, source.with_suffix('.mp4'))
            make_video(root / 'results/detect_all/serbest/a/a.avi')
            with redirect_stdout(io.StringIO()):
                records, _ = collect(root)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r['yolo_output_status'] == 'ambiguous_source_mapping' for r in records))


if __name__ == '__main__':
    unittest.main()
