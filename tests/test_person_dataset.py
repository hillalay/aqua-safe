from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.drowning_demo.analysis import decoded_frames, inspect_video, sha256
from src.person_dataset import (export_clip, export_clips, person_image, plan_rows, prepare, track_video, replay_tracking,
                                validate_dataset, validate_rows, write_csv)
from src.person_dataset_update import capture_token, inventory, source_category
from src.video_classification.preprocessing import VideoInfo, build_clip_records
from tests.test_drowning_demo import make_video


class Frame:
    def __init__(self, index):
        self.pts, self.time_base = index, Fraction(1, 30)

    def to_ndarray(self, format):
        return np.zeros((120, 320, 3), dtype=np.uint8)


class Detector:
    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def track(self, *args, **kwargs):
        detections = next(self.sequence)
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=torch.tensor([d[1] for d in detections]).reshape(-1, 4),
            id=torch.tensor([d[0] for d in detections]),
            conf=torch.tensor([d[2] for d in detections])))]


class PersonDatasetTests(unittest.TestCase):
    def info(self, count):
        return {'fps': 30, 'width': 320, 'height': 120, 'frame_count': count,
                'timestamps_sec': [i/30 for i in range(count)], 'duration_sec': count/30,
                'source_pts_origin_sec': 0}

    def tracked(self, sequence):
        with patch('src.person_dataset.decoded_frames', return_value=iter(Frame(i) for i in range(len(sequence)))):
            return track_video(Path('synthetic'), self.info(len(sequence)), Detector(sequence))

    def rows(self, segments, ledger, info=None, output=Path('/pilot'), source=Path('synthetic')):
        return plan_rows(source, info or self.info(len(ledger)), 'abc123', 'source1',
                         'UNRESOLVED_CAPTURE_PILOT', segments, ledger, output)

    def test_two_people_gap_and_id_change_never_share_clip_frames(self):
        a, b = [10, 10, 40, 90], [180, 10, 210, 90]
        sequence = [[(1, a, .9), (2, b, .9)] if i != 30 else [(2, b, .9)] for i in range(80)]
        segments, ledger, events = self.tracked(sequence)
        rows = self.rows(segments, ledger)
        self.assertTrue(any(e['type'] == 'id_reacquired' for e in events))
        for row in rows:
            points = json.loads(row['bbox_coordinates'])
            self.assertEqual([p['frame'] for p in points], list(range(row['start_frame'], row['end_frame']+1)))
            self.assertTrue(all(p['bbox'] == (a if row['track_id'] == 1 else b) for p in points))
            self.assertEqual(row['label'], 'UNCERTAIN')
            self.assertFalse(row['training_eligible'])
            self.assertEqual(row['padded_frame_count'], 0)
            if row['track_id'] == 1:
                self.assertFalse(row['start_frame'] < 30 < row['end_frame'])
        changed, _, _ = self.tracked([[(1, a, .9)], [(3, a, .9)]])
        self.assertEqual({s['track_id'] for s in changed}, {1, 3})

    def test_cached_tracking_replays_exactly_and_rejects_changed_ledger(self):
        a, b = [10, 10, 40, 90], [180, 10, 210, 90]
        sequence = [[(1, a, .9), (2, b, .9)], [(2, b, .9)], [(3, a, .9)], []]
        original = self.tracked(sequence)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'detections.jsonl'
            def replay():
                with patch('src.person_dataset.decoded_frames', return_value=iter(Frame(i) for i in range(4))):
                    return replay_tracking(Path('synthetic'), self.info(4), log, 'bytetrack.yaml')
            log.write_text('\n'.join(json.dumps(f) for f in original[1]), encoding='utf-8')
            self.assertEqual(replay(), original)
            changed = json.loads(log.read_text().splitlines()[0])
            changed['detections'][0]['usable'] = False
            log.write_text('\n'.join(json.dumps(f) for f in [changed]+original[1][1:]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'decisions'):
                replay()
            log.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                replay()

    def test_prepare_resume_preserves_existing_clip_and_refuses_finalized_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.mp4'
            make_video(source, count=20)
            (root/'yolov8s.pt').write_bytes(b'test detector')
            output = root/'dataset'
            (output/'clips').mkdir(parents=True)
            (output/'previews').mkdir()
            info, digest = inspect_video(source), sha256(source)
            detections = [[(1, [20, 10, 80, 110], .9)]]*20
            segments, ledger, _ = track_video(source, info, Detector(detections))
            log = output/f'{digest[:10]}_detections.jsonl'
            log.write_text('\n'.join(json.dumps(f) for f in ledger), encoding='utf-8')
            rows = plan_rows(source, info, digest, digest, 'UNRESOLVED_CAPTURE_PILOT', segments, ledger, output)
            export_clips(source, info, rows)
            clip = Path(rows[0]['clip_path'])
            before = (sha256(clip), clip.stat().st_mtime_ns)
            with patch('src.person_dataset.ROOT', root), patch('src.person_dataset.YOLO') as detector:
                prepare(output, [source], metadata=[], resume=True)
                detector.assert_not_called()
                self.assertEqual(before, (sha256(clip), clip.stat().st_mtime_ns))
                self.assertEqual(validate_dataset(output/'clips.csv')['clips'], 1)
                with self.assertRaisesRegex(ValueError, 'finalized'):
                    prepare(output, [source], metadata=[], resume=True)

    def test_identity_jump_overlap_low_confidence_and_missing_frames_are_reported(self):
        a, b = [10, 10, 40, 90], [240, 10, 270, 90]
        segments, ledger, events = self.tracked([[(1, a, .9)], [(1, a, .9)],
                                                [(1, b, .9)], [(1, b, .9)]])
        self.assertTrue(any(e['type'] == 'suspected_identity_jump' for e in events))
        self.assertEqual([[p['frame'] for p in s['observations']] for s in segments], [[0, 1], [3]])
        segments, ledger, _ = self.tracked([[(1, a, .9), (2, [30, 10, 60, 90], .9)],
                                          [(1, a, .15)], []])
        self.assertFalse(segments)
        self.assertTrue(all(d['reason'] == 'other_person_in_crop' for d in ledger[0]['detections']))
        self.assertEqual(ledger[1]['detections'][0]['reason'], 'low_detection_confidence')
        self.assertFalse(ledger[2]['detections'])

    def test_validation_blocks_uncertain_short_tampered_and_leaking_rows(self):
        a = [10, 10, 40, 90]
        segments, ledger, _ = self.tracked([[(1, a, .9)]]*100)
        originals = self.rows(segments, ledger)
        rows = [{k: str(v) for k, v in r.items()} for r in originals]
        self.assertTrue(originals[0]['technical_valid'])
        self.assertFalse(originals[1]['technical_valid'])
        self.assertFalse(any(r['training_eligible'] for r in validate_rows(rows, originals)))
        excel_rows = [{**r, 'start_time': format(float(r['start_time']), '.15g'),
                       'end_time': format(float(r['end_time']), '.15g'),
                       'technical_valid': r['technical_valid'].upper(),
                       'training_eligible': 'FALSE'} for r in rows]
        self.assertEqual(len(validate_rows(excel_rows, originals)), len(rows))
        rows[0].update(label='NORMAL', review_status='verified')
        rows[1].update(label='DROWNING', review_status='verified')
        reviewed = validate_rows(rows, originals)
        self.assertTrue(reviewed[0]['training_eligible'])
        self.assertFalse(reviewed[1]['training_eligible'])
        bad = [dict(r) for r in rows]
        bad[0]['valid_frame_count'] = '999'
        with self.assertRaisesRegex(ValueError, 'protected field'):
            validate_rows(bad, originals)
        rows[0]['split'], rows[1]['split'] = 'train', 'test'
        with self.assertRaisesRegex(ValueError, 'group crosses'):
            validate_rows(rows, originals)
        rows[0]['split'] = rows[1]['split'] = 'unassigned'
        with self.assertRaisesRegex(ValueError, 'clip IDs'):
            validate_rows([rows[0], rows[0]], originals)
        rows[0]['label'] = 'butterfly'
        with self.assertRaisesRegex(ValueError, 'invalid label'):
            validate_rows(rows, originals)

    def test_minimum_matches_training_and_usable_tails_are_not_discarded(self):
        def planned(count, fps):
            info = {**self.info(count), 'fps': fps, 'timestamps_sec': [i/fps for i in range(count)],
                    'duration_sec': count/fps}
            points = [{'frame': i, 'time': i/fps, 'bbox': [10, 10, 40, 90], 'confidence': .9}
                      for i in range(count)]
            ledger = [{'detections': [{'track_id': 1, 'bbox': p['bbox']}]} for p in points]
            return self.rows([{'track_id': 1, 'observations': points, 'end_reason': 'end_of_video'}], ledger, info)
        training_info = VideoInfo(25, 46, 46/25, 320, 120)
        clips = build_clip_records({'video_id': 'x', 'path': 'x.mp4', 'style': 'normal'}, training_info)
        self.assertEqual(len(clips), 1)
        self.assertTrue(planned(46, 25)[0]['technical_valid'])
        self.assertFalse(planned(45, 25)[0]['technical_valid'])
        balanced = planned(130, 30)
        self.assertEqual([r['valid_frame_count'] for r in balanced], [65, 65])
        self.assertTrue(all(r['technical_valid'] for r in balanced))
        self.assertEqual(balanced[0]['end_frame']+1, balanced[1]['start_frame'])
        self.assertEqual(len(planned(90, 30)), 1)

    def test_source_hints_are_not_clip_labels_and_filename_groups_are_only_candidates(self):
        self.assertEqual(capture_token(Path('1020 (1)(13)-1.mp4')), capture_token(Path('1020(3)-1.mp4')))
        self.assertNotEqual(capture_token(Path('1020(1)-1.mp4')), capture_token(Path('1021(1)-1.mp4')))
        self.assertEqual(source_category(Path('data/raw/serbest/new.mp4'), []), 'NORMAL')
        self.assertEqual(source_category(Path('data/raw/bogulma/new.mp4'), []), 'DROWNING')
        self.assertEqual(source_category(Path('data/other/new.mp4'), []), 'UNKNOWN')

    def test_inventory_distinguishes_legacy_aliases_new_files_and_duplicate_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root/'data/raw/serbest/old.mp4'
            alias = root/'data/single/serbest/old.mp4'
            new = root/'data/raw/serbest/new.mp4'
            copy = root/'data/raw/serbest/new_copy.mp4'
            for path, content in ((old, b'old'), (alias, b'old'), (new, b'new'), (copy, b'new')):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            previous = root/'results/video_review/master_metadata.csv'
            previous.parent.mkdir(parents=True)
            row = {'path': str(old.relative_to(root)), 'sha256': sha256(old),
                   'candidate_label': 'freestyle', 'visual_hashes': '0000000000000000'}
            write_csv(previous, [row], list(row))
            binary = root/'results/r3d18_binary_drowning/binary_metadata.csv'
            binary.parent.mkdir(parents=True)
            item = {'sha256': sha256(old), 'relation_group': 'OLD', 'initial_training_eligible': 'True',
                    'binary_label': 'normal_swimming'}
            write_csv(binary, [item], list(item))
            write_csv(root/'results/single_metadata.csv', [{'path': str(alias.relative_to(root))}], ['path'])
            before = sha256(previous)
            with patch('src.person_dataset_update.ROOT', root), \
                 patch('src.person_dataset_update.probe_video', return_value=VideoInfo(25, 46, 1.84, 320, 120)), \
                 patch('src.person_dataset_update.inspect_video', return_value={'duration_sec': 1.84}) as decode, \
                 patch('src.person_dataset_update.visual_hashes', return_value=['ffffffffffffffff']):
                inventory(root/'results/update')
                self.assertEqual(decode.call_count, 1)
            result = json.loads((root/'results/update/inventory_report.json').read_text())
            self.assertEqual(result['previous_source_inventory_count'], 1)
            self.assertEqual(result['new_video_files'], 2)
            self.assertEqual(result['new_unique_content'], 1)
            self.assertEqual(result['duplicate_file_copies'], 2)
            self.assertEqual(result['all_video_files_under_data'], 4)
            self.assertEqual(result['new_source_category_hints'], {'NORMAL': 2})
            self.assertEqual(sha256(previous), before)

    def test_batch_export_decodes_source_once_and_letterboxing_fills_small_crops(self):
        from PIL import Image
        image = person_image(SimpleNamespace(to_image=lambda: Image.new('RGB', (100, 100), 'white')),
                             [10, 10, 20, 30])
        self.assertEqual(image.getbbox(), (80, 0, 240, 320))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output/'clips').mkdir()
            (output/'previews').mkdir()
            source = output/'source.mp4'
            make_video(source, count=20)
            info = inspect_video(source)
            points = [{'frame': i, 'time': t, 'bbox': [20, 10, 80, 110], 'confidence': .9}
                      for i, t in enumerate(info['timestamps_sec'])]
            ledger = [{'detections': [{'track_id': 1, 'bbox': p['bbox']}]} for p in points]
            row = self.rows([{'track_id': 1, 'observations': points, 'end_reason': 'end_of_video'}],
                            ledger, info, output, source)[0]
            short = {**row, 'end_frame': 3, 'valid_frame_count': 4,
                     'clip_path': str(output/'clips/short.mp4'), 'preview_path': str(output/'previews/short.jpg')}
            with patch('src.person_dataset.decoded_frames', wraps=decoded_frames) as decode:
                export_clips(source, info, [row, short])
                self.assertEqual(decode.call_count, 1)
            self.assertEqual(inspect_video(Path(row['clip_path']))['frame_count'], 20)
            self.assertEqual(inspect_video(Path(short['clip_path']))['frame_count'], 4)

    def test_real_mp4_exports_keep_original_frames_and_csv_asset_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output/'clips').mkdir()
            (output/'previews').mkdir()
            source = output/'source.mp4'
            make_video(source, count=20)
            info = inspect_video(source)
            observations = [{'frame': i, 'time': t, 'bbox': [20, 10, 80, 110], 'confidence': .9}
                            for i, t in enumerate(info['timestamps_sec'])]
            ledger = [{'detections': [{'track_id': 1, 'bbox': p['bbox']}]} for p in observations]
            segments = [{'track_id': 1, 'observations': observations, 'end_reason': 'end_of_video'}]
            rows = self.rows(segments, ledger, info, output, source)
            export_clip(source, info, rows[0])
            result = inspect_video(Path(rows[0]['clip_path']))
            self.assertEqual(result['frame_count'], 20)
            np.testing.assert_allclose(result['timestamps_sec'], info['timestamps_sec'], atol=.0001)
            (output/'manifest.json').write_text(json.dumps({'rows': rows,
                'protected_sha256': {rows[0]['clip_path']: sha256(Path(rows[0]['clip_path']))}}))
            write_csv(output/'clips.csv', rows)
            self.assertEqual(validate_dataset(output/'clips.csv')['training_eligible'], 0)
            short = {**rows[0], 'clip_id': 'short', 'end_frame': 3, 'end_time': .5,
                     'valid_frame_count': 4, 'clip_path': str(output/'clips/short.mp4')}
            export_clip(source, info, short)
            self.assertEqual(inspect_video(Path(short['clip_path']))['frame_count'], 4)
            Path(rows[0]['clip_path']).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'Missing or modified'):
                validate_dataset(output/'clips.csv')


if __name__ == '__main__':
    unittest.main()
