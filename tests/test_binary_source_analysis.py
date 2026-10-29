import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.video_review.binary_source_analysis import (
    assess_practical_limits, filename_hints, investigate, partition_counts, run, split_feasibility,
    unique_inventory, validate_clip_assignments, validate_partition,
)


def row(index, label='normal', group=None):
    return {'video_id': f'V{index}', 'sha256': f'hash{index}', 'class_label': label,
            'source_id': 'unknown', 'session_id': 'unknown',
            'confirmed_source_group': 'unknown', 'leakage_group': group or f'G{index}',
            'relation_group': f'R{index}', 'filename': f'clip{index}.mp4',
            'source_video_path': f'data/raw/serbest/clip{index}.mp4',
            'visual_hashes': '|'.join(['0000000000000000'] * 5),
            'width': '1920', 'height': '1080', 'duration_sec': '10', 'fps': '30'}


class SourceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(1, 'drowning'), row(2), row(3, 'drowning'), row(4)]
        self.assignments = [{'video_id': r['video_id'], 'split': 'train' if i < 2 else 'validation'}
                            for i, r in enumerate(self.rows)]

    def test_valid_partition_counts_each_content_once(self):
        counts = partition_counts(self.rows, self.assignments)
        self.assertEqual(sum(c['videos'] for c in counts.values()), 4)
        self.assertEqual(counts['validation']['drowning'], 1)

    def test_same_video_cannot_be_assigned_twice_even_in_same_partition(self):
        for split in ('train', 'test'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_partition(self.rows, self.assignments + [{'video_id': 'V1', 'split': split}])

    def test_duplicate_inventory_video_rejected(self):
        with self.assertRaises(ValueError):
            unique_inventory(self.rows + [self.rows[0]])

    def test_exact_duplicate_alias_cannot_enter_another_partition(self):
        self.rows[2]['sha256'] = self.rows[0]['sha256']
        with self.assertRaisesRegex(ValueError, 'duplicate content'):
            validate_partition(self.rows, self.assignments)

    def test_verified_source_and_unresolved_groups_cannot_cross_partitions(self):
        for field in ('source_id', 'confirmed_source_group', 'leakage_group', 'relation_group'):
            with self.subTest(field=field):
                rows = [dict(r) for r in self.rows]
                rows[0][field] = rows[2][field] = 'SHARED'
                with self.assertRaisesRegex(ValueError, 'crosses partitions'):
                    validate_partition(rows, self.assignments)

    def test_same_verified_source_may_stay_together(self):
        self.rows[0]['confirmed_source_group'] = self.rows[1]['confirmed_source_group'] = 'SOURCE1'
        validate_partition(self.rows, self.assignments)

    def test_missing_foreign_ids_and_unknown_partitions_rejected(self):
        for assignments in (self.assignments[:-1], self.assignments + [{'video_id': 'FOREIGN', 'split': 'test'}],
                            [{'video_id': r['video_id'], 'split': 'unassigned'} for r in self.rows]):
            with self.assertRaises(ValueError):
                validate_partition(self.rows, assignments)

    def test_clips_must_inherit_parent_split(self):
        clips = [{'video_id': 'V1', 'clip_id': 'V1_C1', 'split': 'train'},
                 {'video_id': 'V1', 'clip_id': 'V1_C2', 'split': 'train'}]
        validate_clip_assignments(clips, self.assignments)
        clips[1]['split'] = 'validation'
        with self.assertRaises(ValueError):
            validate_clip_assignments(clips, self.assignments)

    def test_duplicate_clip_and_foreign_parent_rejected(self):
        clip = {'video_id': 'V1', 'clip_id': 'C1', 'split': 'train'}
        with self.assertRaises(ValueError):
            validate_clip_assignments([clip, clip], self.assignments)
        with self.assertRaises(ValueError):
            validate_clip_assignments([{**clip, 'video_id': 'unknown'}], self.assignments)

    def test_filename_clues_do_not_turn_class_numbering_into_provenance(self):
        self.assertEqual(filename_hints('bogulma27.mp4'), [])
        self.assertEqual(filename_hints('serbest1.mp4'), [])
        self.assertIn('numeric_recording_prefix:1026', filename_hints('1026 (3)-1.mp4'))
        self.assertIn('filename_date_candidate:16102026', filename_hints('Person_16102026 - Trim.mp4'))
        self.assertNotIn('filename_date_candidate:99139999', filename_hints('Person_99139999.mp4'))

    def test_new_suspicions_are_grouped_without_confirming_source(self):
        rows = [dict(r) for r in self.rows[:2]]
        rows[0]['filename'], rows[1]['filename'] = '1026 (1).mp4', '1026 (2).mp4'
        metadata = {r['video_id']: {} for r in rows}
        enriched, report = investigate(rows, {'suspicious_similar_pairs': [], 'exact_duplicate_groups': []},
                                       metadata, visual_groups={})
        self.assertEqual(len({r['leakage_group'] for r in enriched}), 1)
        self.assertTrue(all(r['confirmed_source_group'] == 'unknown' for r in enriched))
        self.assertEqual(report['unresolved_relationship_count'], 1)
        self.assertEqual(report['relationships'][0]['same_long_video'], 'unknown')
        self.assertFalse(report['relationships'][0]['confirmed_duplicate'])

    def test_transitive_suspicions_stay_together_and_bad_scores_fail(self):
        rows = self.rows[:3]
        for r in rows:
            r['relation_group'] = 'ORIGINAL'
        pairs = [{'video_id_a': 'V1', 'video_id_b': 'V2', 'min_visual_hash_distance': 0},
                 {'video_id_a': 'V2', 'video_id_b': 'V3', 'min_visual_hash_distance': 0}]
        dupes = {'suspicious_similar_pairs': pairs, 'exact_duplicate_groups': []}
        metadata = {r['video_id']: {} for r in rows}
        enriched, _ = investigate(rows, dupes, metadata, visual_groups={})
        self.assertEqual(len({r['leakage_group'] for r in enriched}), 1)
        pairs[0]['min_visual_hash_distance'] = 3
        with self.assertRaises(ValueError):
            investigate(rows, dupes, metadata, visual_groups={})

    def test_three_and_five_fold_feasibility_without_saving_assignments(self):
        rows = [row(i * 2 + j, ('normal', 'drowning')[j], f'G{i}') for i in range(8) for j in range(2)]
        result = split_feasibility(rows)
        self.assertEqual(result['group_distribution']['class_bearing_group_counts']['drowning'], 8)
        self.assertTrue(all(m['status'] == 'draft_feasible' for m in result['methods']))
        for method in result['methods']:
            self.assertFalse(method['independent_evaluation'])
            if 'folds' in method:
                self.assertEqual(sum(f['counts']['validation']['videos'] for f in method['folds']), len(rows))
                self.assertTrue(all(f['counts']['validation']['drowning'] > 0 for f in method['folds']))

    def test_one_drowning_group_cannot_support_class_preserving_splits(self):
        rows = [row(i, 'drowning' if i < 3 else 'normal', 'POS' if i < 3 else f'G{i}') for i in range(9)]
        result = split_feasibility(rows)
        self.assertEqual(Counter(m['status'] for m in result['methods']), {'structurally_infeasible': 5})

    def test_class_presence_does_not_mean_balanced_or_independent(self):
        rows = [row(i, 'drowning' if i % 2 else 'normal', 'BIG' if i < 10 else f'G{i}')
                for i in range(16)]
        scenario = split_feasibility(rows)
        report = {'class_counts': {'drowning': 8, 'normal': 8}, 'unique_video_count': 16,
                  'original_dhash_only_scenario': scenario,
                  'expanded_precautionary_scenario': split_feasibility(rows)}
        text = assess_practical_limits(report)
        self.assertEqual(report['expanded_precautionary_scenario']['group_distribution']['largest_drowning_group_fraction'], 5 / 8)
        self.assertTrue(all(m['equal_size_folds_impossible'] for m in scenario['methods'] if 'n_splits' in m))
        self.assertIn('pratik eğitim hazırlığı değildir', text)
        self.assertIn('Defer', report['current_decision'])

    def test_existing_reports_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'results/aquasafe_binary_v1'
            output.mkdir(parents=True)
            report = output / 'split_recommendation.md'
            report.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                run(root)
            self.assertEqual(report.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
