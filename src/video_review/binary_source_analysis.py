"""Stage 2A: source hypotheses and split feasibility, without assigning final splits."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from math import ceil
from pathlib import Path

import av
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

from src.video_classification.binary_drowning import Groups
from src.video_classification.prepare_binary_v1 import normalized, write_csv
from src.video_review.inventory import hamming, sha256_file


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/aquasafe_binary_v1'
UNKNOWN = {'', 'unknown', 'unassigned', None}
REPORTS = ['similarity_analysis.json', 'source_group_analysis.csv',
           'split_feasibility.json', 'split_recommendation.md']
# Midpoint frames were visually inspected on 2026-10-29. These are camera/context
# hypotheses only, not identities, event labels, confirmed sources or sessions.
VISUAL_REVIEW_GROUPS = {
    'side_wall_and_lane_view': [
        *[f'bogulma/bogulma{i}.mp4' for i in range(3, 10)],
        'kurbagalama/kurbaglama1.mp4', 'kurbagalama/kurbaglama11.mp4',
        'kurbagalama/kurbaglama2.mp4', 'kurbagalama/kurbaglama5.mp4',
        'serbest/serbest17.mp4', 'serbest/serbest18.mp4', 'serbest/serbest19.mp4',
        'sirtustu/sirtustu12.mp4', 'sirtustu/sirtustu13.mp4', 'sirtustu/sirtustu14.mp4'],
    'frontal_lane_end_markings': [
        'bogulma/bogulma10.mp4', 'bogulma/bogulma12.mp4', 'bogulma/bogulma14.mp4',
        'bogulma/bogulma16.mp4', 'bogulma/bogulma25.mp4',
        'kelebek/kelebek1.mp4', 'kelebek/kelebek2.mp4',
        'kurbagalama/kurbaglama16.mp4', 'kurbagalama/kurbaglama18.mp4',
        'serbest/serbest1.mp4', 'serbest/serbest20.mp4',
        'sirtustu/sirtustu17.mp4'],
    'frontal_sunlit_right_wall': [
        'bogulma/bogulma1.mp4', 'bogulma/bogulma2.mp4',
        'kurbagalama/kurbaglama10.mp4', 'kurbagalama/kurbaglama4.mp4',
        'serbest/serbest6.mp4', 'sirtustu/sirtustu8.mp4'],
    'upward_lane_perspective': [
        'bogulma/bogulma18.mp4', 'bogulma/bogulma19.mp4',
        'bogulma/bogulma20.mp4', 'bogulma/bogulma22.mp4',
        'kurbagalama/kurbaglama21.mp4', 'kurbagalama/kurbaglama26.mp4',
        'serbest/serbest14.mp4', 'serbest/serbest15.mp4'],
}


def unique_inventory(rows):
    if not rows or len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Inventory must contain each video ID exactly once')
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Inventory contains exact duplicate content')


def validate_partition(rows, assignments):
    """Validate one hypothetical partition; unknown source IDs are never independence evidence."""
    unique_inventory(rows)
    by_id = {r['video_id']: r for r in rows}
    ids = [a['video_id'] for a in assignments]
    if Counter(ids) != Counter({key: 1 for key in by_id}):
        raise ValueError('Each video must be assigned exactly once, without missing or foreign IDs')
    seen = defaultdict(dict)
    for item in assignments:
        row, split = by_id[item['video_id']], item['split']
        if split not in {'train', 'validation', 'test'}:
            raise ValueError('Invalid partition name')
        for field in ('sha256', 'source_id', 'confirmed_source_group', 'leakage_group', 'relation_group'):
            key = row.get(field)
            if key in UNKNOWN:
                continue
            if key in seen[field] and seen[field][key] != split:
                raise ValueError(f'{field} crosses partitions: {key}')
            seen[field][key] = split


def validate_clip_assignments(clips, parent_assignments):
    ids = [r['video_id'] for r in parent_assignments]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate parent assignment')
    parents = {r['video_id']: r['split'] for r in parent_assignments}
    if len({r['clip_id'] for r in clips}) != len(clips):
        raise ValueError('Duplicate clip ID')
    for clip in clips:
        if clip['video_id'] not in parents or clip['split'] != parents[clip['video_id']]:
            raise ValueError('Clip does not inherit its parent video partition')


def filename_hints(filename):
    stem = normalized(Path(filename).stem)
    hints = []
    prefix = re.match(r'^(\d{4})(?=\s|\(|-|$)', stem)
    if prefix:
        hints.append('numeric_recording_prefix:' + prefix[1])
    for date in re.findall(r'(?<!\d)\d{8}(?!\d)', stem):
        try:
            datetime.strptime(date, '%d%m%Y')
            hints.append('filename_date_candidate:' + date)
        except ValueError:
            pass
    # Generic class numbering (bogulma1, serbest2...) conveys no source identity.
    if re.search(r'\d{8}', stem):
        base = re.sub(r'\s*-?\s*trim$', '', stem).strip()
        hints.append('named_recording_candidate:' + base)
    return hints


def distribution(rows, group_field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_field]].append(row)
    details = [{'group_id': key, 'video_count': len(bundle),
                'class_counts': dict(Counter(r['class_label'] for r in bundle)),
                'video_ids': [r['video_id'] for r in bundle]}
               for key, bundle in sorted(groups.items())]
    return {'group_count': len(groups), 'independence_verified': False,
            'class_bearing_group_counts': {label: sum(any(r['class_label'] == label for r in bundle)
                                                      for bundle in groups.values())
                                           for label in ('drowning', 'normal')},
            'largest_group_videos': max(map(len, groups.values())), 'groups': details}


def investigate(rows, duplicates, metadata, visual_groups=VISUAL_REVIEW_GROUPS):
    unique_inventory(rows)
    by_id = {r['video_id']: r for r in rows}
    pairs = {}

    def link(left, right, method):
        if left == right:
            return
        key = tuple(sorted((left, right)))
        pairs.setdefault(key, set()).add(method)

    for pair in duplicates['suspicious_similar_pairs']:
        left, right = pair['video_id_a'], pair['video_id_b']
        if left not in by_id or right not in by_id or left == right:
            raise ValueError('Invalid similarity endpoint')
        distance = min(hamming(a, b) for a in by_id[left]['visual_hashes'].split('|')
                       for b in by_id[right]['visual_hashes'].split('|'))
        if distance != pair['min_visual_hash_distance'] or distance > 4:
            raise ValueError('Stored dHash score does not match the manifest')
        if by_id[left]['relation_group'] != by_id[right]['relation_group']:
            raise ValueError('Existing relation_group splits a suspicious pair')
        link(left, right, 'stage1_min_sample_dhash_le_4')
    hints, visual_notes = defaultdict(list), []
    for row in rows:
        for hint in filename_hints(row['filename']):
            hints[hint].append(row['video_id'])
        creation = metadata[row['video_id']].get('creation_time', '')
        if creation:
            hints['identical_container_creation_time:' + creation].append(row['video_id'])
    for hint, members in hints.items():
        for a, b in combinations(members, 2):
            link(a, b, hint)
    by_path = {r['source_video_path'].removeprefix('data/raw/'): r['video_id'] for r in rows}
    for name, paths in visual_groups.items():
        members = [by_path[p] for p in paths if p in by_path]
        for a, b in combinations(members, 2):
            link(a, b, 'midpoint_visual_context:' + name)
        visual_notes.append({'hypothesis': name, 'video_ids': members, 'same_source': 'unknown',
                             'evidence_scope': 'Agent visual inspection of one midpoint frame per video; shared wall/lane/camera geometry, no temporal continuity verification.'})
    groups = Groups(list(by_id))
    for row in rows:
        for other in rows:
            if row['relation_group'] == other['relation_group']:
                groups.union(row['video_id'], other['video_id'])
            if row.get('source_id') not in UNKNOWN and row.get('source_id') == other.get('source_id'):
                groups.union(row['video_id'], other['video_id'])
    for left, right in pairs:
        groups.union(left, right)
    analysis = []
    for (left, right), methods in sorted(pairs.items()):
        a, b = by_id[left], by_id[right]
        ah, bh = a['visual_hashes'].split('|'), b['visual_hashes'].split('|')
        distances = [(hamming(x, y), i, j) for i, x in enumerate(ah) for j, y in enumerate(bh)]
        best, ai, bi = min(distances)
        analysis.append({
            'video_id_a': left, 'video_id_b': right,
            'path_a': a['source_video_path'], 'path_b': b['source_video_path'],
            'class_a': a['class_label'], 'class_b': b['class_label'],
            'relation_group_a': a['relation_group'], 'relation_group_b': b['relation_group'],
            'leakage_group': groups.find(left), 'similarity_methods': sorted(methods),
            'min_dhash_distance': best, 'distance_range': [0, 64], 'lower_is_more_similar': True,
            'equal_dimensions': (a['width'], a['height']) == (b['width'], b['height']),
            'aligned_sample_mean_dhash_distance': sum(hamming(x, y) for x, y in zip(ah, bh)) / len(ah),
            'sample_pairs_with_distance_le_4': sum(d <= 4 for d, _, _ in distances),
            'best_sample_fraction_a': (ai + 0.5) / len(ah), 'best_sample_fraction_b': (bi + 0.5) / len(bh),
            'duration_sec_a': float(a['duration_sec']), 'duration_sec_b': float(b['duration_sec']),
            'duration_ratio_short_to_long': min(float(a['duration_sec']), float(b['duration_sec'])) / max(float(a['duration_sec']), float(b['duration_sec'])),
            'fps_a': float(a['fps']), 'fps_b': float(b['fps']),
            'creation_time_a': metadata[left].get('creation_time', ''),
            'creation_time_b': metadata[right].get('creation_time', ''),
            'same_source': 'unknown', 'same_shoot': 'unknown', 'same_long_video': 'unknown',
            'confirmed_duplicate': False, 'review_required': True,
        })
    enriched = [{**r, 'leakage_group': groups.find(r['video_id']),
                 'confirmed_source_group': r['source_id'] if r['source_id'] not in UNKNOWN else 'unknown',
                 'source_relationship_status': 'unknown',
                 'filename_source_hints': '|'.join(filename_hints(r['filename'])),
                 'container_creation_time': metadata[r['video_id']].get('creation_time', ''),
                 'container_encoder': metadata[r['video_id']].get('encoder', ''),
                 'metadata_error': metadata[r['video_id']].get('error', ''),
                 'final_split': 'unassigned'} for r in rows]
    report = {
        'original_suspicious_pair_count': len(duplicates['suspicious_similar_pairs']),
        'original_method': 'Five 64-bit grayscale dHashes at 10/30/50/70/90 percent of each video. Compare all 25 frame pairs only at equal resolution; minimum Hamming distance <=4.',
        'score_interpretation': 'Bit difference count, not same-source probability. One matching frame is enough; no sequence alignment or overlap proof.',
        'limitations': ['Shared static pool backgrounds can create false positives.',
                       'Different resolution, crops, short overlaps and unsampled frames can hide real relationships.',
                       'Creation time may be export/re-encode time; encoder, duration and filename sequences do not prove source/session.',
                       'No midpoint preview establishes event truth, swimmer identity or temporal continuity.'],
        'original_pairs_cross_class': sum(p['class_a'] != p['class_b'] for p in analysis if 'stage1_min_sample_dhash_le_4' in p['similarity_methods']),
        'all_candidate_pair_count': len(analysis), 'unresolved_relationship_count': len(analysis),
        'confirmed_same_source_pairs_between_unique_videos': 0,
        'exact_duplicate_file_groups': duplicates['exact_duplicate_groups'],
        'exact_duplicate_note': 'SHA256 verifies four alias groups, not four independent acquisition sources. Aliases inherit canonical content ID and any future split.',
        'visual_review': visual_notes,
        'broad_context_observation': 'All 138 midpoint previews show similar underwater pool/lane contexts. A common facility is plausible but unknown; even disconnected leakage groups cannot be called independent sources.',
        'relationships': analysis,
    }
    return enriched, report


def partition_counts(rows, assignments):
    validate_partition(rows, assignments)
    labels = {r['video_id']: r['class_label'] for r in rows}
    result = {}
    for split in sorted({a['split'] for a in assignments}):
        ids = {a['video_id'] for a in assignments if a['split'] == split}
        counts = Counter(labels[i] for i in ids)
        result[split] = {'videos': len(ids), 'drowning': counts['drowning'], 'normal': counts['normal']}
    return result


def split_feasibility(rows):
    stats = distribution(rows, 'leakage_group')
    minimum = min(stats['class_bearing_group_counts'].values())
    labels = [r['class_label'] for r in rows]
    group_ids = [r['leakage_group'] for r in rows]
    methods = []
    for name, factory in [('stratified_group_kfold', lambda k: StratifiedGroupKFold(k, shuffle=True, random_state=42)),
                          ('group_kfold', lambda k: GroupKFold(k))]:
        for k in (3, 5):
            item = {'method': name, 'n_splits': k, 'status': 'structurally_infeasible',
                    'independent_evaluation': False}
            if minimum >= k:
                validation_seen, folds = [], []
                for fold, (train, validation) in enumerate(factory(k).split(rows, labels, group_ids), 1):
                    assignments = [{'video_id': rows[int(i)]['video_id'], 'split': 'train'} for i in train]
                    assignments += [{'video_id': rows[int(i)]['video_id'], 'split': 'validation'} for i in validation]
                    counts = partition_counts(rows, assignments)
                    validation_seen.extend(a['video_id'] for a in assignments if a['split'] == 'validation')
                    folds.append({'fold': fold, 'counts': counts})
                if Counter(validation_seen) != Counter(r['video_id'] for r in rows):
                    raise ValueError('Each video must be validation data exactly once per CV plan')
                item.update(status='draft_feasible' if all(all(v['drowning'] and v['normal'] for v in f['counts'].values()) for f in folds) else 'class_missing_in_fold', folds=folds)
            else:
                item['reason'] = f'Only {minimum} class-bearing groups for the rarest class; {k} class-preserving folds are impossible.'
            methods.append(item)
    holdout = {'method': 'group_aware_train_validation_test', 'target_ratios': [0.70, 0.15, 0.15],
               'status': 'structurally_infeasible', 'independent_evaluation': False}
    if minimum >= 3:
        total = Counter(labels)
        best = None
        for train, hold in GroupShuffleSplit(100, train_size=0.70, random_state=42).split(rows, labels, group_ids):
            held_groups = [group_ids[int(i)] for i in hold]
            if len(set(held_groups)) < 2:
                continue
            va, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=42).split(hold, groups=held_groups))
            assignments = [{'video_id': rows[int(i)]['video_id'], 'split': 'train'} for i in train]
            assignments += [{'video_id': rows[int(hold[int(i)])]['video_id'], 'split': split}
                            for indices, split in ((va, 'validation'), (te, 'test')) for i in indices]
            counts = partition_counts(rows, assignments)
            if not all(c['drowning'] and c['normal'] for c in counts.values()):
                continue
            error = sum(abs(counts[s][c] / total[c] - ratio) for s, ratio in zip(('train', 'validation', 'test'), (0.70, 0.15, 0.15)) for c in total)
            if best is None or error < best[0]:
                best = error, counts
        if best:
            holdout.update(status='draft_feasible', counts=best[1], ratio_absolute_error=best[0], search='Best class-ratio error among 100 seeded group-shuffle proposals; ratios refer to videos, not group counts.')
        else:
            holdout.update(status='no_class_preserving_proposal_found', reason='Search failure is not a proof of mathematical impossibility.')
    else:
        holdout['reason'] = f'Only {minimum} class-bearing groups for the rarest class; all three partitions cannot contain both classes.'
    methods.append(holdout)
    return {'group_distribution': stats, 'methods': methods}


def assess_practical_limits(feasibility):
    """Class presence alone is insufficient evidence of a useful balanced evaluation."""
    feasibility['draft_feasible_definition'] = 'Only structural: no known constraint crosses partitions and both classes are present. Does not mean balanced, source-independent or ready for training.'
    feasibility['current_decision'] = 'Defer splitting; review provenance and the largest unresolved component first.'
    feasibility['recommended_method'] = 'After source verification and a new balance check, prefer 3-fold StratifiedGroupKFold only if verified source groups support meaningful folds; otherwise collect independent sources. No current method establishes an independent test.'
    for name in ('original_dhash_only_scenario', 'expanded_precautionary_scenario'):
        scenario = feasibility[name]
        stats = scenario['group_distribution']
        positives = sorted((g['class_counts'].get('drowning', 0) for g in stats['groups']), reverse=True)
        stats['drowning_videos_per_positive_group'] = [p for p in positives if p]
        stats['largest_drowning_group_fraction'] = positives[0] / feasibility['class_counts']['drowning']
        for method in scenario['methods']:
            if 'n_splits' in method:
                k = method['n_splits']
                method['ideal_validation_videos'] = feasibility['unique_video_count'] / k
                method['largest_indivisible_group_videos'] = stats['largest_group_videos']
                method['equal_size_folds_impossible'] = stats['largest_group_videos'] > ceil(feasibility['unique_video_count'] / k)
    expanded = feasibility['expanded_precautionary_scenario']
    lines = ['\n## Mevcut taslakların pratik sınırı\n',
             'Aşağıdaki sayılar yalnız fizibilite simülasyonudur; final atamalar değildir.\n',
             '| Yöntem | Validasyon video sayıları | Validasyon drowning sayıları |',
             '|---|---|---|']
    for method in expanded['methods']:
        if method.get('folds'):
            counts = [f['counts']['validation'] for f in method['folds']]
            lines.append(f"| {method['method']} ({method['n_splits']}) | {', '.join(str(c['videos']) for c in counts)} | {', '.join(str(c['drowning']) for c in counts)} |")
    stats = expanded['group_distribution']
    lines += ['', f"En büyük bölünemez aday grup {stats['largest_group_videos']} video içerir; pozitiflerin en büyük gruptaki payı %{100 * stats['largest_drowning_group_fraction']:.1f}. "
              'Grup bütünlüğü korunurken yaklaşık eşit büyüklükte 3 veya 5 fold elde edilemez. '
              'Stratification bunu çözemez. draft_feasible yalnız iki sınıfın bulunmasını ve bilinen kısıtların korunmasını ifade eder; pratik eğitim hazırlığı değildir. '
              'Önce bu büyük bileşenin ilişkileri incelenmeli; kanıt olmadan grup parçalanmamalıdır.\n']
    return '\n'.join(lines)


def run(root=ROOT):
    root = Path(root).resolve()
    output = root / 'results/aquasafe_binary_v1'
    if any((output / name).exists() for name in REPORTS):
        raise FileExistsError('A Stage 2A report already exists; refusing to overwrite')
    inputs = [output / name for name in ('dataset_manifest.csv', 'duplicate_report.json', 'dataset_summary.json', 'data_readiness_report.json', 'missing_yolo_outputs.csv')]
    before = {str(p): sha256_file(p) for p in inputs}
    with inputs[0].open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    duplicates = json.loads(inputs[1].read_text(encoding='utf-8'))
    unique_inventory(rows)
    metadata = {}
    for row in rows:
        path = (root / row['source_video_path']).resolve()
        if not path.is_relative_to(root / 'data/raw') or sha256_file(path) != row['sha256']:
            raise ValueError('Raw source does not match Stage 1 content: ' + str(path))
        try:
            with av.open(str(path)) as container:
                metadata[row['video_id']] = dict(container.metadata)
        except Exception as error:
            metadata[row['video_id']] = {'error': str(error)}
    for group in duplicates['exact_duplicate_groups']:
        for alias in group['members']:
            path = (root / alias['source_video_path']).resolve()
            if not path.is_relative_to(root / 'data/raw') or sha256_file(path) != group['sha256']:
                raise ValueError('Exact duplicate alias changed: ' + str(path))
    enriched, similarity = investigate(rows, duplicates, metadata)
    baseline = [{**r, 'leakage_group': r['relation_group']} for r in rows]
    feasibility = {
        'stage': '2A', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': before, 'unique_video_count': len(rows),
        'class_counts': dict(Counter(r['class_label'] for r in rows)),
        'confirmed_independent_source_groups': 0, 'independent_source_count': 'unknown',
        'drowning_confirmed_source_count': 'unknown', 'source_unknown_videos': sum(r['source_id'] in UNKNOWN for r in rows),
        'original_dhash_only_scenario': split_feasibility(baseline),
        'expanded_precautionary_scenario': split_feasibility(enriched),
        'common_facility_sensitivity_scenario': {'status': 'unverified_hypothesis', 'group_count_if_true': 1,
                                               'class_preserving_split_possible_if_true': False},
        'metadata_creation_times_present': sum(bool(m.get('creation_time')) for m in metadata.values()),
        'metadata_errors': {key: m['error'] for key, m in metadata.items() if m.get('error')},
        'risk_level': 'high_unquantified', 'reliable_test_set_available': False,
        'final_assignments_saved': False, 'draft_assignments_saved': False,
        'proposal_note': 'Only feasibility counts saved, no split/fold assignment becomes training metadata.',
        'recommended_method': 'After provenance review, prefer 3-fold StratifiedGroupKFold if at least three verified source groups contain drowning and each fold retains both classes; otherwise collect independent sources.',
        'required_checks': ['Clips inherit parent video split.', 'Each content hash counted once; aliases inherit canonical split.',
                            'Verified sources stay together.', 'All unresolved candidate edges stay together.',
                            'Singleton/disconnected candidate groups are not assumed independent.',
                            'Check both classes in every train/validation/test partition and CV validation membership exactly once.'],
        'training_started': False, 'yolo_started': False, 'previous_experiment_results_used': False,
    }
    practical_note = assess_practical_limits(feasibility)
    similarity['input_sha256'] = before
    fields = ['video_id', 'source_video_path', 'filename', 'class_label', 'sha256', 'alias_count',
              'duration_sec', 'fps', 'width', 'height', 'relation_group', 'leakage_group',
              'source_id', 'session_id', 'confirmed_source_group', 'source_relationship_status',
              'filename_source_hints', 'container_creation_time', 'container_encoder', 'metadata_error', 'final_split']
    if before != {str(p): sha256_file(p) for p in inputs}:
        raise RuntimeError('Stage 1 input reports changed during analysis')
    write_csv(output / REPORTS[1], [{k: r[k] for k in fields} for r in enriched], fields)
    for name, content in ((REPORTS[0], similarity), (REPORTS[2], feasibility)):
        with (output / name).open('x', encoding='utf-8') as target:
            json.dump(content, target, ensure_ascii=False, indent=2, allow_nan=False)
    base = feasibility['original_dhash_only_scenario']['group_distribution']
    expanded = feasibility['expanded_precautionary_scenario']['group_distribution']
    text = f'''# Aqua-Safe Aşama 2A — bölme önerisi

138 benzersiz video (27 drowning, 111 normal) incelendi. Doğrulanmış bağımsız
kaynak grubu: **0**; gerçek kaynak sayısı **unknown**. Birebir kopyalar kaynak
bağımsızlığı kanıtı değildir. Mevcut veri kümesi ve ilk beş rapor değiştirilmedi.

## İlişkiler ve belirsizlik

İlk 139 çift: aynı çözünürlükte beş örnek karenin 25 karşılaştırmasında minimum
64-bit dHash Hamming mesafesi ≤4. Skor kaynak olasılığı değildir. Bu çiftlerin
{similarity['original_pairs_cross_class']} tanesi sınıflar arasıdır; sınıf farkı kaynak bağımsızlığı sağlamaz.
İlk 139 çiftin tamamında aynı kaynak/çekim/uzun video ilişkisi unknown kaldı.
Dosya adı, eşit container creation_time ve seçili kamera bağlamı ipuçlarıyla toplam
**{similarity['all_candidate_pair_count']} belirsiz çift** raporlandı. Aynı uzun kayıttan kesit
olduklarına dair zamansal süreklilik kanıtı bulunmadı; ilişkiler kesin kopya sayılmadı.

138 videonun orta kare önizlemeleri incelendi. Görsel hipotezlerin üye listeleri
similarity_analysis.json içindedir; bu listeler doğrulanmış kaynak/oturum değildir.
Tüm önizlemelerde benzer sualtı havuz/kulvar bağlamı var. Olası ortak tesis nedeniyle
bağlantısız gruplar da bağımsız kabul edilemez. Süre/FPS/encoder ve creation_time
yalnız destekleyici ipuçlarıdır; özellikle yeniden kodlama zamanı çekim zamanı değildir.

## Grup dağılımı

| Senaryo | İhtiyati grup | Boğulma içeren grup | En büyük grup |
|---|---:|---:|---:|
| Yalnız ilk dHash ilişkileri | {base['group_count']} | {base['class_bearing_group_counts']['drowning']} | {base['largest_group_videos']} |
| Dosya adı/metadata/görsel ipuçlarıyla genişletilmiş | {expanded['group_count']} | {expanded['class_bearing_group_counts']['drowning']} | {expanded['largest_group_videos']} |

Bu sayılar bağımsız kaynak sayısı değildir. Grup başına boğulma/normal sayıları ve
her yöntemin fold dağılımları split_feasibility.json dosyasındadır.

## Yöntemlerin değerlendirmesi

- Group-aware train/validation/test: yaklaşık 70/15/15 oranı ve her bölümde iki sınıf
  hedeflenerek değerlendirildi. Az sayıda boğulma kaynağını tek bir test bölümüne
  ayırmak tahmini oynak yapar. Güvenilir bağımsız test oluşturulmadı.
- Stratified group k-fold: 3 ve 5 fold için yapısal fizibilite incelendi. Gruplar
  bölünmeden sınıf dengesini gözetir; kaynaklar doğrulandıktan sonra yeterli sayıda
  boğulma kaynağı varsa **3-fold** tercih edilir. Beş fold küçük pozitif sayıları ve
  yüksek belirsizlik nedeniyle ikinci plandadır. Taslak sayılar bağımsızlık kanıtı değildir.
- Group k-fold: 3 ve 5 fold incelendi; grup bütünlüğünü korur fakat sınıf oranını
  hedeflemez. İki sınıfın her fold'da bulunduğu ayrıca denetlendiği için ancak
  deneysel bir alternatif olabilir.

Grup sayısının yetersiz olduğu senaryoda bu yöntemler sınıf koruyarak uygulanamaz.
Salt dHash senaryosu, yeni belirsiz ilişkileri yok sayarak nihai bölme olarak kullanılamaz.
Ortak tesis hipotezi doğruysa tek bir tesis grubu kalır ve bağımsız bölme yapılamaz.

## Karar ve sonraki engeller

Veri sızıntısı riski **yüksek ve nicel olarak bilinmiyor**. Kesin bölme veya fold
ataması kaydedilmedi; source_group_analysis.csv içindeki final_split unassigned'dır.
Fizibilite için yalnız taslak sınıf/grup sayıları kaydedildi.

Önce orijinal uzun kayıtlar, çekim oturumları ve mümkünse kamera/tesis/kişi kayıtlarıyla
kaynaklar doğrulanmalı; belirsiz çiftler gözden geçirilmeli, aynı kaynağın kesitleri
birlikte tutulmalı. Kaynak çeşitliliği yetersizse yeni bağımsız çekimler gerekir.
Ardından gruplar sabitlenip sınıf dengesi ve hash/klip/grup sızıntı testleri tekrar
çalıştırılmalı. Zayıf video etiketleri olay doğrulaması değildir; R3D-18 için video-bag
MIL ve ayrı yapılandırma gerekir. İlk aşamadaki 47 eksik ve 2 kare sayısı uyuşmayan
YOLO çıktısı, YOLO destekli bir akış seçilecekse ayrıca çözülmelidir.

Eğitim ve YOLO çalıştırılmadı; model ağırlıkları ve eski deney sonuçları kullanılmadı.
Tam video sürekliliği veya kişi kimliği doğrulaması yapılmadı.
'''
    with (output / REPORTS[3]).open('x', encoding='utf-8') as target:
        target.write(text + practical_note)
    print(json.dumps({'original_groups': base['group_count'], 'expanded_groups': expanded['group_count'],
                      'expanded_drowning_groups': expanded['class_bearing_group_counts']['drowning'],
                      'unresolved_pairs': similarity['all_candidate_pair_count'],
                      'methods': feasibility['expanded_precautionary_scenario']['methods']}, indent=2))
    return feasibility


if __name__ == '__main__':
    run()
