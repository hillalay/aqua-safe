"""Fresh, read-only source audit; never imports previous experiment results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.video_classification.binary_drowning import Groups
from src.video_classification.preprocessing import probe_video, build_clip_records
from src.video_review.inventory import sha256_file, visual_hashes, similarity_candidates


ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mpg', '.mpeg', '.wmv'}
LABELS = {'bogulma': 'drowning', 'kelebek': 'normal', 'kurbagalama': 'normal',
          'serbest': 'normal', 'sirtustu': 'normal'}
MISSING_FIELDS = ['video_id', 'filename', 'source_video_path', 'reason']


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value.casefold().replace('ı', 'i'))
                   if not unicodedata.combining(c))


def label_for(path, raw):
    folder = normalized(path.relative_to(raw).parts[0])
    label = LABELS.get(folder, '')
    name = normalized(path.stem)
    filename_drowning = any(token in name for token in ('bogul', 'drown'))
    filename_normal = any(token in name for token in ('kelebek', 'kurbagalama', 'serbest', 'sirtustu'))
    conflict = (label == 'normal' and filename_drowning) or (label == 'drowning' and filename_normal)
    return ('' if conflict else label), conflict


def video_files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)


def snapshot(paths):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def inspect_yolo(path, raw_info):
    try:
        info = probe_video(path)
        if raw_info is None:
            return 'source_unreadable'
        if abs(info.frame_count - raw_info.frame_count) > 1:
            return 'frame_count_mismatch'
        if abs(info.duration_sec - raw_info.duration_sec) > max(0.5, 2 / raw_info.fps):
            return 'duration_mismatch'
        return 'metadata_consistent'
    except (RuntimeError, ValueError, OverflowError) as error:
        return 'unreadable: ' + str(error)


def collect(root):
    raw, yolo = root / 'data/raw', root / 'results/detect_all'
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    paths, outputs = video_files(raw), video_files(yolo)
    if not paths:
        raise ValueError('No raw videos found')
    before_raw, before_yolo = snapshot(paths), snapshot(outputs)
    source_keys = Counter((p.parent, p.stem.casefold()) for p in paths)
    records, matched = [], set()
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        label, conflict = label_for(path, raw)
        digest = sha256_file(path)
        # IDs identify content; aliases and conflicting folder labels remain in the duplicate report.
        info, error, hashes, clip_count = None, '', [], 0
        try:
            info = probe_video(path)
            if not math.isfinite(info.fps):
                raise ValueError('Non-finite FPS')
            hashes = visual_hashes(path)
            if len(hashes) != 5:
                raise ValueError('Not all five sampled frames decoded')
            clip_count = len(build_clip_records(
                {'video_id': digest, 'path': relative, 'style': label}, info))
        except (RuntimeError, ValueError, OverflowError) as exc:
            error = str(exc)
        target = yolo / path.parent.relative_to(raw) / path.stem
        candidates = [p for p in outputs if p.parent == target and p.stem == path.stem]
        matched.update(candidates)
        status = 'missing' if not candidates else 'ambiguous'
        if len(candidates) == 1:
            status = inspect_yolo(candidates[0], info)
        if candidates and source_keys[(path.parent, path.stem.casefold())] > 1:
            status = 'ambiguous_source_mapping'
        records.append({
            'video_id': 'SHA256_' + digest, 'filename': path.name, 'source_video_path': relative,
            'yolo_output_path': '|'.join(p.relative_to(root).as_posix() for p in candidates),
            'yolo_output_exists': bool(candidates), 'yolo_output_status': status,
            'class_label': label, 'label_strength': 'weak_video_label',
            'event_verified': False, 'label_conflict': conflict,
            'folder_label': LABELS.get(normalized(path.relative_to(raw).parts[0]), ''),
            'duration_sec': info.duration_sec if info else '', 'fps': info.fps if info else '',
            'frame_count': info.frame_count if info else '', 'width': info.width if info else '',
            'height': info.height if info else '', 'person_detection_present': 'unknown',
            'person_detection_evidence': 'annotated_video_only_no_machine_readable_boxes',
            'sha256': digest, 'visual_hashes': '|'.join(hashes), 'video_probe_error': error,
            'r3d18_clip_count_16_at_8fps': clip_count,
            'source_id': 'unknown', 'session_id': 'unknown', 'split': 'unassigned',
        })
        print(f'[{index}/{len(paths)}] {relative}: {status}', flush=True)
    after_raw, after_yolo = snapshot(video_files(raw)), snapshot(video_files(yolo))
    changes = {
        'raw': sorted(p for p in before_raw.keys() | after_raw.keys() if before_raw.get(p) != after_raw.get(p)),
        'yolo': sorted(p for p in before_yolo.keys() | after_yolo.keys() if before_yolo.get(p) != after_yolo.get(p)),
    }
    return records, {
        'raw_video_files': len(paths), 'yolo_video_files': len(outputs),
        'unmatched_yolo_paths': [p.relative_to(root).as_posix() for p in outputs if p not in matched],
        'files_changed_during_scan': changes,
    }


def summarize(records, scan):
    by_hash = defaultdict(list)
    for row in records:
        by_hash[row['sha256']].append(row)
    manifest, duplicates = [], []
    for digest, aliases in by_hash.items():
        # Prefer a readable canonical source with a consistent YOLO output.
        row = dict(min(aliases, key=lambda r: (bool(r['video_probe_error']),
                    r['yolo_output_status'] != 'metadata_consistent', r['source_video_path'])))
        labels = {r['class_label'] for r in aliases}
        if len(labels) > 1 or any(r['label_conflict'] for r in aliases):
            row['class_label'], row['label_conflict'] = '', True
        row['alias_count'] = len(aliases)
        if len(aliases) > 1:
            duplicates.append({'sha256': digest, 'canonical_video_id': row['video_id'],
                               'canonical_path': row['source_video_path'],
                               'label_conflict': row['label_conflict'],
                               'members': [{k: r[k] for k in ('source_video_path', 'folder_label',
                                           'class_label', 'yolo_output_path')} for r in aliases]})
        manifest.append(row)
    pairs = similarity_candidates(manifest)
    groups = Groups([r['video_id'] for r in manifest])
    for pair in pairs:
        groups.union(pair['video_id_a'], pair['video_id_b'])
    changed = bool(scan['files_changed_during_scan']['raw'] or scan['files_changed_during_scan']['yolo'])
    for row in manifest:
        reasons = []
        if row['label_conflict'] or not row['class_label']:
            reasons.append('label_conflict_or_unknown')
        if row['video_probe_error']:
            reasons.append('video_probe_failed')
        if not row['r3d18_clip_count_16_at_8fps']:
            reasons.append('no_complete_r3d18_clip')
        if changed:
            reasons.append('input_files_changed_during_scan')
        row.update(relation_group=groups.find(row['video_id']),
                   weak_training_candidate=not reasons,
                   weak_training_with_yolo_candidate=not reasons and row['yolo_output_status'] == 'metadata_consistent',
                   trusted_split_eligible=False, exclusion_reasons='|'.join(reasons))
    candidates = [r for r in manifest if r['weak_training_candidate']]
    with_yolo = [r for r in manifest if r['weak_training_with_yolo_candidate']]
    conflicts = [{k: r[k] for k in ('source_video_path', 'filename', 'folder_label')}
                 for r in records if r['label_conflict']]
    counts = lambda rows: dict(Counter(r['class_label'] or 'unresolved' for r in rows))
    summary = {
        **scan, 'raw_class_counts': counts(records),
        'raw_folder_counts': dict(Counter(Path(r['source_video_path']).parent.name for r in records)),
        'raw_videos_with_yolo': sum(r['yolo_output_exists'] for r in records),
        'raw_yolo_status_counts': dict(Counter(r['yolo_output_status'] for r in records)),
        'unique_content_videos': len(manifest), 'unique_class_counts': counts(manifest),
        'weak_training_candidates': len(candidates), 'weak_training_candidate_class_counts': counts(candidates),
        'weak_training_with_yolo_candidates': len(with_yolo),
        'weak_training_with_yolo_class_counts': counts(with_yolo),
        'trusted_split_eligible_videos': 0, 'confirmed_event_labels': 0,
        'person_detection_status_counts': dict(Counter(r['person_detection_present'] for r in records)),
        'label_conflicts': conflicts,
    }
    duplicate_report = {
        'exact_duplicate_groups': duplicates, 'exact_duplicate_group_count': len(duplicates),
        'redundant_exact_duplicate_files': len(records) - len(manifest),
        'files_in_exact_duplicate_groups': sum(len(g['members']) for g in duplicates),
        'suspicious_similar_pairs': pairs, 'suspicious_similar_pair_count': len(pairs),
        'suspicious_similar_unique_videos': len({p[k] for p in pairs for k in ('video_id_a', 'video_id_b')}),
        'method': 'Full-file SHA256; five sampled frame dHashes; minimum pair distance <=4 with equal dimensions.',
        'limitations': 'Similarity is a review hint, not proof. Resized, cropped or short overlapping clips may be missed; shared pool backgrounds may cause false positives.',
    }
    readiness = {
        'training_started': False, 'ready_for_training': False,
        'weak_training_candidates': len(candidates), 'weak_training_with_yolo_candidates': len(with_yolo),
        'reliable_train_validation_test_split': False,
        'split_policy': 'All splits unassigned. Future clips must inherit the parent content ID and relation_group; verify source/session and group them before splitting.',
        'unknown_source_videos': len(manifest), 'unknown_session_videos': len(manifest),
        'blocking_issues': ['Source/session provenance unknown; independent splits cannot be established.',
                            'All labels are weak folder labels; event intervals and event truth are unverified.'],
        'yolo_limitations': 'Existing scripts save annotated AVI only. Person detection presence is unknown; YOLO is never a drowning label. File existence does not prove a complete output.',
        'validation_scope': 'Raw headers, first frame and five sampled frames; YOLO headers and first frame, frame-count/duration consistency. Full sequential decode was not performed.',
        'previous_experiment_results_used': False,
        'r3d18': {'architecture_reusable': True, 'outputs': 2,
                  'class_to_index': {'normal': 0, 'drowning': 1},
                  'preprocessing': 'Existing 16-frame, 8fps clips, 112x112 RGB normalization can be reused.',
                  'training_strategy': 'Existing video-bag MIL is appropriate for weak labels; do not label every clip as a confirmed drowning event.',
                  'manifest_drop_in_compatible': False,
                  'required_before_training': 'Create a new config and manifest adapter for MIL column/class names and verified group folds; use fresh experiment outputs and initialization, not prior fine-tuned checkpoints.'},
    }
    if changed:
        readiness['blocking_issues'].append('Files changed during scan; rerun into a new directory after writers finish.')
    if conflicts or any(g['label_conflict'] for g in duplicates):
        readiness['blocking_issues'].append('Resolve folder/filename or cross-copy label conflicts manually.')
    return manifest, summary, duplicate_report, readiness


def write_csv(path, rows, fields):
    with path.open('x', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output) if output is not None else root / 'results/aquasafe_binary_v1'
    # Refuse existing destinations, including old experiments, before scanning any data.
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    records, scan = collect(root)
    manifest, summary, duplicates, readiness = summarize(records, scan)
    summary.update(scan_started_utc=started, scan_finished_utc=datetime.now(timezone.utc).isoformat())
    write_csv(output / 'dataset_manifest.csv', manifest, list(manifest[0]))
    missing = [{k: r[k] for k in MISSING_FIELDS[:-1]} | {'reason': 'no_matching_yolo_video'}
               for r in records if not r['yolo_output_exists']]
    write_csv(output / 'missing_yolo_outputs.csv', missing, MISSING_FIELDS)
    for name, content in [('dataset_summary.json', summary), ('duplicate_report.json', duplicates),
                          ('data_readiness_report.json', readiness)]:
        with (output / name).open('x', encoding='utf-8') as target:
            json.dump(content, target, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    prepare(output=args.output)
