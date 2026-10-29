"""Inventory new videos, compare existing trackers, then prepare only new source content."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from PIL import ImageDraw
from ultralytics import YOLO

from src.drowning_demo.analysis import ROOT, decoded_frames, inspect_video, overlay_font, sha256
from src.person_dataset import PILOT, plan_rows, prepare, track_video, validate_dataset, write_csv
from src.video_classification.binary_drowning import Groups, read_csv
from src.video_classification.preprocessing import probe_video
from src.video_review.inventory import hamming, visual_hashes
from src.video_review.reviews import atomic_write_json


OUTPUT = ROOT/'results/person_dataset_new_videos'
EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}


def capture_token(path):
    start = re.match(r'^(\d{4})(?=\s|\()', path.stem)
    date = re.search(r'_(\d{8})(?:\D|$)', path.stem)
    return 'prefix:'+start[1] if start else 'date:'+date[1] if date else ''


def source_category(path, matches):
    if path.parent.name == 'bogulma':
        return 'DROWNING'
    if path.parent.name in {'kelebek', 'kurbagalama', 'serbest', 'sirtustu'}:
        return 'NORMAL'
    candidates = {r['candidate_label'] for r in matches}
    if candidates == {'drowning_candidate'}:
        return 'DROWNING'
    return 'NORMAL' if candidates and candidates <= {'butterfly', 'breaststroke', 'freestyle', 'backstroke'} else 'UNKNOWN'


def inventory(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = read_csv(ROOT/'results/video_review/master_metadata.csv')
    binary = read_csv(ROOT/'results/r3d18_binary_drowning/binary_metadata.csv')
    old_hashes = {r['sha256'] for r in previous}
    known_paths = {str((ROOT/r['path']).resolve()).lower() for r in previous}
    known_paths.update(str((ROOT/r['path']).resolve()).lower() for r in read_csv(ROOT/'results/single_metadata.csv'))
    known_paths.update(str((ROOT/p).resolve()).lower() for p in PILOT)
    by_hash = defaultdict(list)
    for row in previous:
        by_hash[row['sha256']].append(row)
    files = sorted(p.resolve() for p in (ROOT/'data').rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    rows, seen_new = [], set()
    for path in files:
        digest = sha256(path)
        known = str(path).lower() in known_paths
        status = 'previous' if known else 'existing_content_alias' if digest in old_hashes else 'new_content'
        if status == 'new_content':
            if digest in seen_new:
                status = 'new_content_duplicate'
            seen_new.add(digest)
        row = {'source_video': str(path), 'relative_path': str(path.relative_to(ROOT)), 'sha256': digest,
               'status': status, 'source_category_hint': source_category(path, by_hash[digest]),
               'readable': False, 'full_decode_verified': False, 'decode_error': '',
               'fps': '', 'frame_count': '', 'duration_sec': '', 'width': '', 'height': '',
               'visual_hashes': '', 'source_group': '', 'shoot_group': 'UNRESOLVED_CAPTURE_NEW',
               'source_id': 'unknown', 'session_id': 'unknown', 'canonical_source': '',
               'is_exact_duplicate': False, 'capture_token': capture_token(path)}
        try:
            info = probe_video(path)
            row.update(info.__dict__, readable=True)
            if status == 'new_content':
                full = inspect_video(path)
                row.update(full_decode_verified=True, duration_sec=full['duration_sec'])
                row['visual_hashes'] = '|'.join(visual_hashes(path))
            elif by_hash[digest]:
                row['visual_hashes'] = by_hash[digest][0]['visual_hashes']
        except Exception as error:
            row['readable'], row['decode_error'] = False, str(error)
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f'Inventory: {len(rows)}/{len(files)} files', flush=True)
    hashes = {r['sha256'] for r in rows}
    groups = Groups(sorted(hashes))
    related = defaultdict(list)
    for r in binary:
        if r['sha256'] in hashes:
            related[r['relation_group']].append(r['sha256'])
    for values in related.values():
        for value in values[1:]:
            groups.union(values[0], value)
    candidates = []
    tokens = defaultdict(list)
    for r in rows:
        if r['capture_token']:
            tokens[r['capture_token']].append(r)
    for token, values in tokens.items():
        representatives = {r['sha256']: r for r in values}
        for other in list(representatives.values())[1:]:
            first = next(iter(representatives.values()))
            groups.union(first['sha256'], other['sha256'])
            candidates.append({'source_a': first['source_video'], 'source_b': other['source_video'],
                               'evidence': 'filename_capture_candidate', 'detail': token, 'confirmed': False})
    representatives = {r['sha256']: r for r in rows}
    checked = set()
    for new in [r for r in rows if r['status'] == 'new_content' and r['readable']]:
        for other in representatives.values():
            pair = tuple(sorted((new['sha256'], other['sha256'])))
            if len(set(pair)) < 2 or pair in checked or not new['visual_hashes'] or not other['visual_hashes']:
                continue
            checked.add(pair)
            if (new['width'], new['height']) != (other['width'], other['height']):
                continue
            distance = min(hamming(a, b) for a in new['visual_hashes'].split('|') for b in other['visual_hashes'].split('|'))
            if distance <= 4:
                groups.union(new['sha256'], other['sha256'])
                candidates.append({'source_a': new['source_video'], 'source_b': other['source_video'],
                                   'evidence': 'visual_source_candidate', 'detail': str(distance), 'confirmed': False})
    exact = defaultdict(list)
    for r in rows:
        exact[r['sha256']].append(r)
    for values in exact.values():
        canonical = min(values, key=lambda r: (r['status'] != 'previous', '/raw/' not in Path(r['source_video']).as_posix(), r['source_video']))
        for r in values:
            r['canonical_source'] = canonical['source_video']
            r['is_exact_duplicate'] = r is not canonical
            r['source_group'] = 'SRC_'+groups.find(r['sha256'])[:16]
    write_csv(output/'inventory.csv', rows, list(rows[0]))
    write_csv(output/'source_candidates.csv', candidates, ['source_a', 'source_b', 'evidence', 'detail', 'confirmed'])
    raw = [r for r in rows if Path(r['source_video']).is_relative_to(ROOT/'data/raw')]
    new = [r for r in rows if r['status'] in {'new_content', 'new_content_duplicate'}]
    raw_unique = {r['sha256']: r for r in raw}
    actual = {r['source_video'].lower(): r for r in rows}
    eligible = [r for r in binary if r['initial_training_eligible'].lower() == 'true']
    report = {'previous_source_inventory_count': len(previous), 'all_video_files_under_data': len(rows),
              'previous_MIL_training_eligible_videos': len(eligible),
              'previous_MIL_training_source_labels': dict(Counter(r['binary_label'] for r in eligible)),
              'previous_present_file_count_including_aliases': len(rows)-len(new),
              'current_raw_source_files': len(raw), 'new_video_files': len(new),
              'new_unique_content': len({r['sha256'] for r in new}),
              'source_category_hints_raw': dict(Counter(r['source_category_hint'] for r in raw)),
              'new_source_category_hints': dict(Counter(r['source_category_hint'] for r in new)),
              'all_unique_content': len(exact), 'exact_duplicate_groups': sum(len(v)>1 for v in exact.values()),
              'raw_unique_content': len(raw_unique),
              'raw_unique_category_hints': dict(Counter(r['source_category_hint'] for r in raw_unique.values())),
              'conflicting_source_hint_hashes': [digest for digest, values in exact.items()
                                                if len({r['source_category_hint'] for r in values}) > 1],
              'duplicate_file_copies': sum(len(v)-1 for v in exact.values()),
              'same_capture_candidates': len(candidates), 'same_capture_candidates_confirmed': False,
              'unknown_capture_source_files': len(rows),
              'unreadable_files': [r['source_video'] for r in rows if not r['readable']],
              'missing_previous_files': [r['path'] for r in previous if not (ROOT/r['path']).is_file()],
              'modified_previous_sources': [r['path'] for r in previous
                  if str((ROOT/r['path']).resolve()).lower() in actual
                  and actual[str((ROOT/r['path']).resolve()).lower()]['sha256'] != r['sha256']],
              'category_note': 'Folder/inventory source hints only; person clips remain UNCERTAIN.',
              'grouping_note': 'SHA copies, previous relation groups, filename and dHash candidates grouped conservatively; pool backgrounds can cause false visual matches.'}
    atomic_write_json(output/'inventory_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def compare(output):
    destination = output/'tracker_comparison'
    destination.mkdir(parents=True, exist_ok=False)
    tracker_names = ('bytetrack.yaml', 'botsort.yaml')
    anchors = ([0, 90, 135, 254, 385, 473], [0, 60, 120, 180, 220, 254], [0, 72, 104, 113, 125, 165, 179])
    results = []
    for relative, chosen in zip(PILOT, anchors):
        source = ROOT/relative
        info = inspect_video(source)
        ledgers, stats = [], []
        for tracker in tracker_names:
            segments, ledger, events = track_video(source, info, YOLO(str(ROOT/'yolov8s.pt')), tracker)
            rows = plan_rows(source, info, sha256(source), 'comparison', 'unresolved', segments, ledger, destination)
            ids = {d['track_id'] for f in ledger for d in f['detections'] if d['track_id'] is not None}
            record = {'tracker': tracker, 'frame_count': len(ledger), 'unique_ids': len(ids),
                      'frames_with_usable_person': sum(any(d['usable'] for d in f['detections']) for f in ledger),
                      'breaks': sum(e['type']=='track_break' for e in events),
                      'identity_jump_flags': sum(e['type']=='suspected_identity_jump' for e in events),
                      'reassignment_flags': sum(e['type']=='possible_id_reassignment' for e in events),
                      'overlap_frames': sum(any(d['reason']=='other_person_in_crop' for d in f['detections']) for f in ledger),
                      'technical_valid_clips': sum(r['technical_valid'] for r in rows),
                      'anchor_ids': {str(i): [d['track_id'] for d in ledger[i]['detections'] if d['usable']] for i in chosen}}
            stats.append(record)
            ledgers.append(ledger)
            atomic_write_json(destination/f'{source.stem}_{tracker[:-5]}_tracks.json', {'ledger': ledger, 'events': events})
        from PIL import Image
        sheet = Image.new('RGB', (1280, len(chosen)*400), '#101923')
        for i, frame in enumerate(decoded_frames(source)):
            if i not in chosen:
                continue
            row = chosen.index(i)
            for col, (tracker, ledger) in enumerate(zip(tracker_names, ledgers)):
                image = frame.to_image().resize((640, 360))
                draw = ImageDraw.Draw(image)
                for detection in ledger[i]['detections']:
                    box = [detection['bbox'][0]*640/info['width'], detection['bbox'][1]*360/info['height'],
                           detection['bbox'][2]*640/info['width'], detection['bbox'][3]*360/info['height']]
                    color = '#30ef90' if detection['usable'] else '#ffc857'
                    draw.rectangle(box, outline=color, width=2)
                    draw.text((max(0, min(box[0], 530)), max(0, box[1]-18)),
                              f'ID {detection["track_id"]} {detection["confidence"]:.2f}', font=overlay_font(15), fill=color)
                sheet.paste(image, (col*640, row*400+40))
                ImageDraw.Draw(sheet).text((col*640+8, row*400+10), f'{tracker} | frame {i}', font=overlay_font(19), fill='white')
        sheet_path = destination/f'{source.stem}_comparison.jpg'
        sheet.save(sheet_path, quality=95)
        results.append({'source': str(source), 'settings': stats, 'contact_sheet': str(sheet_path)})
    atomic_write_json(destination/'comparison_report.json', {
        'detector': 'local yolov8s.pt; conf=0.1; usable confidence>=0.25',
        'trackers': 'Installed defaults; BoT-SORT sparseOptFlow, with_reid=False; no new weights.',
        'videos': results, 'metrics_are_ground_truth_accuracy': False,
        'selection': {'status': 'pending_visual_review', 'tracker': None},
        'limitations': 'Flags and fragmentation are proxies. Review actual people and identity consistency in source frames before selecting.'})
    print(destination/'comparison_report.json', flush=True)


def prepare_new(output, resume=False):
    report = json.loads((output/'tracker_comparison/comparison_report.json').read_text(encoding='utf-8'))
    if report['selection']['status'] != 'reviewed' or report['selection']['tracker'] not in {'bytetrack.yaml', 'botsort.yaml'}:
        raise ValueError('Tracker comparison needs recorded visual review before processing new videos')
    inventory_rows = read_csv(output/'inventory.csv')
    selected = [r for r in inventory_rows if r['status']=='new_content' and r['readable']=='True' and r['full_decode_verified']=='True']
    if not selected:
        raise ValueError('No readable new unique source content')
    for r in selected:
        if sha256(Path(r['source_video'])) != r['sha256']:
            raise ValueError('New video changed after inventory')
    metadata = [{'sha256': r['sha256'], 'relation_group': r['source_group'], 'source_id': 'unknown',
                 'session_id': 'unknown', 'swimmer_count': 'unknown'} for r in inventory_rows]
    dataset = prepare(output/'dataset', [Path(r['source_video']) for r in selected], report['selection']['tracker'], metadata, resume=resume)
    validation = validate_dataset(dataset/'clips.csv')
    pilot = json.loads((dataset/'pilot_report.json').read_text(encoding='utf-8'))
    rows = read_csv(dataset/'clips.csv')
    pilot['new_source_count'] = len(selected)
    pilot['label_review_ready_clips'] = sum(r['technical_valid']=='True' for r in rows)
    pilot['invalidity_reasons'] = dict(Counter(reason for r in rows for reason in r['invalid_reason'].split('|') if reason))
    pilot['unknown_source_or_capture_clips'] = len(rows)
    pilot['validation'] = validation
    pilot['tracker_selection'] = report['selection']
    atomic_write_json(dataset/'pilot_report.json', pilot)
    print(dataset/'clips.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'compare', 'prepare-new'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true', help='Resume interrupted clip preparation; refuses finalized datasets')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'inventory':
        inventory(output)
    elif args.command == 'compare':
        compare(output)
    else:
        prepare_new(output, resume=args.resume)


if __name__ == '__main__':
    main()
