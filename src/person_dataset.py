"""Person-clip preparation and CSV review validation; never trains a model."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from src.drowning_demo.analysis import ROOT, decoded_frames, inspect_video, overlay_font, sha256
from src.video_classification.binary_drowning import read_csv
from src.video_review.reviews import atomic_write_json


PILOT = ('data/raw/kelebek/kelebek2.mp4', 'data/raw/bogulma/bogulma1.mp4',
         'data/test_videos/1020(3)-1.mp4')
OUTPUT = ROOT / 'results/person_dataset_pilot'
EDITABLE = {'label', 'review_status', 'split', 'notes'}
FIELDS = ['clip_id', 'source_video', 'track_id', 'start_time', 'end_time', 'bbox_coordinates',
          'label', 'review_status', 'valid_frame_count', 'training_eligible', 'source_sha256',
          'source_group', 'shoot_group', 'split', 'clip_path', 'preview_path', 'technical_valid',
          'invalid_reason', 'sampled_frame_count', 'padded_frame_count', 'crop_box',
          'start_frame', 'end_frame', 'segment_id', 'segment_end_reason', 'mean_detection_confidence', 'notes']


def write_csv(path, rows, fields=FIELDS):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def crop_bounds(boxes, width, height):
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    margin = max(right-left, bottom-top) * 0.15
    return [max(0, math.floor(left-margin)), max(0, math.floor(top-margin)),
            min(width, math.ceil(right+margin)), min(height, math.ceil(bottom+margin))]


def identity_jump(a, b):
    side = max(a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1], 1)
    distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
    areas = [(v[2]-v[0])*(v[3]-v[1]) for v in (a, b)]
    return distance > 0.75*side or max(areas)/max(min(areas), 1) > 2.5


def track_video(source, info, detector, tracker='bytetrack.yaml'):
    active, history, segments, ledger, events = {}, {}, [], [], []

    def finish(track_id, reason, index):
        observations = active.pop(track_id)
        segments.append({'track_id': track_id, 'observations': observations, 'end_reason': reason})
        if reason != 'end_of_video':
            events.append({'type': 'track_break', 'track_id': track_id, 'frame': index, 'reason': reason})

    for index, frame in enumerate(decoded_frames(source)):
        actual = float(frame.pts*frame.time_base)-info['source_pts_origin_sec']
        if index >= info['frame_count'] or abs(actual-info['timestamps_sec'][index]) > 1e-6:
            raise ValueError('Source timing changed during tracking')
        result = detector.track(frame.to_ndarray(format='bgr24'), persist=True, tracker=tracker,
                                classes=[0], conf=0.1, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        ids = [None]*len(boxes) if result.boxes.id is None else result.boxes.id.int().cpu().tolist()
        detections = []
        for detection_index, (track_id, box, confidence) in enumerate(zip(ids, boxes, confidences)):
            reason = ''
            if not all(math.isfinite(v) for v in (*box, confidence)) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid detector output')
            if track_id is None:
                reason = 'untracked_detection'
            elif confidence < 0.25:
                reason = 'low_detection_confidence'
            else:
                region = crop_bounds([box], info['width'], info['height'])
                if any(other != detection_index and intersection(region, b) > 0 for other, b in enumerate(boxes)):
                    reason = 'other_person_in_crop'
                elif track_id in active and identity_jump(active[track_id][-1]['bbox'], box):
                    reason = 'suspected_identity_jump'
                    events.append({'type': reason, 'track_id': track_id, 'frame': index})
            detections.append({'track_id': track_id, 'bbox': box, 'confidence': confidence,
                               'usable': not reason, 'reason': reason})
        usable = {d['track_id']: d for d in detections if d['usable']}
        for track_id in list(active):
            if track_id not in usable:
                reason = next((d['reason'] for d in detections if d['track_id'] == track_id), 'missing_detection')
                finish(track_id, reason, index)
        for track_id, detection in usable.items():
            if track_id not in active:
                if track_id in history:
                    events.append({'type': 'id_reacquired', 'track_id': track_id, 'frame': index})
                for old_id, old in history.items():
                    if old_id != track_id and old_id not in usable and 0 < index-old['frame'] <= info['fps']/2:
                        area = max((old['bbox'][2]-old['bbox'][0])*(old['bbox'][3]-old['bbox'][1]), 1)
                        if intersection(old['bbox'], detection['bbox'])/area > 0.3:
                            events.append({'type': 'possible_id_reassignment', 'track_id': track_id,
                                           'previous_id': old_id, 'frame': index})
                active[track_id] = []
            observation = {'frame': index, 'time': info['timestamps_sec'][index],
                           'bbox': detection['bbox'], 'confidence': detection['confidence']}
            active[track_id].append(observation)
            history[track_id] = observation
        ledger.append({'frame': index, 'time': info['timestamps_sec'][index], 'detections': detections})
        if index % 120 == 0:
            print(f'{source.name}: tracking {index+1}/{info["frame_count"]}', flush=True)
    for track_id in list(active):
        finish(track_id, 'end_of_video', len(ledger))
    if len(ledger) != info['frame_count']:
        raise ValueError('Source frame count changed')
    return segments, ledger, events


def plan_rows(source, info, digest, source_group, shoot_group, segments, ledger, output):
    rows = []
    step = max(1, round(info['fps']/8))
    minimum_frames = 15*step+1
    for segment_id, segment in enumerate(segments, 1):
        points = segment['observations']
        cursor, chunks = 0, []
        while cursor < len(points):
            stop = cursor+1
            while stop < len(points) and points[stop]['time']-points[cursor]['time'] < 2.5-1e-7:
                stop += 1
            chunks.append(points[cursor:stop])
            cursor = stop
        if len(chunks) > 1 and len(chunks[-1]) < minimum_frames:
            combined = chunks[-2]+chunks[-1]
            last = combined[-1]['frame']
            end = info['timestamps_sec'][last+1] if last+1 < info['frame_count'] else info['duration_sec']
            if end-combined[0]['time'] <= 3.000001:
                chunks[-2:] = [combined]
            elif len(combined) >= 2*minimum_frames:
                middle = len(combined)//2
                chunks[-2:] = [combined[:middle], combined[middle:]]
        for chunk, observations in enumerate(chunks):
            first, last = observations[0]['frame'], observations[-1]['frame']
            if [p['frame'] for p in observations] != list(range(first, last+1)):
                raise ValueError('Non-consecutive source frames in person clip')
            end_time = info['timestamps_sec'][last+1] if last+1 < info['frame_count'] else info['duration_sec']
            crop = crop_bounds([p['bbox'] for p in observations], info['width'], info['height'])
            reasons = []
            sample_count = len(range(0, len(observations), step))
            if end_time-observations[0]['time'] > 3.000001 or sample_count < 16:
                reasons.append('insufficient_real_frames_or_duration')
            if any(intersection(crop, d['bbox']) > 0 for p in observations
                   for d in ledger[p['frame']]['detections'] if d['track_id'] != segment['track_id']):
                reasons.append('other_person_in_fixed_crop')
            clip_id = f'P{digest[:10]}_T{segment["track_id"]:03d}_S{segment_id:03d}_C{chunk:02d}'
            rows.append({'clip_id': clip_id, 'source_video': str(source.resolve()),
                         'track_id': segment['track_id'], 'start_time': observations[0]['time'], 'end_time': end_time,
                         'bbox_coordinates': json.dumps(observations, separators=(',', ':')),
                         'label': 'UNCERTAIN', 'review_status': 'unreviewed',
                         'valid_frame_count': len(observations), 'training_eligible': False,
                         'source_sha256': digest, 'source_group': source_group, 'shoot_group': shoot_group,
                         'split': 'unassigned',
                         'clip_path': '' if 'other_person_in_fixed_crop' in reasons else str(output/'clips'/f'{clip_id}.mp4'),
                         'preview_path': str(output/'previews'/f'{clip_id}.jpg'),
                         'technical_valid': not reasons, 'invalid_reason': '|'.join(reasons),
                         'sampled_frame_count': sample_count, 'padded_frame_count': 0,
                         'crop_box': json.dumps(crop), 'start_frame': first, 'end_frame': last,
                         'segment_id': segment_id, 'segment_end_reason': segment['end_reason'],
                         'mean_detection_confidence': float(np.mean([p['confidence'] for p in observations])), 'notes': ''})
    return rows


def person_image(frame, crop):
    image = ImageOps.contain(frame.to_image().crop(crop), (320, 320), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (320, 320), 'black')
    canvas.paste(image, ((320-image.width)//2, (320-image.height)//2))
    return canvas


def export_clip(source, info, row, frames=None):
    first, last = row['start_frame'], row['end_frame']
    selected = sorted({first, (first+last)//2, last})
    preview = Image.new('RGB', (960, 350), '#101923')
    crop = json.loads(row['crop_box'])
    target = Path(row['clip_path']) if row['clip_path'] else None
    partial = target.with_suffix('.partial.mp4') if target else None
    container = av.open(str(partial), 'w', options={'movflags': '+faststart'}) if target else None
    count = 0
    try:
        if container:
            stream = container.add_stream('libx264', rate=Fraction(info['rate_numerator'], info['rate_denominator']))
            stream.width = stream.height = 320
            stream.pix_fmt = 'yuv420p'
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000000)
            stream.options = {'crf': '20', 'preset': 'veryfast', 'bf': '0'}
        frame_iterator = enumerate(decoded_frames(source)) if frames is None else frames
        for index, frame in frame_iterator:
            if index < first:
                continue
            if index > last:
                break
            image = person_image(frame, crop)
            if index in selected:
                col = selected.index(index)
                preview.paste(image, (col*320, 30))
                ImageDraw.Draw(preview).text((col*320+5, 5), f'frame {index}', font=overlay_font(17), fill='white')
            if container:
                out = av.VideoFrame.from_image(image)
                out.pts = round((info['timestamps_sec'][index]-row['start_time'])*1000000)
                out.time_base = Fraction(1, 1000000)
                end = info['timestamps_sec'][index+1] if index+1 < info['frame_count'] else info['duration_sec']
                out.duration = max(1, round((end-info['timestamps_sec'][index])*1000000))
                for packet in stream.encode(out):
                    container.mux(packet)
            count += 1
        if count != row['valid_frame_count']:
            raise ValueError('Person crop frame count mismatch')
        if container:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            decoded = inspect_video(partial)
            expected = np.array(info['timestamps_sec'][first:last+1])-row['start_time']
            if decoded['frame_count'] != count or not np.allclose(decoded['timestamps_sec'], expected, atol=0.0001, rtol=0):
                raise ValueError('Person MP4 frame/timestamp verification failed')
            partial.rename(target)
        preview.save(row['preview_path'], quality=93)
    finally:
        if container:
            container.close()
        if partial:
            partial.unlink(missing_ok=True)


def export_clips(source, info, rows):
    """Share decoded source frames across overlapping people; retain at most a clip span."""
    pending = sorted(rows, key=lambda r: r['end_frame'])
    if not pending:
        return
    needed = {i for r in rows for i in range(r['start_frame'], r['end_frame']+1)}
    cache, current = {}, 0
    for index, frame in enumerate(decoded_frames(source)):
        actual = float(frame.pts*frame.time_base)-info['source_pts_origin_sec']
        if index >= info['frame_count'] or abs(actual-info['timestamps_sec'][index]) > 1e-6:
            raise ValueError('Source changed during clip export')
        if index in needed:
            cache[index] = frame
        while current < len(pending) and pending[current]['end_frame'] == index:
            row = pending[current]
            export_clip(source, info, row, ((i, cache[i]) for i in range(row['start_frame'], index+1)))
            current += 1
            keep_from = min((r['start_frame'] for r in pending[current:]), default=index+1)
            cache = {i: f for i, f in cache.items() if i >= keep_from}
            if current % 20 == 0:
                print(f'{source.name}: exported {current}/{len(pending)} candidates', flush=True)
    if current != len(pending) or index+1 != info['frame_count']:
        raise ValueError('Clip export incomplete')


def validate_rows(rows, originals):
    baseline = {r['clip_id']: r for r in originals}
    if len(rows) != len(baseline) or len({r['clip_id'] for r in rows}) != len(rows) or {r['clip_id'] for r in rows} != set(baseline):
        raise ValueError('Missing, duplicate or unknown clip IDs')
    groups = defaultdict(set)
    validated = []
    for row in rows:
        original = baseline[row['clip_id']]
        for key in FIELDS:
            if key in EDITABLE:
                continue
            value = str(row.get(key, ''))
            expected = original[key]
            try:
                if key in {'start_time', 'end_time', 'mean_detection_confidence'}:
                    equal = math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-6)
                elif key in {'track_id', 'valid_frame_count', 'sampled_frame_count', 'padded_frame_count',
                             'start_frame', 'end_frame', 'segment_id'}:
                    equal = Decimal(value) == Decimal(str(expected))
                elif key in {'training_eligible', 'technical_valid'}:
                    equal = value.strip().lower() == str(expected).lower()
                else:
                    equal = value == str(expected)
            except (ValueError, InvalidOperation):
                equal = False
            if not equal:
                raise ValueError(f'{row["clip_id"]}: protected field changed: {key}')
        label, status, split = (row[k].strip() for k in ('label', 'review_status', 'split'))
        if label not in {'NORMAL', 'DROWNING', 'UNCERTAIN'} or status not in {'unreviewed', 'verified', 'excluded'}:
            raise ValueError(f'{row["clip_id"]}: invalid label or review_status')
        if split not in {'unassigned', 'train', 'validation', 'test'}:
            raise ValueError('Invalid split')
        if split != 'unassigned':
            for key in ('source_sha256', 'source_group', 'shoot_group'):
                groups[(key, original[key])].add(split)
        eligible = original['technical_valid'] and label != 'UNCERTAIN' and status == 'verified'
        validated.append({**row, 'label': label, 'review_status': status, 'split': split, 'training_eligible': eligible})
    if any(len(values) > 1 for values in groups.values()):
        raise ValueError('Source/capture group crosses train/validation/test; unresolved pilot captures must remain together')
    return validated


def validate_dataset(csv_path):
    csv_path = csv_path.resolve()
    manifest = json.loads((csv_path.parent/'manifest.json').read_text(encoding='utf-8'))
    for path, digest in manifest['protected_sha256'].items():
        if not Path(path).is_file() or sha256(Path(path)) != digest:
            raise ValueError(f'Missing or modified source/clip/preview: {path}')
    rows = validate_rows(read_csv(csv_path), manifest['rows'])
    write_csv(csv_path.parent/'validated_clips.csv', rows)
    report = {'status': 'valid', 'clips': len(rows),
              'training_eligible': sum(r['training_eligible'] for r in rows),
              'labels': dict(Counter(r['label'] for r in rows)),
              'source_provenance_complete': False, 'independent_split_ready': False,
              'note': 'Review verifies both behavior and single-person identity continuity. Unknown captures are grouped conservatively.'}
    atomic_write_json(csv_path.parent/'validation_report.json', report)
    return report


def replay_tracking(source, info, log_path, tracker):
    """Rebuild interrupted preparation from recorded detections, checking identical decisions."""
    from types import SimpleNamespace
    import torch

    ledger = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
    if len(ledger) != info['frame_count']:
        raise ValueError('Incomplete cached detection ledger')
    records = iter(ledger)

    class RecordedDetector:
        def track(self, *args, **kwargs):
            detections = next(records)['detections']
            ids = [d['track_id'] for d in detections]
            if any(i is None for i in ids) and not all(i is None for i in ids):
                raise ValueError('Mixed cached tracked/untracked detections')
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=torch.tensor([d['bbox'] for d in detections], dtype=torch.float64).reshape(-1, 4),
                conf=torch.tensor([d['confidence'] for d in detections], dtype=torch.float64),
                id=None if all(i is None for i in ids) else torch.tensor(ids)))]

    segments, reconstructed, events = track_video(source, info, RecordedDetector(), tracker)
    if reconstructed != ledger:
        raise ValueError('Cached detection decisions or source timing changed')
    return segments, ledger, events


def prepare(output=OUTPUT, sources=None, tracker='bytetrack.yaml', metadata=None, resume=False):
    sources = [(ROOT/p).resolve() for p in (PILOT if sources is None else sources)]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    detector_path = ROOT/'yolov8s.pt'
    if not detector_path.is_file():
        raise FileNotFoundError(detector_path)
    output = output.resolve()
    if resume and any((output/name).exists() for name in ('clips.csv', 'manifest.json')):
        raise ValueError('Refusing to resume a finalized or reviewed dataset')
    output.mkdir(parents=True, exist_ok=resume)
    (output/'clips').mkdir(exist_ok=resume)
    (output/'previews').mkdir(exist_ok=resume)
    metadata = read_csv(ROOT/'results/r3d18_binary_drowning/binary_metadata.csv') if metadata is None else metadata
    all_rows, summaries, protected = [], [], {str(detector_path): sha256(detector_path)}
    try:
        for source in sources:
            info, digest = inspect_video(source), sha256(source)
            protected[str(source)] = digest
            matches = [r for r in metadata if r['sha256'] == digest]
            source_group = min((r['relation_group'] for r in matches), default=digest)
            captures = {f'{r["source_id"]}/{r["session_id"]}' for r in matches
                        if r['source_id'] not in {'', 'unknown'} and r['session_id'] not in {'', 'unknown'}}
            shoot_group = min(captures) if captures else 'UNRESOLVED_CAPTURE_PILOT'
            log_path = output/f'{digest[:10]}_detections.jsonl'
            if resume and log_path.is_file():
                segments, ledger, events = replay_tracking(source, info, log_path, tracker)
            else:
                segments, ledger, events = track_video(source, info, YOLO(str(detector_path)), tracker)
                with log_path.open('w', encoding='utf-8') as log:
                    for frame in ledger:
                        log.write(json.dumps(frame)+'\n')
            rows = plan_rows(source, info, digest, source_group, shoot_group, segments, ledger, output)
            pending = []
            for row in rows:
                if resume and row['clip_path'] and Path(row['clip_path']).is_file():
                    saved = inspect_video(Path(row['clip_path']))
                    expected = np.array(info['timestamps_sec'][row['start_frame']:row['end_frame']+1])-row['start_time']
                    if saved['frame_count'] != row['valid_frame_count'] or not np.allclose(saved['timestamps_sec'], expected, atol=0.0001, rtol=0):
                        raise ValueError('Existing clip fails resume verification')
                    if not Path(row['preview_path']).is_file():
                        pending.append({**row, 'clip_path': ''})
                elif not resume or row['clip_path'] or not Path(row['preview_path']).is_file():
                    pending.append(row)
            export_clips(source, info, pending)
            for row in rows:
                for key in ('clip_path', 'preview_path'):
                    if row[key]:
                        protected[row[key]] = sha256(Path(row[key]))
            counts = Counter(d['reason'] for f in ledger for d in f['detections'] if d['reason'])
            ids = {d['track_id'] for f in ledger for d in f['detections'] if d['track_id'] is not None}
            confidences = [d['confidence'] for f in ledger for d in f['detections']]
            summaries.append({'source_video': str(source), 'source_sha256': digest,
                              'source_group': source_group, 'shoot_group': shoot_group,
                              'metadata_person_count': sorted({r['swimmer_count'] for r in matches}),
                              'unique_track_ids': len(ids), 'track_ids': sorted(ids),
                              'maximum_simultaneous_tracks': max(sum(d['track_id'] is not None for d in f['detections']) for f in ledger),
                              'actual_distinct_people': 'unverified; track IDs can fragment or switch',
                              'total_frames': len(ledger),
                              'sampling_frame_step': max(1, round(info['fps']/8)),
                              'minimum_real_source_frames': 15*max(1, round(info['fps']/8))+1,
                              'frames_without_detections': sum(not f['detections'] for f in ledger),
                              'frames_without_usable_person': sum(not any(d['usable'] for d in f['detections']) for f in ledger),
                              'mean_detection_confidence': float(np.mean(confidences)) if confidences else None,
                              'rejected_detections': dict(counts), 'clip_count': len(rows),
                              'technical_valid_clips': sum(r['technical_valid'] for r in rows),
                              'invalid_clips': sum(not r['technical_valid'] for r in rows),
                              'insufficient_clips': sum('insufficient_real_frames' in r['invalid_reason'] for r in rows),
                              'exported_mp4_clips': sum(bool(r['clip_path']) for r in rows),
                              'tracking_breaks': sum(e['type'] == 'track_break' for e in events),
                              'overlap_ambiguity_frames': sum(any(d['reason'] == 'other_person_in_crop' for d in f['detections']) for f in ledger),
                              'id_mix_suspicions': sum(e['type'] in {'suspected_identity_jump', 'possible_id_reassignment'} for e in events)
                                  + sum(any(d['reason'] == 'other_person_in_crop' for d in f['detections']) for f in ledger),
                              'events': events, 'detection_log': str(log_path)})
            all_rows.extend(rows)
            print(json.dumps({k: summaries[-1][k] for k in ('source_video', 'clip_count', 'technical_valid_clips', 'invalid_clips')}), flush=True)
        sheet = Image.new('RGB', (1600, max(1, math.ceil(len(all_rows)/4))*180), '#101923')
        for index, row in enumerate(all_rows):
            with Image.open(row['preview_path']) as preview:
                sheet.paste(preview.resize((400, 146)), ((index%4)*400, (index//4)*180+34))
            title = f'{index+1}: T{row["track_id"]} {row["valid_frame_count"]}f ' + ('VALID' if row['technical_valid'] else 'INVALID')
            ImageDraw.Draw(sheet).text(((index%4)*400+5, (index//4)*180+7), title, font=overlay_font(17), fill='white')
        sheet.save(output/'contact_sheet.jpg', quality=94)
        write_csv(output/'clips.csv', all_rows)
        atomic_write_json(output/'manifest.json', {'version': 1, 'rows': all_rows, 'protected_sha256': protected})
        atomic_write_json(output/'pilot_report.json', {
            'sources': summaries, 'clips': len(all_rows), 'training_eligible': 0,
            'tracker': tracker, 'detector': str(detector_path),
            'sampling_contract': '16 real frames; step=max(1, round(source_fps/8)); no fixed 2-second floor; no padding',
            'labels': 'All UNCERTAIN; no folder-derived labels', 'split': 'All unassigned',
            'limitations': ['Geometry cannot detect every smooth ID swap; human review of identity AND behavior is mandatory.',
                           'YOLO confidence is not person detection accuracy; undetected people may still enter a crop.',
                           'Unknown capture groups are kept together; no independent train/test split is claimed.',
                           'Invalid short clips are previews only; no frame repetition or duration stretching.'],
            'status': 'complete'})
        return output
    except Exception as error:
        atomic_write_json(output/'error.json', {'status': 'failed', 'error': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    preparation = sub.add_parser('prepare', help='Prepare only the three fixed pilot videos; refuses existing output directories')
    preparation.add_argument('--output', type=Path, default=OUTPUT)
    validation = sub.add_parser('validate', help='Validate edited labels and export derived eligibility; never trains')
    validation.add_argument('--csv', type=Path, default=OUTPUT/'clips.csv')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.output), flush=True)
    else:
        print(json.dumps(validate_dataset(args.csv), indent=2), flush=True)


if __name__ == '__main__':
    main()
