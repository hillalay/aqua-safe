"""Experimental tracked-person crops with the existing full-scene R3D checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
from bisect import bisect_right
from fractions import Fraction
import json
import math
from pathlib import Path

import av
from PIL import ImageDraw
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from src.drowning_demo.analysis import decoded_frames, overlay_font
from src.drowning_demo.detection import make_tracker, attach_tracks, detection_gaps
from src.drowning_demo.swimmers import crop_person, score_clip
from src.video_review.inventory import sha256_file
from .model import build_r3d18
from .preprocessing import probe_video
from .visualize_yolo_r3d18 import verify_render
from .yolo_dataset import YoloClipDataset, annotated_path


ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    'The checkpoint was trained on annotated full-scene clips, not person crops. Crop scores are experimental; person-level accuracy/recall and calibration are unknown.',
    'Parent-video labels are weak labels, not verified person/event truth. No person-level recall is claimed.',
    'Tracker IDs are temporary tracking IDs, not verified human identities; switches and missed detections remain possible.',
    'Detections are rerun on annotated AVI inputs; old YOLO drawings may remain visible. No source file is changed.',
    'A score requires the same confirmed track ID in all 16 real sampled frames. Detection gaps between sampled frames are accepted by user choice; no box interpolation, sample padding, or score extrapolation is used.',
    'Existing crop_person preprocessing uses 1.3x square context and the existing R3D normalization. Nearby people may enter that context.',
]


def person_clip_plan(clips, frames):
    plans, skipped = [], []
    for clip in clips:
        start, end, step = (int(clip[k]) for k in ('start_frame', 'end_frame', 'frame_step'))
        if end >= len(frames):
            raise ValueError('Detection frames do not cover the parent clip')
        indices = list(range(start, end + 1, step))
        if len(indices) != 16:
            raise ValueError('Expected exactly 16 real sampled frames')
        visible = [{p['track_id'] for p in frames[index] if p['track_id'] is not None}
                   for index in indices]
        candidates = set.intersection(*visible)
        if not candidates:
            skipped.append({'clip_id': clip['clip_id'], 'reason': 'no_confirmed_track_in_all_16_sampled_frames'})
        for track in sorted(candidates):
            plans.append({'person_clip_id': f"{clip['clip_id']}_T{track}",
                          'parent_clip_id': clip['clip_id'], 'video_id': clip['video_id'],
                          'source_path': clip['source_path'], 'track_id': track,
                          'start_frame': start, 'end_frame': end, 'frame_step': step,
                          'start_time': float(clip['start_time']), 'end_time': float(clip['end_time']),
                          'parent_video_label': int(clip['video_label']),
                          'label_strength': 'weak_video_label', 'person_event_label': 'unknown',
                          'draft_split': clip['draft_split'], 'leakage_group': clip['leakage_group'],
                          'sample_frame_indices': indices})
    return plans, skipped


def score_people(source, frames, plans, model, device='cuda'):
    if not plans:
        return []
    scheduled, samples, result = defaultdict(list), defaultdict(list), []
    for plan in plans:
        for index in plan['sample_frame_indices']:
            scheduled[index].append(plan)
    for index, frame in enumerate(tqdm(decoded_frames(source), total=len(frames),
                                       desc='Person crops ' + source.stem, unit='frame', mininterval=1)):
        if index not in scheduled:
            continue
        rgb = frame.to_ndarray(format='rgb24')
        by_track = {p['track_id']: p for p in frames[index] if p['track_id'] is not None}
        for plan in scheduled[index]:
            key = plan['person_clip_id']
            samples[key].append(crop_person(rgb, by_track[plan['track_id']]['box']))
            if index == plan['end_frame']:
                probability, padded = score_clip(model, samples.pop(key), device)
                if probability is None or padded or not math.isfinite(probability):
                    raise ValueError('A person score requires 16 real samples, no padding')
                result.append({**{k: v for k, v in plan.items() if k != 'sample_frame_indices'},
                               'normal_probability': 1 - probability, 'drowning_probability': probability,
                               'sample_count': 16, 'padded_frames': 0})
    if samples or len(result) != len(plans):
        raise ValueError('Not every planned person clip was scored exactly once')
    return result


def person_text(person, prediction, threshold):
    identity = f"#{person['track_id']}" if person['track_id'] is not None else '#?'
    confidence = person.get('confidence')
    detection = f"YOLO: %{confidence * 100:.1f}" if confidence is not None else 'YOLO: bilinmiyor'
    if prediction is None:
        return f'{identity} ANALİZ BEKLENİYOR | {detection}', '#FFC857'
    probability = float(prediction['drowning_probability'])
    high = probability >= threshold
    label = 'BOĞULMA UYARISI (deneysel)' if high else 'NORMAL YÜZÜCÜ (model)'
    return (f'{identity} {label}\nR3D boğulma skoru: %{probability * 100:.1f} | {detection}',
            '#FF4444' if high else '#30EF90')


def completed_prediction(rows, ends, index, person):
    position = bisect_right(ends, index) - 1
    if position < 0:
        return None
    row = rows[position]
    if (index > row['display_end_frame']
            or person.get('analysis_segment') != row['analysis_segment']):
        return None
    return row


def render_people(root, clips, frames, predictions, destination, threshold):
    if destination.exists():
        return verify_render(root, clips, destination)
    source, _ = annotated_path(root, clips[0]['source_path'])
    info = probe_video(source)
    tracks = defaultdict(list)
    for prediction in predictions:
        tracks[prediction['track_id']].append(prediction)
    lookups = {}
    for track, rows in tracks.items():
        rows.sort(key=lambda r: r['end_frame'])
        lookups[track] = (rows, [r['end_frame'] for r in rows])
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.mp4')
    if partial.exists():
        raise FileExistsError('Incomplete render exists: ' + str(partial))
    rate = Fraction(str(info.fps)).limit_denominator(100000)
    count = 0
    with av.open(str(partial), 'w') as container:
        stream = container.add_stream('libx264', rate=rate)
        stream.width, stream.height, stream.pix_fmt = info.width, info.height, 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'fast'}
        font = overlay_font(max(16, round(info.width / 85)))
        for index, frame in enumerate(tqdm(decoded_frames(source), total=info.frame_count,
                                          desc='Person MP4 ' + source.stem, unit='frame', mininterval=1)):
            image = frame.to_image()
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, info.width, 42), fill='#101923')
            draw.text((8, 4), f'Deneysel kişi modeli | zaman {index / info.fps:.3f}s | eşik {threshold:.2f}',
                      font=font, fill='white')
            for person in frames[index]:
                rows, starts = lookups.get(person['track_id'], ([], []))
                prediction = completed_prediction(rows, starts, index, person)
                text, color = person_text(person, prediction, threshold)
                x1, y1, x2, y2 = person['box']
                draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
                bounds = draw.multiline_textbbox((0, 0), text, font=font)
                width, height = bounds[2] - bounds[0] + 10, bounds[3] - bounds[1] + 14
                left = max(0, min(x1, info.width - width))
                top = max(44, y1 - height)
                draw.rectangle((left, top, left + width, top + height), fill='#101923')
                draw.multiline_text((left + 4, top + 1), text, font=font, fill=color)
            footer = 'Kişi kırpması bu checkpoint için doğrulanmadı; skor kalibre edilmiş gerçek risk değildir.'
            draw.rectangle((0, info.height - 44, info.width, info.height), fill='#101923')
            draw.text((8, info.height - 40), footer, font=font, fill='#FFC857')
            annotated = av.VideoFrame.from_image(image)
            annotated.pts, annotated.time_base = index, 1 / rate
            for packet in stream.encode(annotated):
                container.mux(packet)
            count += 1
        if count != len(frames):
            raise ValueError('Source changed while rendering')
        for packet in stream.encode():
            container.mux(packet)
    verified = verify_render(root, clips, partial)
    partial.rename(destination)
    verified['output'] = str(destination)
    return verified


def overlap_fraction(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / area if area > 0 else 0


def rolling_person_plan(clips, frames, fps):
    """Keep IDs for display, but reset analysis on gaps or observable ambiguity."""
    # shortcut: geometry cannot detect every ID swap or bad image; validate with person/quality labels before accuracy claims.
    step = max(1, round(fps / 8))
    active, plans, rejected = {}, [], defaultdict(int)
    segment = 0
    for index, people in enumerate(frames):
        current = {}
        for person in people:
            person['analysis_segment'] = None
            track, box = person['track_id'], person['box']
            if track is None:
                rejected['unconfirmed_id'] += 1
                continue
            if (person['confidence'] < 0.15 or box[2] - box[0] < 16 or box[3] - box[1] < 16
                    or not all(math.isfinite(x) for x in box)):
                rejected['insufficient_image'] += 1
                continue
            # Context crops are 1.3x squares: reject neighbours inside that context too.
            side = max(box[2] - box[0], box[3] - box[1]) * 1.3
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            context = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
            if any(other is not person and overlap_fraction(context, other['box']) > 0.25 for other in people):
                rejected['ambiguous_neighbour'] += 1
                continue
            previous = active.get(track)
            if previous and overlap_fraction(previous['box'], box) < 0.1:
                rejected['abrupt_box_jump'] += 1
                previous = None
            if previous is None:
                segment += 1
                previous = {'segment': segment, 'indices': [], 'first': index, 'count': 0}
            state = previous
            state['box'] = box
            person['analysis_segment'] = state['segment']
            current[track] = state
            if (index - state['first']) % step:
                continue
            state['indices'] = (state['indices'] + [index])[-16:]
            state['count'] += 1
            if len(state['indices']) != 16 or (state['count'] - 16) % 8:
                continue
            start = state['indices'][0]
            parent = next((c for c in clips if int(c['start_frame']) <= index <= int(c['end_frame'])), clips[0])
            parent_id = parent['clip_id'] if int(parent['start_frame']) <= index <= int(parent['end_frame']) else None
            plans.append({
                'person_clip_id': f"{parent['video_id']}_S{state['segment']}_F{index}",
                'parent_clip_id': parent_id, 'video_id': parent['video_id'],
                'source_path': parent['source_path'], 'track_id': track,
                'analysis_segment': state['segment'], 'start_frame': start, 'end_frame': index,
                'display_end_frame': index + 8 * step - 1, 'frame_step': step,
                'start_time': start / fps, 'end_time': index / fps,
                'sample_frame_indices': list(state['indices']),
                'parent_video_label': int(parent['video_label']), 'person_event_label': 'unknown',
                'label_strength': 'weak_video_label', 'draft_split': parent['draft_split'],
                'leakage_group': parent['leakage_group'],
            })
        active = current
    return plans, dict(rejected)


def tracking_metrics(frames):
    stats = detection_gaps(frames)
    occurrences = defaultdict(list)
    for index, people in enumerate(frames):
        for person in people:
            if person['track_id'] is not None:
                occurrences[person['track_id']].append(index)
    runs, gaps = [], []
    for indices in occurrences.values():
        length = 1
        for previous, index in zip(indices, indices[1:]):
            if index == previous + 1:
                length += 1
            else:
                runs.append(length)
                gaps.append(index - previous - 1)
                length = 1
        runs.append(length)
    stats.update({
        'frames_with_boxes_ratio': stats['frames_with_boxes'] / len(frames) if frames else 0,
        'detected_boxes': sum(map(len, frames)),
        'confirmed_track_ids': len(occurrences), 'continuous_track_runs': len(runs),
        'same_id_reappearance_gaps': gaps, 'same_id_breaks': len(gaps),
        'continuous_run_lengths': runs,
        'id_switches': 'unknown_without_person_identity_ground_truth',
        'person_box_recall': 'unknown_without_person_box_ground_truth',
    })
    return stats


def detect_comparison(source, info, detector, device, confidence, imgsz):
    trackers = {name: make_tracker(algorithm=name)
                for name in ('bytetrack', 'botsort')}
    frames = {name: [] for name in trackers}
    for frame in tqdm(decoded_frames(source), total=info.frame_count, desc='YOLO + paired trackers'):
        image = frame.to_ndarray(format='bgr24')
        boxes = detector.predict(image, classes=[0], conf=confidence, imgsz=imgsz,
                                 device=device, verbose=False)[0].boxes
        for name, tracker in trackers.items():
            people = attach_tracks(boxes, tracker, image)
            ids = [p['track_id'] for p in people if p['track_id'] is not None]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate track ID in a frame')
            frames[name].append(people)
    if any(len(value) != info.frame_count for value in frames.values()):
        raise ValueError('Detection frame count mismatch')
    return frames


def baseline_report(root, clips):
    base = root / 'results/aquasafe_yolo_r3d18_v1/visualization'
    key = Path(clips[0]['source_path']).stem + '_' + clips[0]['video_id'][-12:] + '.json'
    paths = [base / name / 'detections' / key for name in ('person_level_t030_sampled16', 'person_level_t030')]
    cache = next((path for path in paths if path.exists()), None)
    if cache is None:
        return None
    stored = json.loads(cache.read_text(encoding='utf-8'))
    if stored['source_sha256'] != clips[0]['yolo_sha256']:
        raise ValueError('Baseline source hash mismatch')
    frames = stored['frames']
    plans, _ = person_clip_plan(clips, frames)
    all_detected = all_confirmed = 0
    for clip in clips:
        indices = range(int(clip['start_frame']), int(clip['end_frame']) + 1, int(clip['frame_step']))
        all_detected += all(frames[i] for i in indices)
        all_confirmed += all(any(p['track_id'] is not None for p in frames[i]) for i in indices)
    return {**tracking_metrics(frames), 'parent_clips': len(clips),
            'parents_all_16_detected': all_detected, 'parents_all_16_confirmed': all_confirmed,
            'parents_same_id_16': len({p['parent_clip_id'] for p in plans}),
            'valid_person_clips': len(plans)}


def run(threshold=0.30, root=ROOT, *, output=None, video_id=None, all_videos=False):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be strictly between zero and one')
    root = Path(root).resolve()
    base = root / 'results/aquasafe_yolo_r3d18_v1'
    output = Path(output).resolve() if output else root / 'results/person_tracking_comparison'
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = base / 'training/best_model.pt'
    weight = root / 'yolov8s.pt'
    confidence, imgsz = 0.10, 960
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = YoloClipDataset(root, base / 'clip_manifest.csv', split='validation')
    grouped = defaultdict(list)
    for row in dataset.rows:
        grouped[row['video_id']].append(row)
    baseline = {key: baseline_report(root, clips) for key, clips in grouped.items()}
    if video_id:
        grouped = {key: clips for key, clips in grouped.items() if key == video_id}
        if not grouped:
            raise ValueError('Unknown validation video ID')
    elif not all_videos:
        grouped = dict(sorted(grouped.items(), key=lambda item: int(item[1][0]['source_frame_count']))[:1])
    protected = [checkpoint, weight, base / 'clip_manifest.csv', base / 'training/history.json']
    protected += [p for p in (root / 'results').glob('aquasafe_yolo_r3d18_v*/training/*')
                  if p.is_file() and p.suffix in {'.pt', '.json', '.csv'} and p not in protected]
    protected += [root / clips[0]['source_path'] for clips in grouped.values()]
    hashes = {str(p): sha256_file(p) for p in protected}
    limitations = LIMITATIONS[:4] + [LIMITATIONS[-1],
        'Both trackers receive identical detections on every frame. BoT-SORT uses installed defaults: sparseOptFlow GMC, ReID disabled.',
        'Analysis resets after any missing/invalid/ambiguous frame. 16 real samples at approximately 8 Hz, rolling stride 8 samples. Scores appear only after completion and expire after 8 sample intervals or a segment change.',
        'Geometry/confidence/overlap checks reject observable ambiguity; undetected identity switches and bad images remain possible. IDs are not verified identities.',
        'Box-frame ratio is frames with at least one detection divided by all frames; it is not person recall. Gaps and run lengths are tracking proxies, not IDF1/HOTA.',
        'New rolling windows overlap; report their count separately from coverage of the original 209 parent clips.',
        'Detector changed from yolov8n/640/conf=0.15 to configured settings; detection changes cannot be attributed to tracker choice.',
    ]
    config = {'threshold': threshold, 'detector': str(weight), 'confidence': confidence,
              'imgsz': imgsz, 'device': device, 'input_sha256': hashes,
              'trackers': ['bytetrack', 'botsort'], 'limitations': limitations}
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    detector = YOLO(str(weight))
    model = build_r3d18(2, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    report = {'baseline': baseline, 'videos': [], 'limitations': limitations, 'threshold': threshold,
              'training_started': False, 'complete': False}
    for key, clips in grouped.items():
        source, _ = annotated_path(root, clips[0]['source_path'])
        if sha256_file(source) != clips[0]['yolo_sha256']:
            raise ValueError('Source hash mismatch')
        info = probe_video(source)
        paired = detect_comparison(source, info, detector, device, confidence, imgsz)
        summary = {'video_id': key, 'source_path': clips[0]['source_path'], 'trackers': {}}
        for name, frames in paired.items():
            destination = output / key / name
            destination.mkdir(parents=True)
            fixed, _ = person_clip_plan(clips, frames)
            plans, rejected = rolling_person_plan(clips, frames, info.fps)
            predictions = score_people(source, frames, plans, model, device=device)
            (destination / 'detections.json').write_text(json.dumps(frames), encoding='utf-8')
            (destination / 'scores.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
            exported = render_people(root, clips, frames, predictions, destination / 'overlay.mp4', threshold)
            summary['trackers'][name] = {**tracking_metrics(frames), **exported,
                'original_grid_same_id_candidates': len(fixed),
                'original_grid_parents_with_same_id_candidate': len({p['parent_clip_id'] for p in fixed}),
                'valid_person_classification_clips': len(predictions),
                'parents_with_completed_rolling_score': len({p['parent_clip_id'] for p in predictions
                                                           if p['parent_clip_id'] is not None}),
                'rejected_detection_frames': rejected}
        report['videos'].append(summary)
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if hashes != {str(p): sha256_file(p) for p in protected}:
        raise RuntimeError('Protected inputs changed during analysis')
    report.update(complete=True, protected_inputs_unchanged=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'videos': len(report['videos'])}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--output', type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--video-id')
    selection.add_argument('--all-videos', action='store_true')
    args = parser.parse_args()
    run(args.threshold, output=args.output, video_id=args.video_id, all_videos=args.all_videos)
