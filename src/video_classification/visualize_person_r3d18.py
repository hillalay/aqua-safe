"""Experimental tracked-person crops with the existing full-scene R3D checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from src.drowning_demo.detection import make_tracker, attach_tracks
from src.drowning_demo.swimmers import crop_person, score_clip
from src.video_review.inventory import sha256_file
from .model import build_r3d18
from .preprocessing import probe_video
from .prepare_binary_v1 import write_csv
from .visualize_yolo_r3d18 import prediction_at_frame, verify_render
from .yolo_dataset import YoloClipDataset, annotated_path


ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    'The checkpoint was trained on annotated full-scene clips, not person crops. Crop scores are experimental; person-level accuracy/recall and calibration are unknown.',
    'Parent-video labels are weak labels, not verified person/event truth. No person-level recall is claimed.',
    'ByteTrack IDs are temporary tracking IDs, not verified human identities; switches and missed detections remain possible.',
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


def detect_people(source, info, detector):
    tracker, frames = make_tracker(), []
    for frame in tqdm(decoded_frames(source), total=info.frame_count,
                      desc='YOLO+ByteTrack ' + source.stem, unit='frame', mininterval=1):
        image = frame.to_ndarray(format='bgr24')
        result = detector.predict(image, classes=[0], conf=0.15, imgsz=640,
                                  device='cuda', verbose=False)[0]
        people = attach_tracks(result.boxes, tracker, image)
        ids = [p['track_id'] for p in people if p['track_id'] is not None]
        if len(ids) != len(set(ids)):
            raise ValueError('Tracker assigned the same ID to multiple detections in one frame')
        frames.append(people)
    if len(frames) != info.frame_count:
        raise ValueError('Source/detection frame count mismatch')
    return frames


def score_people(source, frames, plans, model):
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
                probability, padded = score_clip(model, samples.pop(key), 'cuda')
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
    if prediction is None:
        return f'{identity} Skor yok | 16 gerçek takip örneği yok', '#FFC857'
    probability = float(prediction['drowning_probability'])
    high = probability >= threshold
    label = 'Boğulma riski (deneysel)' if high else 'Normal yüzücü (model)'
    return f'{identity} {label} | P(boğulma sınıfı): %{probability * 100:.1f}', '#FF4444' if high else '#30EF90'


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
        rows.sort(key=lambda r: r['start_frame'])
        lookups[track] = (rows, [r['start_frame'] for r in rows])
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
                prediction = prediction_at_frame(rows, starts, index)
                text, color = person_text(person, prediction, threshold)
                x1, y1, x2, y2 = person['box']
                draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
                bounds = draw.textbbox((0, 0), text, font=font)
                width, height = bounds[2] - bounds[0] + 10, bounds[3] - bounds[1] + 14
                left = max(0, min(x1, info.width - width))
                top = max(44, y1 - height)
                draw.rectangle((left, top, left + width, top + height), fill='#101923')
                draw.text((left + 4, top + 1), text, font=font, fill=color)
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


def run(threshold, root=ROOT):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be strictly between zero and one')
    root = Path(root).resolve()
    base = root / 'results/aquasafe_yolo_r3d18_v1'
    output = base / 'visualization/person_level_t030_sampled16'
    legacy = base / 'visualization/person_level_t030'
    checkpoint, weight = base / 'training/best_model.pt', root / 'yolov8n.pt'
    if threshold != 0.30:
        raise ValueError('This experiment was explicitly configured for the user-selected threshold 0.30')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for person detection and crop inference')
    dataset = YoloClipDataset(root, base / 'clip_manifest.csv', split='validation')
    grouped = defaultdict(list)
    for row in dataset.rows:
        grouped[row['video_id']].append(row)
    protected = [checkpoint, weight, base / 'clip_manifest.csv', base / 'training/history.json',
                 root / 'src/video_classification/train_yolo_r3d18.py',
                 root / 'src/video_classification/model.py', root / 'src/video_classification/preprocessing.py',
                 Path(__file__).resolve(), root / 'src/drowning_demo/detection.py',
                 root / 'src/drowning_demo/swimmers.py']
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in protected}
    provenance = {'input_sha256': hashes, 'threshold': threshold, 'detector': 'yolov8n',
                  'confidence': 0.15, 'imgsz': 640, 'tracker': 'existing make_tracker() ByteTrack defaults',
                  'crop': 'existing crop_person: 1.3x square context, existing transform_clip',
                  'clip_policy': 'User-approved: same confirmed ID in all 16 real sampled frames of the existing parent clip; unsampled-frame detection gaps accepted.'}
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'config.json'
    if config.exists():
        if json.loads(config.read_text(encoding='utf-8')) != provenance:
            raise ValueError('Existing person experiment has different inputs/config; refusing to reuse it')
    else:
        config.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    if (legacy / 'config.json').exists():
        old = json.loads((legacy / 'config.json').read_text(encoding='utf-8'))
        if (any(old[k] != provenance[k] for k in ('detector', 'confidence', 'imgsz', 'tracker'))
                or old['input_sha256']['yolov8n.pt'] != hashes['yolov8n.pt']):
            raise ValueError('Existing detection settings differ; refusing to reuse coordinates')
    (output / 'detections').mkdir(exist_ok=True)
    (output / 'scores').mkdir(exist_ok=True)
    detector = YOLO(str(weight))
    model = build_r3d18(2, pretrained=False).to('cuda')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    all_predictions, summaries = [], []
    for video_id, clips in tqdm(grouped.items(), desc='Person validation videos', unit='video'):
        source, _ = annotated_path(root, clips[0]['source_path'])
        source_hash = sha256_file(source)
        if source_hash != clips[0]['yolo_sha256']:
            raise ValueError('Source AVI changed: ' + str(source))
        info = probe_video(source)
        key = source.stem + '_' + video_id[-12:]
        cache = output / 'detections' / (key + '.json')
        if not cache.exists() and (legacy / 'detections' / cache.name).exists():
            cache = legacy / 'detections' / cache.name
        if cache.exists():
            stored = json.loads(cache.read_text(encoding='utf-8'))
            if stored['source_sha256'] != source_hash:
                raise ValueError('Detection cache source mismatch')
            frames = stored['frames']
        else:
            frames = detect_people(source, info, detector)
            with cache.open('x', encoding='utf-8') as target:
                json.dump({'source_path': clips[0]['source_path'], 'source_sha256': source_hash,
                           'fps': info.fps, 'frame_count': info.frame_count, 'frames': frames}, target)
        plans, skipped = person_clip_plan(clips, frames)
        score_path = output / 'scores' / (key + '.json')
        if score_path.exists():
            predictions = json.loads(score_path.read_text(encoding='utf-8'))
            if {r['person_clip_id'] for r in predictions} != {r['person_clip_id'] for r in plans}:
                raise ValueError('Person score cache does not match current track plan')
        else:
            predictions = score_people(source, frames, plans, model)
            with score_path.open('x', encoding='utf-8') as target:
                json.dump(predictions, target, ensure_ascii=False, indent=2)
        all_predictions.extend(predictions)
        name = ('drowning_' if int(clips[0]['video_label']) else 'normal_') + source.stem + '.mp4'
        exported = render_people(root, clips, frames, predictions, output / 'videos' / name, threshold)
        if sha256_file(source) != source_hash:
            raise ValueError('Source changed during analysis')
        summaries.append({**exported, 'detected_person_boxes': sum(map(len, frames)),
                          'detection_cache': str(cache),
                          'confirmed_track_ids': len({p['track_id'] for f in frames for p in f if p['track_id'] is not None}),
                          'scored_person_clips': len(predictions), 'skipped_parent_clips': skipped})
        report = {'complete': len(summaries) == len(grouped), 'videos': summaries,
                  'expected_videos': len(grouped), 'scored_person_clips': len(all_predictions),
                  'threshold': threshold, 'person_level_recall': 'unknown',
                  'checkpoint_crop_suitability': 'Architecture compatible; domain suitability unverified',
                  'limitations': LIMITATIONS, 'training_started': False}
        (output / 'person_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_path = output / 'person_predictions.csv'
    if not csv_path.exists() and all_predictions:
        write_csv(csv_path, all_predictions, list(all_predictions[0]))
    if hashes != {str(p.relative_to(root)): sha256_file(p) for p in protected}:
        raise RuntimeError('A protected input changed while person analysis was running')
    report['protected_inputs_unchanged'] = True
    (output / 'person_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'videos': len(summaries), 'person_clips': len(all_predictions), 'output': str(output)}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()
    try:
        run(args.threshold)
    except Exception as error:
        raise SystemExit(f'Person analysis failed: {error}. Existing sources/checkpoints are not overwritten.') from error
