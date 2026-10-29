"""Offline per-track behavior scoring and annotated MP4 export."""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np
from PIL import ImageDraw
import torch
from ultralytics import YOLO

from src.drowning_demo.analysis import (
    ROOT, DEFAULT_CHECKPOINT, decoded_frames, inspect_video, load_model,
    new_output_dir, overlay_font, sha256,
)
from src.video_classification.preprocessing import transform_clip


def crop_person(rgb, box):
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = box
    # Square context preserves the whole person through the model's center crop.
    side = max(x2 - x1, y2 - y1) * 1.3
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, right = max(0, int(cx - side / 2)), min(width, int(cx + side / 2))
    top, bottom = max(0, int(cy - side / 2)), min(height, int(cy + side / 2))
    if right <= left or bottom <= top:
        raise ValueError("Geçersiz kişi kutusu")
    return transform_clip(rgb[None, top:bottom, left:right])[:, 0]


def score_clip(model, samples, device):
    frames = list(samples)
    if not frames:
        raise ValueError("Boş kişi dizisi")
    if len(frames) < 16:
        return None, 0
    if len(frames) != 16:
        raise ValueError("Model tam 16 gerçek örnek gerektirir")
    with torch.inference_mode():
        logits = model(torch.stack(frames, dim=1).unsqueeze(0).to(device)).float()
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("Geçersiz model skoru")
    return float((logits[0, 1] - logits[0, 0]).sigmoid()), 0


def collect_tracks(source, info, detector, model, device):
    step = max(1, round(info['fps'] / 8))
    active, windows, frames = {}, {}, []
    segment_id = 0

    def score(state):
        value, padded = score_clip(model, state['samples'], device)
        windows[state['segment']].append({
            'center_frame': (state['indices'][0] + state['indices'][-1]) / 2,
            'start_frame': state['indices'][0], 'end_frame': state['indices'][-1],
            'drowning_behavior_score': value, 'padded_frames': padded,
            'real_sample_count': len(state['samples']),
            'status': 'scored' if value is not None else 'insufficient_real_frames',
        })
        state['last_scored'] = state['indices'][-1]

    for index, frame in enumerate(decoded_frames(source)):
        rgb = frame.to_ndarray(format='rgb24')
        result = detector.track(rgb[:, :, ::-1].copy(), persist=True, tracker='bytetrack.yaml',
                                classes=[0], conf=0.1, device=device, verbose=False)[0]
        visible = []
        # A detection gap ends the scoring segment; never combine separated motion.
        ids = [] if result.boxes.id is None else result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        for track_id in list(active):
            if track_id not in ids:
                state = active.pop(track_id)
                if state['indices'] and state['last_scored'] != state['indices'][-1]:
                    score(state)
        for track_id, box in zip(ids, boxes):
            if track_id not in active:
                segment_id += 1
                active[track_id] = {'segment': segment_id, 'samples': deque(maxlen=16),
                                    'indices': deque(maxlen=16), 'count': 0, 'last_scored': -1,
                                    'first': index}
                windows[segment_id] = []
            state = active[track_id]
            if (index - state['first']) % step == 0:
                state['samples'].append(crop_person(rgb, box))
                state['indices'].append(index)
                state['count'] += 1
                if state['count'] >= 16 and (state['count'] - 16) % 8 == 0:
                    score(state)
            visible.append({'track_id': track_id, 'segment': state['segment'], 'box': box})
        frames.append(visible)
        if index % 60 == 0:
            print(f"Kişi analizi: {index + 1}/{info['frame_count']}", flush=True)
    for state in active.values():
        if state['indices'] and state['last_scored'] != state['indices'][-1]:
            score(state)
    if len(frames) != info['frame_count']:
        raise ValueError('Kaynak kare sayısı değişti')
    return frames, windows


def draw_people(frame, people, windows, index):
    image = frame.to_image()
    draw = ImageDraw.Draw(image)
    font = overlay_font(max(14, round(image.width / 85)))
    for person in people:
        record = min(windows[person['segment']], key=lambda row: abs(row['center_frame'] - index)) if windows is not None else None
        score = record['drowning_behavior_score'] if record else None
        if record is None:
            color = '#ffc857'
            identity = f'#{person["track_id"]}' if person['track_id'] is not None else 'ID BEKLENİYOR'
            text = f'{identity} ANALİZ BEKLENİYOR | İnsan tespit skoru: %{person["confidence"]*100:.1f}'
        elif score is None:
            color = '#ffc857'
            text = f"#{person['track_id']} YETERSİZ VERİ | {record['real_sample_count']}/16 gerçek örnek | Skor yok"
        else:
            color = '#ff5555' if score >= 0.5 else '#30ef90'
            label = 'BOĞULMA DAVRANIŞI' if score >= 0.5 else 'NORMAL YÜZÜCÜ'
            # Always show the drowning-class score, including for normal swimmers.
            text = f"#{person['track_id']} {label} | Boğulma skoru: %{score * 100:.1f}"
        x1, y1, x2, y2 = person['box']
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        bounds = draw.textbbox((0, 0), text, font=font)
        tw, th = bounds[2], bounds[3] + 8
        left, top = max(0, min(x1, image.width - tw - 8)), max(0, y1 - th)
        draw.rectangle((left, top, left + tw + 8, top + th), fill='#101923')
        draw.text((left + 4, top + 2), text, fill=color, font=font)
    if not people:
        draw.text((8, 8), 'KİŞİ TESPİT EDİLEMEDİ | Skor yok', font=font, fill='#ffc857',
                  stroke_width=2, stroke_fill='black')
    footer = 'Yalnız insan tespiti • Davranış analizi çalıştırılmadı' if windows is None else 'Deneysel davranış skoru • Kalibre olasılık değildir'
    draw.text((8, image.height - 30), footer,
              font=font, fill='white', stroke_width=2, stroke_fill='black')
    return image


def analyze_swimmers(source, checkpoint=DEFAULT_CHECKPOINT, device='cpu', detector_path=ROOT / 'yolov8s.pt',
                     *, detection_only=False, confidence=0.1, imgsz=960):
    source, checkpoint, detector_path = Path(source).resolve(), Path(checkpoint).resolve(), Path(detector_path).resolve()
    for path in ((source, detector_path) if detection_only else (source, checkpoint, detector_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = new_output_dir()
    partial = output / 'annotated_video.partial.mp4'
    try:
        info = inspect_video(source)
        detector = YOLO(str(detector_path))
        if detection_only:
            from src.drowning_demo.detection import collect_detections
            frames = collect_detections(source, info, detector, device, confidence, imgsz)
            windows = None
        else:
            model, _ = load_model(checkpoint, device)
            frames, windows = collect_tracks(source, info, detector, model, device)
        time_base = Fraction(1, 1000000)
        with av.open(str(partial), 'w', options={'movflags': '+faststart'}) as container:
            stream = container.add_stream('libx264', rate=Fraction(info['rate_numerator'], info['rate_denominator']))
            stream.width, stream.height = info['width'], info['height']
            stream.pix_fmt = 'yuv420p' if info['width'] % 2 == info['height'] % 2 == 0 else 'yuv444p'
            stream.time_base = stream.codec_context.time_base = time_base
            stream.options = {'crf': '20', 'preset': 'veryfast', 'bf': '0'}
            for index, frame in enumerate(decoded_frames(source)):
                timestamp = float(frame.pts * frame.time_base) - info['source_pts_origin_sec']
                if index >= len(frames) or abs(timestamp - info['timestamps_sec'][index]) > 1e-6:
                    raise ValueError('Kaynak video değişti')
                annotated = av.VideoFrame.from_image(draw_people(frame, frames[index], windows, index))
                annotated.pts = round(timestamp / time_base)
                annotated.time_base = time_base
                end = info['timestamps_sec'][index + 1] if index + 1 < len(frames) else info['duration_sec']
                annotated.duration = max(1, round((end - timestamp) / time_base))
                for packet in stream.encode(annotated):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        exported = inspect_video(partial)
        if any(exported[key] != info[key] for key in ('width', 'height', 'frame_count')) or not np.allclose(
                exported['timestamps_sec'], info['timestamps_sec'], atol=0.0001, rtol=0):
            raise ValueError('MP4 kare/zaman doğrulaması başarısız')
        summary = {'source': str(source), 'source_sha256': sha256(source), 'checkpoint': None if detection_only else str(checkpoint),
                   'checkpoint_sha256': None if detection_only else sha256(checkpoint), 'video': info,
                   'mode': 'person_detection_only' if detection_only else 'behavior',
                   'detector': str(detector_path), 'detector_sha256': sha256(detector_path),
                   'detection_settings': {'confidence': confidence, 'imgsz': imgsz} if detection_only else None,
                   'track_count': len({p['track_id'] for row in frames for p in row if p['track_id'] is not None}),
                   'max_simultaneous_tracks': max(map(len, frames)),
                   'frames_with_detections': sum(bool(row) for row in frames),
                   'windows': windows, 'tracks_by_frame': frames, 'audio_preserved': False,
                   'scored_windows': sum(w['status'] == 'scored' for rows in (windows or {}).values() for w in rows),
                   'insufficient_windows': sum(w['status'] == 'insufficient_real_frames' for rows in (windows or {}).values() for w in rows),
                   'minimum_real_samples': 16, 'short_track_policy': 'no repetition, no score or class label',
                   'score_is_calibrated_probability': False,
                   'limitations': 'Video-label model applied to person crops; person-level accuracy unvalidated. '
                   'YOLO detects people, including poolside people; missed detections and ID switches are possible. '
                   'Short tracks have no score or behavior label. Offline nearest-window scoring, not event localization.'}
        if detection_only:
            from src.drowning_demo.detection import detection_gaps
            summary['detection_continuity'] = detection_gaps(frames)
            summary['limitations'] = ('All raw detections are drawn immediately; no predicted or interpolated boxes. '
                                      'Unconfirmed identities remain null. Detection confidence is not behavior probability. '
                                      'Detector misses and false positives remain possible; no behavior model was loaded.')
        (output / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        partial.rename(output / 'annotated_video.mp4')
        print(f"Takip: {summary['frames_with_detections']}/{info['frame_count']} karede kişi; "
              f"{summary['track_count']} ID; {summary['scored_windows']} skorlanan, "
              f"{summary['insufficient_windows']} yetersiz pencere", flush=True)
        return output / 'annotated_video.mp4'
    except Exception as error:
        partial.unlink(missing_ok=True)
        (output / 'error.json').write_text(json.dumps({'error': str(error)}, ensure_ascii=False), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description='Her kişiyi takip et, davranış skorunu MP4 üzerine yaz.')
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--detection-only', action='store_true', help='Draw every person detection without loading R3D-18')
    parser.add_argument('--detector', type=Path, default=ROOT/'yolov8s.pt')
    parser.add_argument('--confidence', type=float, default=0.1)
    parser.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args()
    for video in args.videos:
        print(analyze_swimmers(video, args.checkpoint, args.device, args.detector,
                              detection_only=args.detection_only, confidence=args.confidence, imgsz=args.imgsz), flush=True)


if __name__ == '__main__':
    main()
