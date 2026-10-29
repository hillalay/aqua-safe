"""Raw person detections remain visible while ByteTrack confirms their identities."""
from __future__ import annotations

from types import SimpleNamespace

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from src.drowning_demo.analysis import decoded_frames


def make_tracker(new_track_threshold=0.25):
    settings = YAML.load(check_yaml('bytetrack.yaml'))
    settings['new_track_thresh'] = new_track_threshold
    settings['track_high_thresh'] = new_track_threshold
    return BYTETracker(SimpleNamespace(**settings))


def attach_tracks(boxes, tracker, image=None):
    """IDs supplement raw detections; missing or unconfirmed IDs never hide a box."""
    tracks = tracker.update(boxes.cpu().numpy(), image)
    identities = {int(track[-1]): int(track[4]) for track in tracks}
    return [{'box': box, 'confidence': float(confidence), 'track_id': identities.get(index)}
            for index, (box, confidence) in enumerate(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()))]


def collect_detections(source, info, detector, device, confidence=0.1, imgsz=960):
    if not 0 < confidence < 1 or imgsz < 32:
        raise ValueError('Invalid detection confidence or input resolution')
    tracker, frames = make_tracker(), []
    for index, frame in enumerate(decoded_frames(source)):
        image = frame.to_ndarray(format='bgr24')
        result = detector.predict(image, classes=[0], conf=confidence, imgsz=imgsz,
                                  device=device, verbose=False)[0]
        frames.append(attach_tracks(result.boxes, tracker, image))
        if index % 60 == 0:
            print(f'İnsan tespiti: {index+1}/{info["frame_count"]}', flush=True)
    if len(frames) != info['frame_count']:
        raise ValueError('Source frame count changed during detection')
    return frames


def detection_gaps(frames):
    """Report observed box absence, without claiming that no human was present."""
    lengths, current = [], 0
    for people in frames:
        if not people:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return {'frames_with_boxes': sum(bool(p) for p in frames), 'total_frames': len(frames),
            'empty_frame_runs': lengths, 'longest_empty_run': max(lengths, default=0),
            'detections_without_confirmed_id': sum(p['track_id'] is None for f in frames for p in f),
            'maximum_simultaneous_boxes': max(map(len, frames), default=0)}
