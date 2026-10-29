"""Compare local person detectors on fixed frames; reference boxes are evaluation-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from src.drowning_demo.analysis import ROOT, decoded_frames, sha256
from src.video_review.reviews import atomic_write_json


SOURCES = {
    'data/raw/kelebek/kelebek2.mp4': [0, 30, 60, 90, 120, 160, 200, 254, 320, 385, 430, 473],
    'data/raw/bogulma/bogulma1.mp4': [0, 20, 40, 60, 80, 100, 120, 150, 180, 210, 235, 254],
    'data/test_videos/1020(3)-1.mp4': [0, 15, 30, 45, 60, 72, 85, 104, 113, 125, 150, 179],
}


def match_boxes(reference, predictions, threshold=0.5):
    pairs = []
    for i, a in enumerate(reference):
        for j, b in enumerate(predictions):
            area = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
            if union > 0 and area/union >= threshold:
                pairs.append((area/union, i, j))
    used_reference, used_predictions = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_reference and j not in used_predictions:
            used_reference.add(i)
            used_predictions.add(j)
    return {'tp': len(used_reference), 'fn': len(reference)-len(used_reference),
            'fp': len(predictions)-len(used_predictions),
            'missed_reference_indices': sorted(set(range(len(reference)))-used_reference),
            'false_prediction_indices': sorted(set(range(len(predictions)))-used_predictions)}


def sweep(output):
    if output.exists():
        raise FileExistsError(output)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = [(path, index, frame.to_ndarray(format='bgr24')) for path, anchors in SOURCES.items()
              for index, frame in enumerate(decoded_frames(ROOT/path)) if index in anchors]
    report = {'device': device, 'sources': {p: sha256(ROOT/p) for p in SOURCES},
              'scope': '36 fixed frames; tuning and assessment on the same three videos, not independent video accuracy.',
              'reference_boxes': {}, 'experiments': []}
    for weight in ('yolov8n.pt', 'yolov8s.pt'):
        model = YOLO(str(ROOT/weight))
        for size in (640, 960, 1280, 1600, 1920):
            records = []
            for path, index, image in frames:
                result = model.predict(image, classes=[0], conf=0.05, imgsz=size, device=device, verbose=False)[0]
                records.append({'source': path, 'frame': index, 'boxes': result.boxes.xyxy.cpu().tolist(),
                                'confidence': result.boxes.conf.cpu().tolist()})
            report['experiments'].append({'weights': weight, 'weights_sha256': sha256(ROOT/weight),
                                          'imgsz': size, 'predictions_at_confidence_005': records})
            print(f'{weight} imgsz={size}: {sum(len(r["boxes"]) for r in records)} boxes on 36 frames', flush=True)
    atomic_write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/person_detection_review/detection_comparison.json')
    args = parser.parse_args()
    sweep(args.output.resolve())


if __name__ == '__main__':
    main()
