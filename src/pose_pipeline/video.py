from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .extractors import PoseExtractor
from .quality import landmark_is_usable, landmark_quality, pose_is_usable
from .schema import COMMON_LANDMARKS, Landmark, PoseFrame


def sampled_frame_numbers(frame_count: int, source_fps: float, target_fps: float) -> list[int]:
    if frame_count < 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame count must be non-negative and FPS values must be positive")
    effective_fps = min(source_fps, target_fps)
    count = int(np.ceil(frame_count * effective_fps / source_fps))
    return sorted({min(frame_count - 1, round(i * source_fps / effective_fps)) for i in range(count)}) if frame_count else []


def iter_sampled_frames(path: Path, target_fps: float) -> Iterator[tuple[int, float, np.ndarray, float]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for wanted in sampled_frame_numbers(frame_count, source_fps, target_fps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, wanted)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Frame {wanted} could not be read from {path}")
            yield wanted, wanted / source_fps, frame, min(source_fps, target_fps)
    finally:
        cap.release()


def _is_present(point: Landmark | None, threshold: float) -> bool:
    return landmark_is_usable(point, threshold)


def draw_pose(frame: np.ndarray, common: dict[str, Landmark | None], extractor: PoseExtractor, threshold: float) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for start, end in extractor.connections:
        a, b = common[start], common[end]
        if _is_present(a, threshold) and _is_present(b, threshold):
            cv2.line(output, (round(a.x * width), round(a.y * height)), (round(b.x * width), round(b.y * height)), (0, 220, 0), 2)
    for point in common.values():
        if _is_present(point, threshold):
            cv2.circle(output, (round(point.x * width), round(point.y * height)), 3, (0, 80, 255), -1)
    return output


def process_video(
    video_path: Path,
    video_id: str,
    extractor: PoseExtractor,
    output_dir: Path,
    target_fps: float = 10.0,
    quality_threshold: float = 0.5,
) -> tuple[list[PoseFrame], dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[PoseFrame] = []
    writer = None
    started = time.perf_counter()
    try:
        for frame_number, timestamp, frame, output_fps in iter_sampled_frames(video_path, target_fps):
            native, common = extractor.extract(frame, round(timestamp * 1000))
            record = PoseFrame(video_id, frame_number, timestamp, extractor.name, extractor.version, bool(native), native, common)
            record.validate()
            records.append(record)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Annotated video writer could not be opened")
            writer.write(draw_pose(frame, common, extractor, quality_threshold))
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    with (output_dir / "frames.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    metrics = calculate_metrics(records, elapsed, quality_threshold)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return records, metrics


def calculate_metrics(records: list[PoseFrame], elapsed: float, threshold: float = 0.5) -> dict[str, object]:
    detected = sum(record.detected for record in records)
    usable = sum(pose_is_usable(record.common_landmarks, threshold) for record in records)
    points = [record.common_landmarks[name] for record in records for name in COMMON_LANDMARKS]
    present = sum(_is_present(point, threshold) for point in points)
    joint_quality = {}
    for name in COMMON_LANDMARKS:
        values = []
        for record in records:
            point = record.common_landmarks[name]
            if point:
                value = landmark_quality(point)
                if value is not None:
                    values.append(value)
        joint_quality[name] = float(np.mean(values)) if values else None
    movement = []
    for previous, current in zip(records, records[1:]):
        distances = []
        for name in COMMON_LANDMARKS:
            a, b = previous.common_landmarks[name], current.common_landmarks[name]
            if _is_present(a, threshold) and _is_present(b, threshold):
                distances.append(float(np.hypot(b.x - a.x, b.y - a.y)))
        if distances:
            movement.append(float(np.mean(distances)))
    total = len(records)
    return {
        "sampled_frames": total,
        "pose_detection_rate": detected / total if total else 0.0,
        "raw_pose_detection_rate": detected / total if total else 0.0,
        "usable_pose_detection_rate": usable / total if total else 0.0,
        "missing_common_landmark_rate": 1 - present / len(points) if points else 1.0,
        "mean_joint_quality": joint_quality,
        "mean_interframe_normalized_displacement": float(np.mean(movement)) if movement else None,
        "processing_seconds": elapsed,
        "processing_fps": total / elapsed if elapsed else None,
        "warning": "These proxy metrics do not measure pose accuracy without ground truth.",
    }
