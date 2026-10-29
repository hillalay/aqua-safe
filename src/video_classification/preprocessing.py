from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


R3D_MEAN = (0.43216, 0.394666, 0.37645)
R3D_STD = (0.22803, 0.22145, 0.216989)


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass(frozen=True)
class ClipRecord:
    video_id: str
    clip_id: str
    source_path: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    style: str
    split: str
    frame_step: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if not ok or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video stream: {path}")
    return VideoInfo(fps, frame_count, frame_count / fps, width, height)


def build_clip_records(row: dict[str, str], info: VideoInfo, sampling_fps: float = 8.0,
                       clip_length: int = 16, stride: int = 16) -> list[ClipRecord]:
    if sampling_fps <= 0 or clip_length <= 0 or stride <= 0:
        raise ValueError("Sampling FPS, clip length and stride must be positive")
    frame_step = max(1, round(info.fps / sampling_fps))
    span = (clip_length - 1) * frame_step + 1
    records = []
    for clip_index, start in enumerate(range(0, max(0, info.frame_count - span + 1), stride * frame_step)):
        end = start + (clip_length - 1) * frame_step
        records.append(ClipRecord(
            row["video_id"], f"{row['video_id']}_C{clip_index:04d}", row["path"], start, end,
            start / info.fps, end / info.fps, row["style"], row.get("split", "unassigned"), frame_step,
        ))
    return records


def decode_clip(path: Path, record: ClipRecord, clip_length: int = 16) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {path}")
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, record.start_frame)
        for index in range(clip_length):
            frame_number = record.start_frame + index * record.frame_step
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Frame {frame_number} could not be decoded: {path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for _ in range(record.frame_step - 1 if index < clip_length - 1 else 0):
                if not cap.grab():
                    raise RuntimeError(f"Frame after {frame_number} could not be decoded: {path}")
    finally:
        cap.release()
    if len(frames) != clip_length:
        raise RuntimeError(f"Expected {clip_length} frames, decoded {len(frames)}: {path}")
    return np.stack(frames)


def transform_clip(frames: np.ndarray, resize_short_side: int = 128, crop_size: int = 112) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB frames with shape (T,H,W,3)")
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    height, width = tensor.shape[-2:]
    scale = resize_short_side / min(height, width)
    resized = functional.interpolate(tensor, size=(round(height * scale), round(width * scale)),
                                     mode="bilinear", align_corners=False)
    top = (resized.shape[-2] - crop_size) // 2
    left = (resized.shape[-1] - crop_size) // 2
    if top < 0 or left < 0:
        raise ValueError("crop_size cannot exceed resized frame dimensions")
    tensor = resized[:, :, top:top + crop_size, left:left + crop_size]
    mean = torch.tensor(R3D_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(R3D_STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).permute(1, 0, 2, 3).contiguous()
