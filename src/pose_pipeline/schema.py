from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


COMMON_LANDMARKS = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    z: float | None = None
    confidence: float | None = None
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True)
class PoseFrame:
    video_id: str
    frame_number: int
    timestamp_sec: float
    model_name: str
    model_version: str
    detected: bool
    native_landmarks: tuple[Landmark, ...]
    common_landmarks: dict[str, Landmark | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "video_id": self.video_id,
            "frame_number": self.frame_number,
            "timestamp_sec": self.timestamp_sec,
            "model": {"name": self.model_name, "version": self.model_version},
            "pose_status": "detected" if self.detected else "not_detected",
            "native_landmarks": [asdict(point) for point in self.native_landmarks],
            "common_landmarks": {
                name: asdict(self.common_landmarks[name]) if self.common_landmarks.get(name) else None
                for name in COMMON_LANDMARKS
            },
        }

    def validate(self) -> None:
        if self.timestamp_sec < 0 or self.frame_number < 0:
            raise ValueError("Frame number and timestamp must be non-negative")
        if set(self.common_landmarks) != set(COMMON_LANDMARKS):
            raise ValueError("Common landmark keys do not match the schema")
        if not self.detected and self.native_landmarks:
            raise ValueError("A non-detected pose cannot contain native landmarks")
