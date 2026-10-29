from __future__ import annotations

from .schema import COMMON_LANDMARKS, Landmark, PoseFrame


def landmark_quality(point: Landmark | None) -> float | None:
    if point is None:
        return None
    values = (value for value in (point.confidence, point.visibility, point.presence) if value is not None)
    return min(values, default=None)


def landmark_is_usable(point: Landmark | None, threshold: float = 0.5) -> bool:
    quality = landmark_quality(point)
    return point is not None and quality is not None and quality >= threshold


def pose_is_usable(common: dict[str, Landmark | None], threshold: float = 0.5) -> bool:
    usable = {name for name, point in common.items() if landmark_is_usable(point, threshold)}
    torso = bool(usable & {"left_shoulder", "right_shoulder"}) and bool(usable & {"left_hip", "right_hip"})
    return len(usable) >= 6 and torso


def roi_to_original(point: Landmark, crop: tuple[int, int, int, int], frame_width: int, frame_height: int) -> Landmark:
    left, top, right, bottom = crop
    return Landmark(
        point.name,
        (left + point.x * (right - left)) / frame_width,
        (top + point.y * (bottom - top)) / frame_height,
        point.z,
        point.confidence,
        point.visibility,
        point.presence,
    )


def window_quality(records: list[PoseFrame], threshold: float = 0.5) -> tuple[float, bool]:
    if not records:
        return 1.0, False
    present = sum(
        landmark_is_usable(record.common_landmarks[name], threshold)
        for record in records for name in COMMON_LANDMARKS
    )
    missing_rate = 1 - present / (len(records) * len(COMMON_LANDMARKS))
    usable_frames = sum(pose_is_usable(record.common_landmarks, threshold) for record in records)
    return missing_rate, missing_rate <= 0.3 and usable_frames / len(records) >= 0.8
