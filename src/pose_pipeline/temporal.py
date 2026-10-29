from __future__ import annotations

from pathlib import Path

import numpy as np

from .quality import landmark_is_usable, landmark_quality, window_quality
from .schema import COMMON_LANDMARKS, PoseFrame


def validate_split_groups(rows: list[dict[str, str]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        split = row.get("split", "unassigned")
        swimmer, session = row.get("swimmer_id", "unknown"), row.get("session_id", "unknown")
        if swimmer != "unknown":
            group = ("swimmer", swimmer)
        elif session != "unknown":
            group = ("session", session)
        else:
            group = ("video", row["video_id"])
        if group in seen and seen[group] != split:
            raise ValueError(f"Group {group} occurs in both {seen[group]} and {split}")
        seen[group] = split


def save_windows(records: list[PoseFrame], output: Path, style: str, split: str, window_size: int = 30) -> int:
    # shortcut: non-overlapping windows drop the incomplete tail, revisit when the training protocol defines padding/stride
    window_count = len(records) // window_size
    features = np.zeros((window_count, window_size, len(COMMON_LANDMARKS), 3), dtype=np.float32)
    mask = np.zeros((window_count, window_size, len(COMMON_LANDMARKS)), dtype=bool)
    timestamps = np.zeros((window_count, window_size), dtype=np.float64)
    missing_rate = np.zeros(window_count, dtype=np.float32)
    training_usable = np.zeros(window_count, dtype=bool)
    for window_index in range(window_count):
        window = records[window_index * window_size:(window_index + 1) * window_size]
        missing_rate[window_index], training_usable[window_index] = window_quality(window)
        for time_index, record in enumerate(window):
            timestamps[window_index, time_index] = record.timestamp_sec
            for point_index, name in enumerate(COMMON_LANDMARKS):
                point = record.common_landmarks[name]
                if not landmark_is_usable(point):
                    continue
                features[window_index, time_index, point_index] = (point.x, point.y, landmark_quality(point))
                mask[window_index, time_index, point_index] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, mask=mask, timestamps=timestamps,
                        missing_rate=missing_rate, training_usable=training_usable,
                        style=np.array(style), split=np.array(split), video_id=np.array(records[0].video_id if records else ""))
    return window_count
