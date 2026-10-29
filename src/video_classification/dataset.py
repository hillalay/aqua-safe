from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .preprocessing import ClipRecord, decode_clip, transform_clip


REQUIRED_COLUMNS = {
    "video_id", "path", "style", "duration_sec", "swimmer_id", "session_id", "style_verified", "split",
}


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Metadata is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("video_id values must be unique")
    return rows


def exclusion_reason(row: dict[str, str]) -> str | None:
    if row["style_verified"].lower() not in {"verified", "true", "yes"}:
        return "style_not_verified"
    if row["swimmer_id"].lower() == "unknown" or row["session_id"].lower() == "unknown":
        return "identity_or_session_unknown"
    if row["split"] not in {"train", "validation", "test"}:
        return "split_unassigned"
    return None


def validate_split_leakage(rows: list[dict[str, str]]) -> None:
    seen_video: dict[str, str] = {}
    seen_group: dict[tuple[str, str], str] = {}
    for row in rows:
        split = row["split"]
        video_id = row["video_id"]
        if video_id in seen_video and seen_video[video_id] != split:
            raise ValueError(f"Video {video_id} crosses splits")
        seen_video[video_id] = split
        if row["swimmer_id"] != "unknown" and row["session_id"] != "unknown":
            group = (row["swimmer_id"], row["session_id"])
            if group in seen_group and seen_group[group] != split:
                raise ValueError(f"Swimmer/session {group} crosses splits")
            seen_group[group] = split


def duplicate_groups(paths: list[Path]) -> list[list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        hashes[digest.hexdigest()].append(str(path))
    return [group for group in hashes.values() if len(group) > 1]


class VideoClipDataset(Dataset):
    def __init__(self, root: Path, clips: list[ClipRecord], classes: list[str],
                 tensor_cache: dict[str, torch.Tensor] | None = None):
        self.root = root
        self.clips = clips
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.tensor_cache = tensor_cache

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, object]]:
        record = self.clips[index]
        tensor = self.tensor_cache.get(record.clip_id) if self.tensor_cache is not None else None
        if tensor is None:
            tensor = transform_clip(decode_clip(self.root / Path(record.source_path), record))
            if self.tensor_cache is not None:
                self.tensor_cache[record.clip_id] = tensor
        return tensor, self.class_to_index[record.style], record.to_dict()


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["style"] for row in rows).items()))
