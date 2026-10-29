"""Lazy R3D-18 inputs from annotated YOLO videos, with experimental split checks."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import ClipRecord, decode_clip, transform_clip


FOLDER_LABELS = {'bogulma': 1, 'kelebek': 0, 'kurbagalama': 0, 'serbest': 0, 'sirtustu': 0}
UNKNOWN = {'', 'unknown', 'unassigned', None}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def annotated_path(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or path.drive or '..' in path.parts:
        raise ValueError('Expected a project-relative YOLO path')
    path = (root / path).resolve()
    output = (root / 'results/detect_all').resolve()
    if not output.is_relative_to(root) or not path.is_relative_to(output):
        raise ValueError('Only results/detect_all videos may be loaded')
    parts = path.relative_to(output).parts
    if len(parts) != 3 or parts[0] not in FOLDER_LABELS or path.suffix.lower() != '.avi':
        raise ValueError('Unexpected annotated video layout')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, FOLDER_LABELS[parts[0]]


def clip_record(row):
    return ClipRecord(row['video_id'], row['clip_id'], row['source_path'],
                      int(row['start_frame']), int(row['end_frame']),
                      float(row['start_time']), float(row['end_time']),
                      row['class_label'], row['draft_split'], int(row['frame_step']))


def validate_manifest(root, rows):
    if not rows:
        raise ValueError('Empty clip manifest')
    seen_clips, group_splits, by_video = set(), defaultdict(dict), {}
    paths, hashes, resolved, spans = {}, {}, {}, defaultdict(list)
    for row in rows:
        if row['clip_id'] in seen_clips:
            raise ValueError('Repeated clip ID')
        seen_clips.add(row['clip_id'])
        if row['source_path'] not in resolved:
            resolved[row['source_path']] = annotated_path(root, row['source_path'])
        path, folder_label = resolved[row['source_path']]
        label = int(row['video_label'])
        if label != folder_label or row['class_label'] != ('drowning' if label else 'normal'):
            raise ValueError('Folder and video label disagree')
        if row['label_strength'] != 'weak_video_label' or row['clip_event_label'] != 'unknown':
            raise ValueError('Folder labels must not become verified clip labels')
        if row['split_status'] != 'experimental_draft' or row['draft_split'] not in {'train', 'validation'}:
            raise ValueError('Only experimental train/validation drafts are supported')
        record = clip_record(row)
        fps, count = float(row['source_fps']), int(row['source_frame_count'])
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Invalid source metadata')
        if (record.frame_step != max(1, round(fps / 8)) or record.start_frame < 0 or record.end_frame >= count
                or record.end_frame != record.start_frame + 15 * record.frame_step):
            raise ValueError('A clip must contain exactly 16 in-range sampled frames')
        if (not math.isfinite(record.start_time) or not math.isfinite(record.end_time)
                or abs(record.start_time - record.start_frame / fps) > 1e-6
                or abs(record.end_time - record.end_frame / fps) > 1e-6):
            raise ValueError('Frame/time mismatch')
        identity = tuple(str(row[k]) for k in ('source_path', 'video_label', 'sha256', 'yolo_sha256',
                                              'source_fps', 'source_frame_count', 'relation_group',
                                              'leakage_group', 'source_id', 'session_id', 'confirmed_source_group'))
        if row['video_id'] in by_video and by_video[row['video_id']] != identity:
            raise ValueError('Inconsistent parent video metadata')
        by_video[row['video_id']] = identity
        spans[row['video_id']].append((record.start_frame, record.end_frame))
        for key, mapping in ((str(path), paths), (row['sha256'], hashes), (row['yolo_sha256'], hashes)):
            if not key:
                raise ValueError('Missing content hash')
            if key in mapping and mapping[key] != row['video_id']:
                raise ValueError('Duplicate video content/path under different IDs')
            mapping[key] = row['video_id']
        for field in ('video_id', 'source_path', 'sha256', 'yolo_sha256', 'relation_group',
                      'leakage_group', 'source_id', 'confirmed_source_group'):
            key = row[field]
            if key in UNKNOWN:
                if field in {'relation_group', 'leakage_group'}:
                    raise ValueError('Missing precautionary group')
                continue
            if key in group_splits[field] and group_splits[field][key] != row['draft_split']:
                raise ValueError(f'{field} crosses partitions')
            group_splits[field][key] = row['draft_split']
    for intervals in spans.values():
        ordered = sorted(intervals)
        if any(current[0] <= previous[1] for previous, current in zip(ordered, ordered[1:])):
            raise ValueError('Repeated or overlapping clips within a video')
    return len(by_video)


class YoloClipDataset(Dataset):
    """Returns (RGB tensor, weak parent-video target, provenance metadata)."""

    def __init__(self, root, manifest, split=None):
        self.root = Path(root).resolve()
        rows = read_csv(manifest)
        validate_manifest(self.root, rows)
        if split not in {None, 'train', 'validation'}:
            raise ValueError('There is no independent test split')
        self.rows = [r for r in rows if split is None or r['draft_split'] == split]
        if not self.rows:
            raise ValueError('No clips in requested partition')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path, _ = annotated_path(self.root, row['source_path'])
        tensor = transform_clip(decode_clip(path, clip_record(row)))
        if tuple(tensor.shape) != (3, 16, 112, 112) or not torch.isfinite(tensor).all():
            raise ValueError('Invalid R3D-18 input tensor')
        return tensor, int(row['video_label']), dict(row)


def make_loaders(root, manifest, batch_size=2, num_workers=0, seed=42):
    """Two draft loaders; batches are [B,3,16,112,112], with int64 video targets."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError('Invalid DataLoader settings')
    datasets = {split: YoloClipDataset(root, manifest, split) for split in ('train', 'validation')}
    return {split: DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                              num_workers=num_workers, drop_last=False,
                              generator=torch.Generator().manual_seed(seed))
            for split, dataset in datasets.items()}
