"""Prepare the fixed 92-video annotated-YOLO cohort; never starts training."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from src.video_review.binary_source_analysis import validate_partition
from src.video_review.inventory import sha256_file
from .preprocessing import build_clip_records, probe_video
from .prepare_binary_v1 import write_csv
from .yolo_dataset import YoloClipDataset, annotated_path, make_loaders, read_csv, validate_manifest


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAME = 'aquasafe_yolo_r3d18_v1'


def select_videos(manifest, source_groups):
    if len({r['video_id'] for r in source_groups}) != len(source_groups):
        raise ValueError('Repeated source-group metadata ID')
    groups = {r['video_id']: r for r in source_groups}
    selected = [r for r in manifest if r['weak_training_with_yolo_candidate'] == 'True']
    if Counter(r['class_label'] for r in selected) != {'drowning': 27, 'normal': 65}:
        raise ValueError('Expected exactly the previously approved 27 drowning and 65 normal videos')
    if len({r['video_id'] for r in selected}) != 92 or len({r['sha256'] for r in selected}) != 92:
        raise ValueError('The fixed cohort contains duplicate IDs/content')
    result = []
    for row in selected:
        group = groups[row['video_id']]
        if any(row[k] != group[k] for k in ('sha256', 'class_label', 'relation_group')):
            raise ValueError('Stage 1 and Stage 2A metadata disagree')
        result.append({**row, **{k: group[k] for k in ('leakage_group', 'confirmed_source_group',
                                                     'source_id', 'session_id')}})
    return result


def experimental_split(videos):
    """80/20 target by video/class counts, with all existing constraints retained."""
    labels = [r['class_label'] for r in videos]
    groups = [r['leakage_group'] for r in videos]
    totals = Counter(labels)
    best = None
    for train, validation in GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=42).split(videos, labels, groups):
        assignments = [{'video_id': videos[int(i)]['video_id'], 'split': split}
                       for indices, split in ((train, 'train'), (validation, 'validation')) for i in indices]
        validate_partition(videos, assignments)
        counts = {split: Counter(videos[int(i)]['class_label'] for i in indices)
                  for indices, split in ((train, 'train'), (validation, 'validation'))}
        if not all(counts[s][c] for s in counts for c in totals):
            continue
        error = abs(len(validation) / len(videos) - 0.2) + sum(
            abs(counts['validation'][c] / totals[c] - 0.2) for c in totals)
        if best is None or error < best[0]:
            best = error, assignments
    if best is None:
        raise ValueError('No class-preserving grouped draft found; do not split groups to force a ratio')
    return {r['video_id']: r['split'] for r in best[1]}


def build_clips(videos, assignments, root):
    result, prepared_videos = [], []
    for row in videos:
        path, label = annotated_path(root, row['yolo_output_path'])
        if label != int(row['class_label'] == 'drowning'):
            raise ValueError('YOLO folder label disagrees with the approved manifest')
        info = probe_video(path)
        if (info.frame_count != int(row['frame_count'])
                or abs(info.duration_sec - float(row['duration_sec'])) > max(0.5, 2 / info.fps)):
            raise ValueError('YOLO stream metadata no longer matches the approved cohort: ' + str(path))
        digest = sha256_file(path)
        records = build_clip_records({'video_id': row['video_id'], 'path': row['yolo_output_path'],
                                      'style': row['class_label'], 'split': assignments[row['video_id']]},
                                     info, sampling_fps=8, clip_length=16, stride=16)
        if not records:
            raise ValueError('Approved video has no complete 16-frame clip: ' + str(path))
        shared = {k: row[k] for k in ('video_id', 'class_label', 'sha256', 'relation_group',
                                      'leakage_group', 'source_id', 'session_id', 'confirmed_source_group')}
        shared.update(source_path=row['yolo_output_path'], yolo_sha256=digest,
                      video_label=label, label_strength='weak_video_label', clip_event_label='unknown',
                      draft_split=assignments[row['video_id']], split_status='experimental_draft',
                      source_fps=info.fps, source_frame_count=info.frame_count)
        for record in records:
            result.append({**shared, **{k: v for k, v in asdict(record).items()
                                        if k not in {'style', 'split'}},
                           'effective_sampling_fps': info.fps / record.frame_step})
        prepared_videos.append({**shared, 'duration_sec': info.duration_sec, 'width': info.width,
                                'height': info.height, 'clip_count': len(records),
                                'frame_step': records[0].frame_step,
                                'effective_sampling_fps': info.fps / records[0].frame_step})
        print(f"Manifest [{len(prepared_videos)}/{len(videos)}]: {path.name}, {len(records)} clips", flush=True)
    validate_manifest(root, result)
    return result, prepared_videos


def prepare(root=ROOT):
    root = Path(root).resolve()
    output = root / 'results' / OUTPUT_NAME
    if output.exists():
        raise FileExistsError('Refusing to change existing output: ' + str(output))
    previous = root / 'results/aquasafe_binary_v1'
    inputs = [p for p in previous.iterdir() if p.is_file()]
    input_hashes = {p.name: sha256_file(p) for p in inputs}
    selected = select_videos(read_csv(previous / 'dataset_manifest.csv'),
                             read_csv(previous / 'source_group_analysis.csv'))
    media_before = {r['yolo_output_path']: (annotated_path(root, r['yolo_output_path'])[0].stat().st_size,
                                          annotated_path(root, r['yolo_output_path'])[0].stat().st_mtime_ns)
                    for r in selected}
    assignments = experimental_split(selected)
    clips, videos = build_clips(selected, assignments, root)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / 'clip_manifest.csv'
    write_csv(manifest, clips, list(clips[0]))
    write_csv(output / 'video_manifest.csv', videos, list(videos[0]))
    dataset = YoloClipDataset(root, manifest)
    indices = {}
    for index, row in enumerate(dataset.rows):
        indices.setdefault(row['video_id'], []).append(index)
    checks = []
    for video_number, (video_id, items) in enumerate(indices.items(), 1):
        for index in sorted({items[0], items[-1]}):
            tensor, target, metadata = dataset[index]
            checks.append({'clip_id': metadata['clip_id'], 'shape': list(tensor.shape),
                           'dtype': str(tensor.dtype), 'video_label': target})
        print(f'Decode [{video_number}/{len(videos)}]: {video_id}', flush=True)
    loaders = make_loaders(root, manifest, batch_size=2, num_workers=0)
    batch_shapes = {}
    for split, loader in loaders.items():
        tensors, targets, metadata = next(iter(loader))
        batch_shapes[split] = {'shape': list(tensors.shape), 'target_dtype': str(targets.dtype),
                               'target_shape': list(targets.shape), 'video_ids': metadata['video_id']}
    if input_hashes != {p.name: sha256_file(p) for p in inputs}:
        raise RuntimeError('Previous reports changed during preparation')
    if media_before != {r['yolo_output_path']: (annotated_path(root, r['yolo_output_path'])[0].stat().st_size,
                                               annotated_path(root, r['yolo_output_path'])[0].stat().st_mtime_ns)
                         for r in selected}:
        raise RuntimeError('YOLO inputs changed during preparation')
    counts = {split: {'videos': sum(v['draft_split'] == split for v in videos),
                      'video_class_counts': dict(Counter(v['class_label'] for v in videos if v['draft_split'] == split)),
                      'clips': sum(c['draft_split'] == split for c in clips),
                      'clip_video_label_counts': dict(Counter(c['class_label'] for c in clips if c['draft_split'] == split)),
                      'precautionary_groups': len({v['leakage_group'] for v in videos if v['draft_split'] == split})}
              for split in ('train', 'validation')}
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_report_sha256': input_hashes,
        'input_video_root': 'results/detect_all', 'raw_videos_loaded': False,
        'videos': len(videos), 'video_class_counts': dict(Counter(v['class_label'] for v in videos)),
        'clips': len(clips), 'class_to_index': {'normal': 0, 'drowning': 1},
        'label_strength': 'weak_video_label', 'confirmed_clip_event_labels': 0,
        'sampling': {'requested_fps': 8, 'frame_step': 'max(1, round(source_fps / 8))',
                     'clip_length': 16, 'stride_in_sampled_frames': 16, 'overlap': False,
                     'incomplete_tail_policy': 'discard; no repetition or padding'},
        'tensor': {'shape': [3, 16, 112, 112], 'dtype': 'float32', 'color': 'RGB',
                   'resize_short_side': 128, 'crop': 'center',
                   'mean': [0.43216, 0.394666, 0.37645], 'std': [0.22803, 0.22145, 0.216989],
                   'materialization': 'Lazy decode and normalization in Dataset; no video/tensor copies saved.'},
        'draft_split': {'target_ratios': {'train': 0.8, 'validation': 0.2}, 'seed': 42,
                        'method': 'Best video/class ratio error among 100 GroupShuffleSplit candidates; full Stage 2A leakage groups retained.',
                        'counts': counts, 'test_set_created': False, 'independent_evaluation': False},
        'leakage_checks': {'status': 'passed', 'video_id': True, 'original_content_sha256': True,
                           'yolo_content_sha256': True, 'paths': True, 'relation_group': True,
                           'leakage_group': True, 'known_source': True, 'unique_clip_ids': True},
        'hash_definitions': {'sha256': 'Original content identity from Stage 1 metadata; raw videos are not opened.',
                             'yolo_sha256': 'Full SHA256 of the actual annotated AVI input.'},
        'decode_verification': {'first_and_last_clip_per_video': checks, 'dataloader_batches': batch_shapes,
                                'all_clips_decoded': False, 'remaining_clips': 'Decoded on access; failures raise errors.'},
        'limitations': ['All 92 source/session identities remain unknown. Precautionary groups are not confirmed independent sources.',
                       'Video targets are weak folder labels; a positive video target does not confirm drowning in every clip. Keep video IDs for later video-bag/MIL training.',
                       'Saved YOLO overlays are used as requested. Box presence in every frame is not machine-verified; overlays can become visual shortcuts.',
                       'Video durations produce unequal clip counts. Clip counts must not be treated as independent video/event counts.',
                       'Grouping prevents known/suspected cross-partition relations, but cannot guarantee absence of unrecognized common sources.'],
        'dataloader_infrastructure_ready': True, 'ready_for_independent_evaluation': False,
        'training_started': False, 'yolo_started': False, 'existing_files_modified': False,
    }
    with (output / 'preparation_report.json').open('x', encoding='utf-8') as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
    text = f'''# Aqua-Safe — YOLO videolarından R3D-18 veri hazırlığı

Yalnız results/detect_all altındaki daha önce seçilmiş **92 işaretlenmiş video**
kullanıldı: 27 drowning (1), 65 normal (0). Ham videolar açılmadı.
**{len(clips)} klip** manifestte kayıtlı; tensörler Dataset erişiminde üretilir.
Yaklaşık 8 FPS, 16 kare, örtüşmesiz klipler; kısa son bölüm atılır.
Örnek çıktı float32 RGB [3,16,112,112], batch [B,3,16,112,112].
Kısa kenar 128, merkez kırpma 112 ve mevcut R3D normalizasyonu kullanılır.

## Deneysel ayrım taslağı

Hedef %80 train / %20 validation; test yok. Grup bütünlüğü oranlardan önceliklidir.
Train: {counts['train']['videos']} video, {counts['train']['video_class_counts']}, {counts['train']['clips']} klip.
Validation: {counts['validation']['videos']} video, {counts['validation']['video_class_counts']}, {counts['validation']['clips']} klip.
Aşama 2A ihtiyati grup kimlikleri, dışarıda kalan videoların bağlantıları dahil,
korunur. Bütün klipler üst videonun taslak ayrımını miras alır.
Kaynak/oturumlar bilinmiyor; bu ayrım bağımsız doğrulama değildir.

## Kullanım

```python
from pathlib import Path
from src.video_classification.yolo_dataset import make_loaders

root = Path("{root.as_posix()}")
manifest = root / "results/{OUTPUT_NAME}/clip_manifest.csv"
loaders = make_loaders(root, manifest, batch_size=2, num_workers=0)
x, video_targets, metadata = next(iter(loaders["train"]))
# x: [B,3,16,112,112]; video_targets: int64 [B]
# metadata['video_id'] üst videoyu, metadata['leakage_group'] ihtiyati grubu korur.
```

video_targets zayıf video etiketidir; doğrulanmış klip olayı değildir.
clip_event_label unknown kalır. Sonraki eğitimde video-bag/MIL için video kimlikleri
korunmalıdır; aynı videonun çok sayıda klibi bağımsız örnek sayılmamalıdır.

## Doğrulama ve sınırlar

92 AVI dosyasının tam hash'i ve metadata bilgileri alındı. Her videonun ilk ve son
klibi çözümlenip tensör boyutu/sonluluğu denetlendi ({len(checks)} örnek).
Train ve validation DataLoader batch'leri okundu. Bütün klipler önceden
çözümlenmedi; okuma hataları erişim sırasında açık hata üretir.
Önceki raporların hash'leri ve giriş videolarının boyut/değiştirilme zamanları
hazırlık sonunda kontrol edildi. Mevcut dosyalar değiştirilmedi.

YOLO kutuları model için görsel bir kestirme ipucu olabilir. Her karede kutu varlığı
yeniden doğrulanmadı. Bilinmeyen kaynak ilişkileri nedeniyle veri sızıntısının
tamamen giderildiği iddia edilmez. Etiketler olay düzeyinde doğrulanmış değildir.
Eğitim veya YOLO çalıştırılmadı; paket kurulmadı, ağırlıklar yüklenmedi.
'''
    with (output / 'preparation_report.md').open('x', encoding='utf-8') as target:
        target.write(text)
    print(json.dumps({'videos': len(videos), 'clips': len(clips), 'draft_counts': counts}, indent=2))
    return report


if __name__ == '__main__':
    prepare()
