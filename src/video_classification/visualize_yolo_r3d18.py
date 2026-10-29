"""CUDA validation inference and timestamp-aligned overlays; never trains a model."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path

import av
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.video_review.inventory import sha256_file
from .evaluate import classification_metrics
from .model import build_r3d18
from .preprocessing import probe_video
from .prepare_binary_v1 import write_csv
from .yolo_dataset import YoloClipDataset, annotated_path, read_csv


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['normal', 'drowning']
EXPECTED = {'accuracy': 0.7847, 'macro_f1': 0.6183, 'drowning_f1': 0.3662,
            'drowning_recall': 0.2955, 'confusion_matrix': [[151, 14], [31, 13]]}
LIMITATIONS = [
    'Targets are weak_video_label and clip_event_label=unknown. TP/TN/FP/FN compare predictions with parent-video labels, not verified drowning events.',
    'Source/session identities remain unknown. Existing precautionary groups are retained; this is experimental validation, not an independent test.',
    'Softmax scores are uncalibrated class probabilities, not real-world drowning risk.',
    'The checkpoint was selected on this validation set; these are development results.',
    'Only annotated YOLO AVI inputs are read. YOLO is not run again. Rendered MP4s contain no audio.',
]


def error_group(target, prediction):
    if target not in (0, 1) or prediction not in (0, 1):
        raise ValueError('Binary labels must be 0 or 1')
    return ('TP' if prediction else 'FN') if target else ('FP' if prediction else 'TN')


def summarize_predictions(rows):
    if not rows or len({r['clip_id'] for r in rows}) != len(rows):
        raise ValueError('Predictions must contain each validation clip exactly once')
    labels = [int(r['video_label']) for r in rows]
    predictions = [int(r['predicted_label']) for r in rows]
    metrics = classification_metrics(labels, predictions, 2)
    metrics.update(drowning_f1=metrics['f1'][1], drowning_recall=metrics['recall'][1],
                   clips=len(rows), videos=len({r['video_id'] for r in rows}),
                   class_to_index={'normal': 0, 'drowning': 1},
                   confusion_matrix_axes={'rows': 'weak parent-video target', 'columns': 'model prediction'},
                   decision_rule='argmax of two softmax scores; ties select normal (index 0)',
                   weak_label_groups={name: {'count': sum(r['error_group'] == name for r in rows),
                                             'clip_ids': [r['clip_id'] for r in rows if r['error_group'] == name]}
                                      for name in ('TP', 'TN', 'FP', 'FN')},
                   limitations=LIMITATIONS, independent_test=False, training_started=False)
    metrics['user_reported_reference'] = EXPECTED
    metrics['matches_user_reported_reference'] = (
        metrics['confusion_matrix'] == EXPECTED['confusion_matrix']
        and all(round(metrics[k], 4) == EXPECTED[k]
                for k in ('accuracy', 'macro_f1', 'drowning_f1', 'drowning_recall')))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['video_id']].append(row)
    metrics['per_video_weak_label_counts'] = [
        {'video_id': key, 'source_path': bundle[0]['source_path'],
         'weak_video_label': int(bundle[0]['video_label']),
         **{name: sum(r['error_group'] == name for r in bundle) for name in ('TP', 'TN', 'FP', 'FN')}}
        for key, bundle in grouped.items()]
    return metrics


def infer(model, loader, device):
    """YoloClipDataset uses the existing decode_clip and transform_clip unchanged."""
    rows, offset = [], 0
    model.eval()
    with torch.inference_mode():
        for tensors, targets, metadata in tqdm(loader, desc='CUDA validation inference', unit='batch'):
            logits = model(tensors.to(device))
            if logits.shape != (len(targets), 2) or not torch.isfinite(logits).all():
                raise ValueError('Model returned invalid binary logits')
            probabilities = logits.softmax(dim=1).cpu()
            for index, scores in enumerate(probabilities):
                parent = dict(loader.dataset.rows[offset])
                if parent['clip_id'] != metadata['clip_id'][index]:
                    raise ValueError('Inference order does not match manifest; shuffle must be False')
                target, prediction = int(targets[index]), int(scores.argmax())
                rows.append({**parent, 'predicted_label': prediction,
                             'predicted_class': CLASSES[prediction],
                             'normal_probability': float(scores[0]),
                             'drowning_probability': float(scores[1]),
                             'error_group': error_group(target, prediction)})
                offset += 1
    if offset != len(loader.dataset):
        raise ValueError('Some validation clips were not evaluated')
    return rows


def prediction_at_frame(rows, starts, frame_index):
    index = bisect_right(starts, frame_index) - 1
    if index < 0 or frame_index > int(rows[index]['end_frame']):
        return None
    return rows[index]


def overlay_lines(prediction, seconds):
    lines = [f'VIDEO TIME {seconds:.3f}s | MODEL PREDICTION']
    if prediction is None:
        return lines + ['NO CLIP / NO PREDICTION']
    return lines + [
        f"PRED: {prediction['predicted_class']} | P(normal)={float(prediction['normal_probability']):.4f} | P(drowning)={float(prediction['drowning_probability']):.4f}",
        f"Clip: {float(prediction['start_time']):.3f}s - {float(prediction['end_time']):.3f}s | uncalibrated softmax",
    ]


def render_video(root, predictions, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite rendered video: ' + str(destination))
    if not predictions or len({r['video_id'] for r in predictions}) != 1:
        raise ValueError('Rendering requires predictions for exactly one video')
    rows = sorted(predictions, key=lambda r: int(r['start_frame']))
    source, _ = annotated_path(root, rows[0]['source_path'])
    if any(r['source_path'] != rows[0]['source_path'] for r in rows):
        raise ValueError('Mixed source paths')
    info = probe_video(source)
    starts = [int(r['start_frame']) for r in rows]
    for i, row in enumerate(rows):
        if (int(row['start_frame']) < 0 or int(row['end_frame']) >= info.frame_count
                or (i and int(rows[i - 1]['end_frame']) >= starts[i])):
            raise ValueError('Overlapping or out-of-range prediction intervals')
        if (abs(float(row['start_time']) - starts[i] / info.fps) > 1e-6
                or abs(float(row['end_time']) - int(row['end_frame']) / info.fps) > 1e-6):
            raise ValueError('Manifest times do not match source FPS')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial.mp4')
    if temporary.exists():
        raise FileExistsError('Partial render already exists: ' + str(temporary))
    fps = Fraction(str(info.fps)).limit_denominator(100000)
    cap, count = cv2.VideoCapture(str(source)), 0
    try:
        with av.open(str(temporary), 'w') as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width, stream.height, stream.pix_fmt = info.width, info.height, 'yuv420p'
            stream.options = {'crf': '23', 'preset': 'fast'}
            with tqdm(total=info.frame_count, desc='MP4 ' + source.stem, unit='frame') as progress:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    prediction = prediction_at_frame(rows, starts, count)
                    color = (0, 0, 255) if prediction is not None and int(prediction['predicted_label']) else (40, 220, 40)
                    if prediction is None:
                        color = (180, 180, 180)
                    lines = overlay_lines(prediction, count / info.fps)
                    scale = max(0.35, min(info.width / 1600, 1.0))
                    line_height = max(20, round(34 * scale))
                    cv2.rectangle(frame, (0, 0), (info.width, line_height * (len(lines) + 1)), (0, 0, 0), -1)
                    for index, line in enumerate(lines, 1):
                        cv2.putText(frame, line, (8, line_height * index), cv2.FONT_HERSHEY_SIMPLEX,
                                    scale, color, max(1, round(2 * scale)), cv2.LINE_AA)
                    video_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
                    video_frame.pts, video_frame.time_base = count, 1 / fps
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                    count += 1
                    progress.update()
            if count != info.frame_count:
                raise RuntimeError(f'Incomplete source decode: {source}; expected {info.frame_count}, read {count}')
            for packet in stream.encode():
                container.mux(packet)
    finally:
        cap.release()
    output_info = probe_video(temporary)
    if output_info.frame_count != count or abs(output_info.duration_sec - info.duration_sec) > 2 / info.fps:
        raise RuntimeError('Rendered MP4 timing/frame-count validation failed: ' + str(temporary))
    temporary.rename(destination)
    return {'source_path': rows[0]['source_path'], 'output': str(destination),
            'video_id': rows[0]['video_id'], 'frames': count, 'fps': info.fps,
            'duration_sec': output_info.duration_sec, 'codec': 'H264', 'audio_preserved': False}


def plot_confusion(metrics, destination):
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = metrics['confusion_matrix']
    ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                    color='white' if matrix[i][j] > max(map(max, matrix)) / 2 else 'black', fontsize=20)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Model prediction', ylabel='Weak parent-video label',
           title=f"Experimental validation: {metrics['clips']} clips\nNot verified drowning-event ground truth")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def validate_cached_predictions(predictions, manifest_rows):
    expected = {r['clip_id']: r for r in manifest_rows}
    if Counter(r['clip_id'] for r in predictions) != Counter({key: 1 for key in expected}):
        raise ValueError('Cached predictions must cover each validation clip exactly once')
    for row in predictions:
        if any(str(row.get(k)) != str(v) for k, v in expected[row['clip_id']].items()):
            raise ValueError('Cached prediction metadata disagrees with current manifest: ' + row['clip_id'])
        scores = [float(row['normal_probability']), float(row['drowning_probability'])]
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in scores) or abs(sum(scores) - 1) > 1e-5:
            raise ValueError('Invalid cached class probabilities: ' + row['clip_id'])
        predicted = int(scores[1] > scores[0])
        if (int(row['predicted_label']) != predicted or row['predicted_class'] != CLASSES[predicted]
                or row['error_group'] != error_group(int(row['video_label']), predicted)):
            raise ValueError('Inconsistent cached prediction: ' + row['clip_id'])


def verify_render(root, rows, destination):
    """Decode every output frame, checking codec, dimensions, count and timestamps."""
    info = probe_video(annotated_path(root, rows[0]['source_path'])[0])
    count, last_time = 0, None
    with av.open(str(destination)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != 'h264':
            raise ValueError('Expected an H.264 MP4: ' + str(destination))
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (info.width, info.height):
                raise ValueError('Output dimensions differ from source: ' + str(destination))
            if frame.time is None or abs(frame.time - count / info.fps) > 2 / info.fps:
                raise ValueError('Output timestamps differ from source: ' + str(destination))
            if last_time is not None and frame.time <= last_time:
                raise ValueError('Non-increasing output timestamps: ' + str(destination))
            count, last_time = count + 1, frame.time
    if count != info.frame_count:
        raise ValueError(f'Incomplete MP4 {destination}: expected {info.frame_count}, decoded {count}')
    return {'output': str(destination), 'source_path': rows[0]['source_path'],
            'video_id': rows[0]['video_id'], 'weak_video_label': int(rows[0]['video_label']),
            'decoded_frames': count, 'fps': info.fps, 'playability_check': 'full_decode_passed'}


def render_all_validation(root=ROOT):
    root = Path(root).resolve()
    base = root / 'results/aquasafe_yolo_r3d18_v1'
    output = base / 'visualization'
    csv_path, metrics_path = output / 'predictions.csv', output / 'metrics.json'
    if not csv_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError('--all-videos requires existing predictions.csv and metrics.json; it never reruns inference')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if sha256_file(csv_path) != metrics['prediction_csv_sha256']:
        raise ValueError('Cached predictions.csv hash does not match metrics.json')
    protected = {root / name: digest for name, digest in metrics['input_sha256'].items()}
    # A later training-code edit does not change cached predictions from a fixed checkpoint.
    training_script = root / 'src/video_classification/train_yolo_r3d18.py'
    training_script_changed = False
    if training_script in protected:
        current_hash = sha256_file(training_script)
        training_script_changed = current_hash != protected[training_script]
        protected[training_script] = current_hash
    protected.update({csv_path: sha256_file(csv_path), metrics_path: sha256_file(metrics_path)})
    for path, digest in protected.items():
        if not path.resolve().is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError('Cached inference input changed: ' + str(path))
    dataset = YoloClipDataset(root, base / 'clip_manifest.csv', split='validation')
    predictions = read_csv(csv_path)
    validate_cached_predictions(predictions, dataset.rows)
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row['video_id']].append(row)
    destinations = {}
    sources = {}
    for video_id, rows in grouped.items():
        row = rows[0]
        sources[row['source_path']] = row['yolo_sha256']
        label = CLASSES[int(row['video_label'])]
        destinations[video_id] = output / 'videos' / (label + '_' + Path(row['source_path']).stem + '.mp4')
    if len(set(destinations.values())) != len(destinations):
        raise ValueError('Output filename collision between distinct source videos')
    for relative, digest in tqdm(sources.items(), desc='Verify cached AVI inputs', unit='video'):
        source, _ = annotated_path(root, relative)
        if digest != metrics['source_yolo_sha256'].get(relative) or sha256_file(source) != digest:
            raise ValueError('Source AVI no longer matches cached inference: ' + relative)
    existing = {path: sha256_file(path) for path in destinations.values() if path.exists()}
    report = {'expected_videos': len(grouped), 'cached_clips': len(predictions),
              'inference_rerun': False, 'created': 0, 'reused': 0, 'failed': [], 'videos': [],
              'label_strength': 'weak_video_label', 'independent_test': False,
              'training_script_changed_since_inference': training_script_changed,
              'complete': False}
    report_path = output / 'all_videos_report.json'
    for video_id, rows in tqdm(sorted(grouped.items(), key=lambda item: item[1][0]['source_path']),
                               desc='All validation videos', unit='video'):
        destination = destinations[video_id]
        action = 'reused' if destination in existing else 'created'
        try:
            if action == 'created':
                render_video(root, rows, destination)
            verified = verify_render(root, rows, destination)
            verified['action'] = action
            report['videos'].append(verified)
            report[action] += 1
        except Exception as error:
            report['failed'].append({'source_path': rows[0]['source_path'],
                                     'output': str(destination), 'error': str(error)})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for path, digest in {**protected, **existing}.items():
        if sha256_file(path) != digest:
            raise RuntimeError('Protected input or existing MP4 changed: ' + str(path))
    for relative, digest in tqdm(sources.items(), desc='Verify AVI unchanged', unit='video'):
        if sha256_file(annotated_path(root, relative)[0]) != digest:
            raise RuntimeError('Source AVI changed: ' + relative)
    report.update(complete=len(report['videos']) == len(grouped), protected_inputs_unchanged=True,
                  existing_mp4_unchanged=True, missing_outputs=[str(p) for p in destinations.values() if not p.is_file()])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'videos'}, ensure_ascii=False, indent=2))
    if not report['complete']:
        raise RuntimeError('Some videos failed; see ' + str(report_path))
    return report


def run(root=ROOT, all_videos=False):
    if all_videos:
        return render_all_validation(root)
    root = Path(root).resolve()
    base = root / 'results/aquasafe_yolo_r3d18_v1'
    output, checkpoint, manifest = base / 'visualization', base / 'training/best_model.pt', base / 'clip_manifest.csv'
    if output.exists():
        raise FileExistsError('Visualization directory already exists; existing predictions are not overwritten or recomputed: ' + str(output))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Activate the project .venv and check the NVIDIA driver/PyTorch CUDA installation.')
    if not checkpoint.is_file():
        raise FileNotFoundError('Trained checkpoint not found: ' + str(checkpoint))
    av.codec.Codec('libx264', 'w')
    dataset = YoloClipDataset(root, manifest, split='validation')
    if len(dataset) != 209 or Counter(r['video_label'] for r in dataset.rows) != {'0': 165, '1': 44}:
        raise ValueError('Validation cohort differs from the verified 209 clips (165 normal, 44 drowning)')
    protected = [manifest, checkpoint, base / 'training/history.json',
                 root / 'src/video_classification/train_yolo_r3d18.py',
                 root / 'src/video_classification/model.py', root / 'src/video_classification/yolo_dataset.py',
                 root / 'src/video_classification/preprocessing.py',
                 root / 'results/aquasafe_binary_v1/source_group_analysis.csv']
    before = {str(p.relative_to(root)): sha256_file(p) for p in protected}
    sources = {r['source_path']: r['yolo_sha256'] for r in dataset.rows}
    for relative, digest in tqdm(sources.items(), desc='Verify validation AVI hashes', unit='video'):
        if sha256_file(annotated_path(root, relative)[0]) != digest:
            raise ValueError('Annotated source changed since preparation: ' + relative)
    model = build_r3d18(num_classes=2, pretrained=False).to('cuda')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    predictions = infer(model, loader, torch.device('cuda'))
    metrics = summarize_predictions(predictions)
    metrics.update(device='cuda', gpu=torch.cuda.get_device_name(0), input_sha256=before,
                   source_yolo_sha256=sources, evaluation_level='clip',
                   prediction_intervals='Inclusive manifest start_frame/end_frame. Gaps and trailing uncovered frames have no prediction.',
                   model_initialization='build_r3d18(num_classes=2, pretrained=False); strict state_dict load; eval; float32; inference_mode',
                   visualization_selection='One weak normal video with most FP clips and one weak drowning video with most FN clips; ties by source path. Selection is error-focused, not representative.',
                   rendered_videos=[], visualization_complete=False)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'predictions.csv', predictions, list(predictions[0]))
    errors = [r for r in predictions if r['error_group'] in {'FP', 'FN'}]
    write_csv(output / 'errors.csv', errors, list(predictions[0]))
    plot_confusion(metrics, output / 'confusion_matrix.png')
    def save_metrics():
        (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    save_metrics()
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row['video_id']].append(row)
    for label, error in ((0, 'FP'), (1, 'FN')):
        candidates = [bundle for bundle in grouped.values() if int(bundle[0]['video_label']) == label]
        chosen = min(candidates, key=lambda bundle: (-sum(r['error_group'] == error for r in bundle), bundle[0]['source_path']))
        name = ('normal_' if label == 0 else 'drowning_') + Path(chosen[0]['source_path']).stem + '.mp4'
        rendered = render_video(root, chosen, output / 'videos' / name)
        metrics['rendered_videos'].append(rendered)
        save_metrics()
    if before != {str(p.relative_to(root)): sha256_file(p) for p in protected}:
        raise RuntimeError('A protected input changed while inference/rendering was running')
    for relative, digest in tqdm(sources.items(), desc='Verify AVI unchanged', unit='video'):
        if sha256_file(annotated_path(root, relative)[0]) != digest:
            raise RuntimeError('Source AVI changed during rendering: ' + relative)
    metrics['protected_inputs_unchanged'] = True
    metrics['visualization_complete'] = True
    metrics['prediction_csv_sha256'] = sha256_file(output / 'predictions.csv')
    save_metrics()
    print(json.dumps({k: metrics[k] for k in ('accuracy', 'macro_f1', 'drowning_f1', 'drowning_recall',
                                             'confusion_matrix', 'matches_user_reported_reference', 'rendered_videos')}, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-videos', action='store_true',
                        help='Render every validation video using verified predictions.csv; reuse valid MP4s without inference')
    args = parser.parse_args()
    try:
        run(all_videos=args.all_videos)
    except Exception as error:
        raise SystemExit(f'Visualization failed: {error}. Any completed CSVs remain in visualization/; no input is overwritten.') from error
