# Aqua-Safe pose comparison

Kişi takibi ve kutulu MP4 üretimi: [Kullanım kılavuzu](README-demo.md).

This pilot pipeline compares MediaPipe Pose Landmarker (33 native landmarks) with
Ultralytics YOLOv8-Pose (17 COCO landmarks). Native outputs stay separate. A
13-joint shared subset is also emitted for comparison and later GRU input.

## Setup

From the repository root, using Python 3.10:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-pose.txt
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" `
  -OutFile "models\pose_landmarker_full.task"
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'models\yolov8n-pose.pt')"
```

The MediaPipe model URL is the official Google-hosted `full/float16/latest`
bundle. Keep the downloaded file unchanged; each run records its path and the
installed library version. Ultralytics also downloads official weights on first
use if `YOLO("yolov8n-pose.pt")` is used; this project keeps the selected weight
under `models/` so the run is explicit.

Official references: [MediaPipe Pose Landmarker Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker),
[Ultralytics pose documentation](https://docs.ultralytics.com/tasks/pose/).

## Pilot run

The CLI deliberately requires one or two video IDs, preventing an accidental
full-dataset run before visual review:

```powershell
.\.venv\Scripts\python.exe src\run_pose_comparison.py --video-id V001
```

Run one backend while setting it up:

```powershell
.\.venv\Scripts\python.exe src\run_pose_comparison.py --video-id V001 --models yolo
```

Each model writes to `results/pose_comparison/<video_id>/<model>/`:

- `annotated.mp4`: 10 FPS sampled frames with the shared skeleton overlaid.
- `frames.jsonl`: one record per sample, including frame/time/model, untouched
  native landmarks, shared landmarks, and explicit `not_detected` frames.
- `metrics.json`: detection/missing rates, joint quality, normalized inter-frame
  motion, elapsed time, and processing FPS.
- `gru_windows.npz`: non-overlapping 30-step windows containing
  `features[..., (x,y,quality)]`, a boolean `mask`, timestamps, style, split and
  video ID.
- `manifest.json`: input, model/library/weights, sampling setting and metrics.

Coordinates are normalized, so resizing inside either model keeps aspect-ratio
semantics and the annotated video retains the source dimensions. A landmark
below the 0.5 quality threshold is considered missing for metrics/drawing, but
its native model output is still preserved. No-pose frames contain no invented
coordinates.

`style`, `style_verified`, and `split` come from `results/single_metadata.csv`.
Known swimmer or session groups are rejected if they cross splits. When those
IDs are `unknown`, only video identity can be protected; the current metadata
therefore does **not** support a claim of person-independent evaluation.

The quality metrics are diagnostic proxies only. Without ground-truth joint
annotations they do not measure actual pose accuracy; review both annotated
videos before processing more data.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Pose diagnostics

The fixed 20-frame V001 and multi-swimmer parameter sweep writes its report,
CSV, raw/usable JSONL records, and contact sheets under
`results/pose_diagnostics/`:

```powershell
.\.venv\Scripts\python.exe src\diagnose_pose.py
```

Low-confidence "usable" counts are heuristic candidates, not accuracy; the
generated report records the required visual false-positive review.

## R3D-18 video classification pilot

The classifier validates metadata and group-safe splits before training. With
the current pending labels and unknown swimmer/session IDs it intentionally runs
only one real-video, pretrained-model smoke batch:

```powershell
.\.venv\Scripts\python.exe -m src.video_classification.train
```

Configuration is in `configs/r3d18_pilot.yaml`; inventory, clip manifest,
checkpoint and report are written under `results/r3d18_pilot/`. Multi-epoch
training starts only when verified rows contain non-empty `train` and
`validation` splits.

The verified 33-video experiment uses grouped, class-preserving video-level
cross-validation and never claims an independent test:

```powershell
.\.venv\Scripts\python.exe -m src.video_classification.experimental_cv --check-only
.\.venv\Scripts\python.exe -m src.video_classification.experimental_cv
```

Configuration is in `configs/r3d18_verified.yaml`; outputs are written under
`results/r3d18_verified/` and labeled `experimental_not_independent_test`.

## Binary drowning-behavior preparation

Prepare the folder-derived binary metadata and leakage constraints without
starting training:

```powershell
.\.venv\Scripts\python.exe -m src.video_classification.binary_drowning
```

The command preserves review fields, removes exact copies from the training
set, keeps visual/source candidates in one fold, and writes under
`results/r3d18_binary_drowning/`. Drowning-folder labels remain
`weak_video_label`; their clips are `unlabeled`. Model output must be presented
as `drowning_behavior_score`, not as a calibrated real-world risk probability.

Prepare leakage-checked video and clip manifests without starting training or
loading pretrained weights:

```powershell
.\.venv\Scripts\python.exe -m src.video_classification.mil_train --prepare-only
```

Start the five-fold CUDA run manually when ready:

```powershell
.\.venv\Scripts\python.exe -m src.video_classification.mil_train
```

Each video is one MIL bag. Training samples at most 8 clips per bag and
resamples them deterministically each epoch; validation uses at most 32 evenly
spaced clips. Clips are forwarded four at a time to bound GPU memory. Both
positive and negative bags use the mean of the top two drowning logits: for a
negative bag this penalizes its highest-risk clips. Bag batch size, clip limits,
top-k and clip-forward batch size are configurable in
`configs/r3d18_binary_drowning.yaml`.

Outputs go to `results/r3d18_binary_drowning/mil/fold_<n>/`: manifests,
`training_log.csv`, `metrics.json`, `video_predictions.csv`, and
`best_checkpoint.pt`. Results are weak-label experimental measurements, not
event-ground-truth performance or calibrated drowning risk.
