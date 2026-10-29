from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from pose_pipeline.extractors import MediaPipeExtractor, YoloPoseExtractor
from pose_pipeline.temporal import save_windows, validate_split_groups
from pose_pipeline.video import process_video


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pilot pose comparison on explicitly selected videos.")
    parser.add_argument("--video-id", action="append", required=True, help="Metadata video_id; pass once or twice for a pilot")
    parser.add_argument("--models", nargs="+", choices=("mediapipe", "yolo"), default=("mediapipe", "yolo"))
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--mediapipe-model", type=Path, default=ROOT / "models" / "pose_landmarker_full.task")
    parser.add_argument("--yolo-weights", type=Path, default=ROOT / "models" / "yolov8n-pose.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.video_id) > 2:
        raise SystemExit("Pilot mode accepts at most two --video-id values")
    with (ROOT / "results" / "single_metadata.csv").open(encoding="utf-8-sig", newline="") as source:
        metadata = list(csv.DictReader(source))
    validate_split_groups(metadata)
    selected = {row["video_id"]: row for row in metadata if row["video_id"] in args.video_id}
    missing = set(args.video_id) - set(selected)
    if missing:
        raise SystemExit(f"Unknown video_id: {', '.join(sorted(missing))}")

    for video_id in args.video_id:
        row = selected[video_id]
        video_path = ROOT / Path(row["path"])
        for model_name in args.models:
            extractor = (MediaPipeExtractor(args.mediapipe_model) if model_name == "mediapipe"
                         else YoloPoseExtractor(args.yolo_weights, "0" if torch.cuda.is_available() else "cpu"))
            output_dir = ROOT / "results" / "pose_comparison" / video_id / extractor.name
            with extractor:
                records, metrics = process_video(video_path, video_id, extractor, output_dir, args.target_fps)
            save_windows(records, output_dir / "gru_windows.npz", row["style"], row["split"])
            manifest = {
                "video": str(video_path.relative_to(ROOT)), "style": row["style"],
                "style_verified": row["style_verified"], "split": row["split"],
                "model": extractor.name, "library_version": extractor.version,
                "weights": str((args.mediapipe_model if "mediapipe" in extractor.name else args.yolo_weights).relative_to(ROOT)),
                "target_fps": args.target_fps, "metrics": metrics,
            }
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Saved {video_id} / {extractor.name}: {output_dir}")


if __name__ == "__main__":
    main()
