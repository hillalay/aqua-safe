from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .reviews import export_training_metadata, validate_review


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate saved reviews and export R3D-ready metadata.")
    parser.add_argument("--metadata", type=Path, default=ROOT / "results" / "video_review" / "master_metadata.csv")
    parser.add_argument("--reviews", type=Path, default=ROOT / "results" / "video_review" / "review_state.json")
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8-sig", newline="") as source:
        master = list(csv.DictReader(source))
    raw_state = json.loads(args.reviews.read_text(encoding="utf-8"))
    known = {row["video_id"] for row in master}
    state = {video_id: validate_review({"video_id": video_id, **review}, known) for video_id, review in raw_state.items()}
    report = export_training_metadata(master, state, args.metadata.parent)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
