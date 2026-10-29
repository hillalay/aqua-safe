from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src.video_classification.preprocessing import probe_video


ROOT = Path(__file__).resolve().parents[2]
STYLE_CANDIDATES = {
    "kelebek": "butterfly", "kurbagalama": "breaststroke", "serbest": "freestyle",
    "sirtustu": "backstroke", "bogulma": "drowning_candidate",
}
REVIEW_FIELDS = ("style", "behavior", "swimmer_count", "mixed_styles", "review_status",
                 "source_id", "session_id", "notes")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(frame) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return f"{sum(int(bit) << index for index, bit in enumerate(bits.flat)):016x}"


def visual_hashes(path: Path, count: int = 5) -> list[str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hashes = []
    try:
        for index in range(count):
            frame_number = round((frame_count - 1) * (index + 0.5) / count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if ok:
                hashes.append(difference_hash(frame))
    finally:
        cap.release()
    return hashes


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def similarity_candidates(records: list[dict[str, object]], max_distance: int = 4) -> list[dict[str, object]]:
    candidates = []
    for index, left in enumerate(records):
        left_hashes = str(left["visual_hashes"]).split("|") if left["visual_hashes"] else []
        for right in records[index + 1:]:
            right_hashes = str(right["visual_hashes"]).split("|") if right["visual_hashes"] else []
            if not left_hashes or not right_hashes or (left["width"], left["height"]) != (right["width"], right["height"]):
                continue
            distance = min(hamming(a, b) for a in left_hashes for b in right_hashes)
            if distance <= max_distance:
                candidates.append({"video_id_a": left["video_id"], "video_id_b": right["video_id"],
                                   "min_visual_hash_distance": distance, "review_required": True})
    return candidates


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def stable_id(relative_path: str, legacy: dict[str, dict[str, str]], previous: dict[str, dict[str, str]]) -> str:
    if relative_path in previous:
        return previous[relative_path]["video_id"]
    key = relative_path.replace("data\\raw\\", "data\\single\\")
    if key in legacy:
        return legacy[key]["video_id"]
    return "R" + hashlib.sha1(relative_path.lower().encode()).hexdigest()[:10].upper()


def build_inventory(root: Path = ROOT) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = root / "results" / "video_review"
    master_path = output / "master_metadata.csv"
    legacy_rows = load_csv(root / "results" / "single_metadata.csv")
    legacy = {row["path"]: row for row in legacy_rows}
    previous = {row["path"]: row for row in load_csv(master_path)}
    records = []
    for path in sorted((root / "data" / "raw").glob("*/*.mp4")):
        relative = str(path.relative_to(root))
        info = probe_video(path)
        legacy_key = relative.replace("data\\raw\\", "data\\single\\")
        old = previous.get(relative, {})
        old_legacy = legacy.get(legacy_key, {})
        candidate = STYLE_CANDIDATES.get(path.parent.name, "unknown")
        legacy_style = STYLE_CANDIDATES.get(old_legacy.get("style", ""), "unknown")
        record = {
            "video_id": stable_id(relative, legacy, previous), "filename": path.name, "path": relative,
            "candidate_label": candidate, "fps": round(info.fps, 4), "frame_count": info.frame_count,
            "duration_sec": round(info.duration_sec, 4), "width": info.width, "height": info.height,
            "sha256": sha256_file(path), "visual_hashes": "|".join(visual_hashes(path)),
            "legacy_video_id": old_legacy.get("video_id", ""),
            "style": old.get("style", legacy_style if old_legacy.get("style_verified") == "verified" else "unknown"),
            "behavior": old.get("behavior", "unknown"), "swimmer_count": old.get("swimmer_count", "unknown"),
            "mixed_styles": old.get("mixed_styles", "unknown"),
            "review_status": old.get("review_status", "verified" if old_legacy.get("style_verified") == "verified" else "unreviewed"),
            "video_style_verification": old.get("video_style_verification", "unknown"),
            "person_style_verification": old.get("person_style_verification", "unknown"),
            "source_id": old.get("source_id", "unknown"), "session_id": old.get("session_id", old_legacy.get("session_id", "unknown")),
            "notes": old.get("notes", ""), "split": old.get("split", "unassigned"),
        }
        records.append(record)
    candidates = similarity_candidates(records)
    return records, candidates


def write_outputs(records: list[dict[str, object]], candidates: list[dict[str, object]], root: Path = ROOT) -> None:
    output = root / "results" / "video_review"
    output.mkdir(parents=True, exist_ok=True)
    state = output / "review_state.json"
    reviews = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
    merged = [{**record, **reviews.get(str(record["video_id"]), {})} for record in records]
    with (output / "master_metadata.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "similarity_candidates.csv").open("w", newline="", encoding="utf-8-sig") as target:
        fields = ["video_id_a", "video_id_b", "min_visual_hash_distance", "review_required"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    exact = defaultdict(list)
    for record in records:
        exact[record["sha256"]].append(record["video_id"])
    report = {
        "videos": len(records), "candidate_labels": dict(sorted(Counter(record["candidate_label"] for record in records).items())),
        "review_status": dict(sorted(Counter(record["review_status"] for record in merged).items())),
        "verified_styles": dict(sorted(Counter(
            record["style"] for record in merged if record["review_status"] == "verified"
        ).items())),
        "exact_duplicate_groups": [ids for ids in exact.values() if len(ids) > 1],
        "visual_similarity_candidates": len(candidates),
        "warning": "Folder labels and visual similarity are review hints, not verified truth.",
    }
    (output / "inventory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not state.exists():
        state.write_text("{}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    records, candidates = build_inventory()
    write_outputs(records, candidates)
    print(f"Wrote {len(records)} videos and {len(candidates)} similarity candidates")


if __name__ == "__main__":
    main()
