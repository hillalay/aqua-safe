from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path


STYLES = {"butterfly", "breaststroke", "freestyle", "backstroke", "unknown"}
BEHAVIORS = {"normal", "drowning_candidate", "unknown"}
SWIMMER_COUNTS = {"single", "multiple", "unknown"}
TRISTATE = {"yes", "no", "unknown"}
STATUSES = {"unreviewed", "verified", "ambiguous", "excluded"}
EDITABLE_FIELDS = {"style", "behavior", "swimmer_count", "mixed_styles", "review_status",
                   "source_id", "session_id", "notes"}
VERIFICATIONS = {"verified", "unknown"}


def validate_review(review: dict[str, object], known_ids: set[str]) -> dict[str, str]:
    video_id = str(review.get("video_id", ""))
    if video_id not in known_ids:
        raise ValueError(f"Unknown video_id: {video_id}")
    clean = {"video_id": video_id}
    for field in EDITABLE_FIELDS:
        clean[field] = str(review.get(field, "")).strip()
    if clean["style"] not in STYLES or clean["behavior"] not in BEHAVIORS:
        raise ValueError("Invalid style or behavior")
    if clean["swimmer_count"] not in SWIMMER_COUNTS or clean["mixed_styles"] not in TRISTATE:
        raise ValueError("Invalid swimmer count or mixed-style value")
    if clean["review_status"] not in STATUSES:
        raise ValueError("Invalid review status")
    clean["video_style_verification"] = str(review.get(
        "video_style_verification", "verified" if clean["review_status"] == "verified" else "unknown"
    )).strip()
    clean["person_style_verification"] = str(review.get(
        "person_style_verification",
        "verified" if clean["review_status"] == "verified" and clean["swimmer_count"] == "single" else "unknown",
    )).strip()
    if clean["video_style_verification"] not in VERIFICATIONS or clean["person_style_verification"] not in VERIFICATIONS:
        raise ValueError("Invalid video/person style verification")
    if clean["behavior"] == "drowning_candidate" and clean["review_status"] == "verified":
        clean["review_status"] = "ambiguous"
    return clean


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def merge_reviews(master: list[dict[str, str]], state: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [{**row, **state.get(row["video_id"], {})} for row in master]


def training_reason(row: dict[str, str]) -> str | None:
    if row.get("video_style_verification", row["review_status"]) != "verified":
        return "not_verified"
    if row["style"] not in STYLES - {"unknown"}:
        return "style_unknown"
    if row["behavior"] == "drowning_candidate":
        return "drowning_is_candidate_only"
    if row["mixed_styles"] != "no":
        return "mixed_styles_not_ruled_out"
    if row["source_id"] in {"", "unknown"} or row["session_id"] in {"", "unknown"}:
        return "source_or_session_unknown"
    return None


def export_training_metadata(master: list[dict[str, str]], state: dict[str, dict[str, str]], output_dir: Path) -> dict[str, object]:
    merged = merge_reviews(master, state)
    validated = []
    eligible = []
    for row in merged:
        reason = training_reason(row)
        video_verification = row.get("video_style_verification", "verified" if row["review_status"] == "verified" else "unknown")
        person_verification = row.get(
            "person_style_verification",
            "verified" if row["review_status"] == "verified" and row["swimmer_count"] == "single" else "unknown",
        )
        exported = {**row, "video_style_verification": video_verification,
                    "person_style_verification": person_verification, "style_verified": video_verification,
                    "swimmer_id": "unknown", "training_eligible": reason is None, "exclusion_reason": reason or ""}
        validated.append(exported)
        if reason is None:
            eligible.append(exported)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("validated_metadata.csv", validated), ("training_metadata.csv", eligible)):
        fields = list(validated[0])
        with (output_dir / name).open("w", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    groups = Counter((row["source_id"], row["session_id"]) for row in eligible)
    verified_styles = Counter(row["style"] for row in validated if row["video_style_verification"] == "verified")
    blockers = Counter(row["exclusion_reason"] for row in validated if row["exclusion_reason"])
    report = {"total": len(merged), "verified": sum(row["review_status"] == "verified" for row in merged),
              "verified_styles": dict(sorted(verified_styles.items())),
              "training_eligible": len(eligible), "eligible_styles": dict(Counter(row["style"] for row in eligible)),
              "training_blockers": dict(sorted(blockers.items())),
              "known_groups": len(groups), "split_ready": False,
              "reason": "Splits remain unassigned until every eligible class has enough independent source/session groups."}
    (output_dir / "training_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
