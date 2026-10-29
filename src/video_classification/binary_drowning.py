from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.video_review.reviews import merge_reviews


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["normal_swimming", "drowning_behavior"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Groups:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def prepare_rows(master: list[dict[str, str]], state: dict[str, dict[str, str]],
                 similarity: list[dict[str, str]], fold_count: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = merge_reviews(master, state)
    groups = Groups([row["video_id"] for row in rows])
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_source: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
        if row.get("source_id") not in {"", "unknown", None} and row.get("session_id") not in {"", "unknown", None}:
            by_source[(row["source_id"], row["session_id"])].append(row["video_id"])
    for related in by_hash.values():
        for row in related[1:]:
            groups.union(related[0]["video_id"], row["video_id"])
    for related in by_source.values():
        for video_id in related[1:]:
            groups.union(related[0], video_id)
    known_ids = groups.parent.keys()
    for pair in similarity:
        if pair["video_id_a"] in known_ids and pair["video_id_b"] in known_ids:
            groups.union(pair["video_id_a"], pair["video_id_b"])

    canonical = {row["video_id"]: min(item["video_id"] for item in by_hash[row["sha256"]]) for row in rows}
    enriched = []
    for row in rows:
        drowning = Path(row["path"]).parent.name == "bogulma"
        if drowning:
            reason = "" if row.get("behavior") == "drowning_candidate" and row.get("review_status") != "excluded" else "review_conflicts_with_drowning_candidate"
        elif row.get("review_status") != "verified":
            reason = "normal_review_not_verified"
        elif row.get("behavior") != "normal":
            reason = "normal_behavior_uncertain_or_different"
        elif row.get("mixed_styles") != "no":
            reason = "mixed_or_uncertain_behavior"
        else:
            reason = ""
        is_canonical = row["video_id"] == canonical[row["video_id"]]
        if not is_canonical:
            reason = "exact_duplicate"
        label = CLASSES[1] if drowning else CLASSES[0]
        enriched.append({
            **row,
            "binary_label": label,
            "label_strength": "weak_video_label" if drowning else "video_level_label",
            "clip_label": "unlabeled" if drowning else label,
            "canonical_video_id": canonical[row["video_id"]],
            "is_exact_duplicate": not is_canonical,
            "relation_group": groups.find(row["video_id"]),
            "initial_training_eligible": is_canonical and not reason,
            "binary_exclusion_reason": reason,
        })

    bundles: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        bundles[str(row["relation_group"])].append(row)
    fold_counts = [Counter() for _ in range(fold_count)]
    assignments = {}
    ordered = sorted(bundles.items(), key=lambda item: (-sum(bool(row["initial_training_eligible"]) for row in item[1]), item[0]))
    for group, bundle in ordered:
        counts = Counter(str(row["binary_label"]) for row in bundle if row["initial_training_eligible"])
        fold = min(range(fold_count), key=lambda index: (
            sum(fold_counts[index].values()), fold_counts[index][CLASSES[1]], fold_counts[index][CLASSES[0]], index
        ))
        assignments[group] = fold
        fold_counts[fold].update(counts)
    for row in enriched:
        row["fold"] = assignments[str(row["relation_group"])]

    by_id = {str(row["video_id"]): row for row in enriched}
    for pair in similarity:
        left, right = by_id.get(pair["video_id_a"]), by_id.get(pair["video_id_b"])
        if left and right and left["fold"] != right["fold"]:
            raise AssertionError("Visual relation crosses folds")
    duplicates = []
    for digest, related in by_hash.items():
        if len(related) > 1:
            canonical_id = min(row["video_id"] for row in related)
            duplicates.append({"sha256": digest, "canonical_video_id": canonical_id,
                               "duplicate_video_ids": sorted(row["video_id"] for row in related if row["video_id"] != canonical_id)})
    return enriched, {"duplicates": duplicates, "fold_class_counts": [dict(counts) for counts in fold_counts]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare binary drowning metadata without training a model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "r3d18_binary_drowning.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    master = read_csv(ROOT / config["master_metadata"])
    state = json.loads((ROOT / config["review_state"]).read_text(encoding="utf-8"))
    rows, checks = prepare_rows(master, state, read_csv(ROOT / config["similarity_candidates"]), int(config["cv_folds"]))
    write_csv(output / "binary_metadata.csv", rows)
    unique = [row for row in rows if not row["is_exact_duplicate"]]
    eligible = [row for row in rows if row["initial_training_eligible"]]
    report = {
        "unique_normal_videos": sum(row["binary_label"] == CLASSES[0] for row in unique),
        "unique_drowning_candidates": sum(row["binary_label"] == CLASSES[1] for row in unique),
        "initial_training_videos": dict(Counter(str(row["binary_label"]) for row in eligible)),
        "excluded_unique_videos": dict(Counter(str(row["binary_exclusion_reason"]) for row in unique if row["binary_exclusion_reason"])),
        "exact_duplicate_groups": len(checks["duplicates"]),
        "fold_class_counts": checks["fold_class_counts"],
        "model": {"architecture": "torchvision.models.video.r3d_18", "pretrained": True,
                  "classes": CLASSES, "output_units": 2, "score_name": "drowning_behavior_score",
                  "positive_class_index": 1, "score_definition": "softmax class score",
                  "score_is_calibrated_risk_probability": False},
        "training_strategy": config["training_strategy"],
        "evaluation": {"level": "video", "metrics": ["precision", "recall", "f1", "false_positive_rate"]},
        "label_uncertainty": {
            "weak_drowning_video_labels": sum(row["binary_label"] == CLASSES[1] for row in eligible),
            "confirmed_positive_clips": 0,
            "excluded_uncertain_or_different_normal_videos": sum(
                row["binary_label"] == CLASSES[0] and bool(row["binary_exclusion_reason"]) for row in unique
            ),
            "evaluation_truth": "folder-derived weak video labels, not event-level ground truth",
        },
        "limitations": [
            "Drowning labels are folder-derived weak video labels; no clip is a confirmed positive.",
            "Visual relations are conservative candidates, not confirmed common sources.",
            "No independent test set or calibrated real-world risk probability is available.",
        ],
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "deduplication.json").write_text(json.dumps(checks["duplicates"], indent=2), encoding="utf-8")
    (output / "model_config.json").write_text(json.dumps(report["model"], indent=2), encoding="utf-8")
    (output / "preparation_report.md").write_text(f"""# AquaMed ikili R3D-18 hazırlık raporu

- Benzersiz normal yüzme videosu: **{report['unique_normal_videos']}**
- Benzersiz boğulma adayı: **{report['unique_drowning_candidates']}**
- İlk eğitime uygun normal video: **{report['initial_training_videos'].get(CLASSES[0], 0)}**
- Zayıf video etiketiyle kullanılabilecek boğulma adayı: **{report['initial_training_videos'].get(CLASSES[1], 0)}**
- Kesin pozitif klip: **0**
- Tekilleştirilen birebir kopya grubu: **{report['exact_duplicate_groups']}**

Boğulma kayıtları video-bag düzeyinde `weak_video_label` taşır; klipleri `unlabeled` kalır.
Başlangıç stratejisi pretrained R3D-18'in donmuş omurgası ve iki çıkışlı başlığı üzerinde,
video-bag/top-k multiple-instance learning'dir. Normal tarafta yalnız `verified`, `behavior=normal`
ve `mixed_styles=no` kayıtlar ilk havuza alınmıştır.

Değerlendirme video düzeyinde precision, recall, F1 ve false positive rate üretir. Pozitif sınıf
çıktısının adı `drowning_behavior_score`dur; kalibre edilmiş gerçek risk olasılığı değildir.
Mevcut zayıf etiketlere karşı ölçülen sonuç, olay düzeyi gerçek doğruluk olarak yorumlanamaz.

Eksikler: boğulma olay zaman aralıkları ve doğrulanmış pozitif klipler, kişi takibi,
doğrulanmış kaynak/oturum kimlikleri, bağımsız test kümesi ve saha kalibrasyonu.
""", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
