from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .reviews import STYLES, merge_reviews, training_reason, validate_review


ROOT = Path(__file__).resolve().parents[2]
KNOWN_STYLES = STYLES - {"unknown"}


def exclusion_reasons(row: dict[str, str]) -> list[str]:
    reasons = []
    if row.get("video_style_verification", row["review_status"]) != "verified":
        reasons.append("not_verified")
    if row["style"] not in KNOWN_STYLES:
        reasons.append("style_unknown")
    if row["behavior"] == "drowning_candidate":
        reasons.append("drowning_is_candidate_only")
    if row["mixed_styles"] != "no":
        reasons.append("mixed_styles_not_ruled_out")
    if row["source_id"] in {"", "unknown"}:
        reasons.append("source_unknown")
    if row["session_id"] in {"", "unknown"}:
        reasons.append("session_unknown")
    return reasons


def recovery_action(row: dict[str, str]) -> str:
    if row["review_status"] == "verified":
        return "source_and_session_attribution"
    if row["behavior"] == "drowning_candidate":
        return "event_interval_and_outcome_validation"
    if row["mixed_styles"] == "yes" and row["swimmer_count"] == "multiple":
        return "person_crop_and_or_time_range_label"
    if row["mixed_styles"] == "yes":
        return "time_range_label"
    if row["swimmer_count"] == "multiple":
        return "person_crop_then_status_review"
    return "status_review"


def diagnostic_rows(master: list[dict[str, str]], state: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for row in merge_reviews(master, state):
        reasons = exclusion_reasons(row)
        style_label_eligible = (
            row.get("video_style_verification", row["review_status"]) == "verified"
            and row["style"] in KNOWN_STYLES
            and row["behavior"] != "drowning_candidate"
            and row["mixed_styles"] == "no"
        )
        behavior_label_eligible = row["review_status"] == "verified" and row["behavior"] == "normal"
        rows.append({
            "video_id": row["video_id"], "filename": row["filename"],
            "review_status": row["review_status"], "style": row["style"],
            "video_style_verification": row.get("video_style_verification", "unknown"),
            "person_style_verification": row.get("person_style_verification", "unknown"),
            "behavior": row["behavior"], "swimmer_count": row["swimmer_count"],
            "mixed_styles": row["mixed_styles"], "source_id": row["source_id"],
            "session_id": row["session_id"],
            "style_label_eligible": style_label_eligible,
            "behavior_label_eligible": behavior_label_eligible,
            "split_eligible": style_label_eligible and not ({row["source_id"], row["session_id"]} & {"", "unknown"}),
            "current_training_eligible": training_reason(row) is None,
            "exclusion_reasons": "|".join(reasons),
            "recovery_action": recovery_action(row),
        })
    return rows


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    verified = [row for row in rows if row["review_status"] == "verified"]
    ambiguous = [row for row in rows if row["review_status"] == "ambiguous"]
    style_ready = [row for row in rows if row["style_label_eligible"]]
    behavior_ready = [row for row in rows if row["behavior_label_eligible"]]
    only_group_missing = [row for row in rows if row["style_label_eligible"] and not row["split_eligible"]]

    def counts(field: str, subset: list[dict[str, object]]) -> str:
        return ", ".join(f"{key}: {value}" for key, value in sorted(Counter(row[field] for row in subset).items()))

    recovery = Counter(row["recovery_action"] for row in ambiguous)
    crop_candidates = sum(row["swimmer_count"] == "multiple" for row in ambiguous)
    interval_candidates = sum(row["mixed_styles"] == "yes" or row["behavior"] == "drowning_candidate" for row in ambiguous)
    drowning_multiple = sum(row["behavior"] == "drowning_candidate" and row["swimmer_count"] == "multiple" for row in ambiguous)
    report = f"""# Video review eligibility diagnostic

## Sonuç

- Toplam kayıt: **{len(rows)}**
- Doğrulanmış: **{len(verified)}**; belirsiz: **{len(ambiguous)}**
- Stil etiketi bakımından uygun: **{len(style_ready)}**
- Yalnızca kaynak/oturum eksikliği nedeniyle split'e giremeyen: **{len(only_group_missing)}**
- Mevcut kuralla eğitim uygun: **{sum(bool(row['current_training_eligible']) for row in rows)}**

`training_eligible=0` bir etiket başarısızlığı değildir. Stil için doğrulanmış {len(style_ready)} kaydın tümünde `source_id` ve `session_id` bilinmiyor. Mevcut filtre, aynı kaynağın farklı split'lere sızmasını önlemek için bu kayıtları dışlıyor. Bilinmeyen gruplara rastgele veya video başına benzersiz kimlik atanmadı.

## {len(verified)} doğrulanmış videonun dağılımı

- Stil: {counts('style', verified)}
- Davranış: {counts('behavior', verified)}
- Yüzücü sayısı: {counts('swimmer_count', verified)}
- Karma stil: {counts('mixed_styles', verified)}

Bu kayıtların {len(style_ready)} tanesi stil etiketi bakımından kullanılabilir. Davranış bakımından {len(behavior_ready)} tanesi doğrulanmış `normal` örnektir; doğrulanmış pozitif boğulma örneği yoktur. Bu nedenle ikili boğulma davranışı modeli eğitilemez.

## Diğer {len(ambiguous)} video

Bu kayıtların tamamı `ambiguous` durumundadır; hiçbiri otomatik doğrulanmış sayılmadı.

- Stil: {counts('style', ambiguous)}
- Davranış: {counts('behavior', ambiguous)}
- Yüzücü sayısı: {counts('swimmer_count', ambiguous)}
- Karma stil: {counts('mixed_styles', ambiguous)}
- Önerilen ek anotasyon: {', '.join(f'{key}: {value}' for key, value in sorted(recovery.items()))}

`person_crop_and_or_time_range_label` kayıtları birden fazla yüzücü ve aynı anda/ardışık karma stil içerir. `person_crop_then_status_review` kayıtlarında kişiler ayrıştırılabilir. `event_interval_and_outcome_validation` grubundaki boğulma adayları için kişi takibi, olay zaman aralığı ve gerçek pozitif/negatif doğrulaması gerekir.

Kişi bazlı kırpma **{crop_candidates}** çok yüzücülü kayıtta potansiyel olarak yararlıdır. Zaman aralığı etiketi, karma stil veya boğulma adayı olan **{interval_candidates}** kayıtta gereklidir; iki küme çakışır ve boğulma adaylarının **{drowning_multiple}** tanesi aynı zamanda çok yüzücülüdür. Kalan `status_review` kayıtları temiz görünebilir, fakat mevcut insan kararı `ambiguous` olduğu için otomatik doğrulanamaz.

`review_state.json` ayrı bir “belirsizlik nedeni” alanı içermiyor. Bu sınıflandırma yalnızca kaydedilmiş davranış, yüzücü sayısı ve karma-stil alanlarına dayanır; kullanıcının karar gerekçesi notta yazmıyorsa daha kesin bir neden uydurulmamıştır.

## Etiket uygunluğu ve split uygunluğu

Etiket uygunluğu, videonun hedef sınıfının güvenilir olup olmadığını söyler. Split uygunluğu ise aynı kaynak veya kayıt oturumunun train/validation/test arasında dağılmadığının kanıtlanabilmesini gerektirir. Mevcut durumda stil etiketi uygun {len(style_ready)} video vardır, fakat bilinen kaynak/oturum grubu **0** olduğu için split uygun video **0** ve `split_ready=false` kalır.

Dört sınıflı stil eğitimi ayrıca hazır değildir: doğrulanmış kayıtlarda `backstroke` sınıfı yoktur. Mevcut etiketlerle yalnızca butterfly/breaststroke/freestyle kapsanır. Veri sayısı da gerçek model değerlendirmesi için küçüktür.

## Eğitime başlayabilmek için asgari işlemler

1. Stil etiketi uygun {len(style_ready)} video için gerçek kaynak ve oturum ilişkilerini belirle; bilinmeyenlere rastgele ID verme.
2. Aynı kaynak/oturum ve olası yeniden kesitleri tek grup altında tut; sonra grup bazlı train/validation/test split oluştur.
3. Dört stil hedefleniyorsa yeterli sayıda temiz `backstroke` kaydını doğrula; aksi halde hedefi açıkça üç sınıfla sınırla.
4. Boğulma davranışı hedefleniyorsa 27 adayı kişi ve zaman aralığı düzeyinde doğrula; doğrulanmış pozitif ve negatif gruplar oluşmadan eğitim başlatma.

Her videonun tüm dışlanma gerekçeleri `eligibility_diagnostic.csv` içinde ayrı satırda verilmiştir. Mevcut etiketler ve inceleme kararları değiştirilmemiştir.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain review, label and split eligibility without changing reviews.")
    output = ROOT / "results" / "video_review"
    parser.add_argument("--metadata", type=Path, default=output / "master_metadata.csv")
    parser.add_argument("--reviews", type=Path, default=output / "review_state.json")
    parser.add_argument("--output-dir", type=Path, default=output)
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8-sig", newline="") as source:
        master = list(csv.DictReader(source))
    raw_state = json.loads(args.reviews.read_text(encoding="utf-8"))
    known = {row["video_id"] for row in master}
    state = {video_id: validate_review({"video_id": video_id, **review}, known) for video_id, review in raw_state.items()}
    rows = diagnostic_rows(master, state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "eligibility_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, args.output_dir / "eligibility_diagnostic_report.md")
    print(json.dumps({"total": len(rows), "verified": sum(row["review_status"] == "verified" for row in rows),
                      "style_label_eligible": sum(bool(row["style_label_eligible"]) for row in rows),
                      "split_eligible": sum(bool(row["split_eligible"]) for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
