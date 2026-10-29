from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .reviews import merge_reviews, validate_review


ROOT = Path(__file__).resolve().parents[2]
KNOWN_STYLES = {"butterfly", "breaststroke", "freestyle", "backstroke"}


def triage(row: dict[str, str]) -> str:
    if row["review_status"] != "ambiguous":
        return "already_decided"
    if row["behavior"] == "drowning_candidate" or row["mixed_styles"] == "yes":
        return "deferred_complex"
    clean_style = row["style"] in KNOWN_STYLES and row["behavior"] == "normal" and row["mixed_styles"] == "no"
    if clean_style and row["swimmer_count"] == "single":
        return "bulk_confirmation_candidate"
    if clean_style and row["swimmer_count"] == "multiple":
        return "same_style_multiple_swimmers"
    return "needs_manual_review"


def note_evidence(row: dict[str, str]) -> str:
    note = row["notes"].lower()
    if "karma stil" in note:
        return "note_conflicts_with_mixed_styles_no"
    if note in {"", "unknown"}:
        return "structured_fields_only"
    return "note_supports_same_style"


def write_csv(path: Path, rows: list[dict[str, str]], extra: dict[str, str] | None = None) -> None:
    fields = ["video_id", "filename", "path", "style", "behavior", "swimmer_count", "mixed_styles",
              "review_status", "source_id", "session_id", "notes"]
    if rows and "evidence" in rows[0]:
        fields.append("evidence")
    output = [{field: row[field] for field in fields} | (extra or {}) for row in rows]
    if extra:
        fields += list(extra)
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    output = ROOT / "results" / "video_review"
    parser = argparse.ArgumentParser(description="List review candidates without changing review decisions.")
    parser.add_argument("--metadata", type=Path, default=output / "master_metadata.csv")
    parser.add_argument("--reviews", type=Path, default=output / "review_state.json")
    parser.add_argument("--output-dir", type=Path, default=output)
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8-sig", newline="") as source:
        master = list(csv.DictReader(source))
    raw = json.loads(args.reviews.read_text(encoding="utf-8"))
    known = {row["video_id"] for row in master}
    state = {video_id: validate_review({"video_id": video_id, **review}, known) for video_id, review in raw.items()}
    rows = merge_reviews(master, state)
    groups = {name: [row for row in rows if triage(row) == name] for name in {
        "bulk_confirmation_candidate", "same_style_multiple_swimmers", "deferred_complex", "needs_manual_review"
    }}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "bulk_confirmation_candidates.csv", groups["bulk_confirmation_candidate"],
              {"recommendation": "pending_user_approval"})

    multi = []
    for row in groups["same_style_multiple_swimmers"]:
        multi.append(row | {"evidence": note_evidence(row)})
    write_csv(args.output_dir / "same_style_multiple_swimmers.csv", multi)
    write_csv(args.output_dir / "deferred_complex_cases.csv", groups["deferred_complex"])

    verified = [row for row in rows if row["review_status"] == "verified"]
    verified_style = [row for row in verified if row["style"] in KNOWN_STYLES and row["mixed_styles"] == "no"]
    statuses = Counter(row["review_status"] for row in rows)
    candidate_ids = ", ".join(row["video_id"] for row in groups["bulk_confirmation_candidate"])
    evidence = Counter(note_evidence(row) for row in groups["same_style_multiple_swimmers"])
    missing_group = sum(row["source_id"] in {"", "unknown"} or row["session_id"] in {"", "unknown"} for row in rows)

    label_report = f"""# Etiket doğrulama uygunluk raporu

## Mevcut durum

- Toplam: **{len(rows)}** ({', '.join(f'{key}: {value}' for key, value in sorted(statuses.items()))})
- Halen doğrulanmış ve stil etiketi kullanılabilir: **{len(verified_style)}**
- Kullanıcı toplu onayına sunulabilecek tek yüzücülü temiz aday: **{len(groups['bulk_confirmation_candidate'])}**
- Çok yüzücülü, `mixed_styles=no` olarak kaydedilmiş ayrı inceleme grubu: **{len(groups['same_style_multiple_swimmers'])}**
- Karma stil veya boğulma adayı olarak ertelenen: **{len(groups['deferred_complex'])}**

Hiçbir `review_status` değiştirilmedi. Aşağıdaki kayıtlar yalnızca toplu onay önerisidir:

`{candidate_ids}`

Bu altı kayıt `single`, bilinen stil, `mixed_styles=no` ve `behavior=normal` koşullarını karşılıyor. Kullanıcı onaylarsa stil bakımından kullanılabilir doğrulanmış sayı 27'den **{len(verified_style) + len(groups['bulk_confirmation_candidate'])}**'e çıkabilir.

## V028, V029 ve V030

Üçü de `backstroke`, `single`, `mixed_styles=no`, `behavior=normal`; ancak kayıtlı `review_status=ambiguous` ve `notes=unknown`. Dolayısıyla belirsizliğin insan tarafından seçilmiş özel nedeni JSON'dan çıkarılamıyor. Kaynak/oturum eksikliği doğrulama kararından ayrı bir split sorunudur. V030 görsel benzerlik tablosunda RF3FF011533, V031 ve V032 ile yakın eşleşir; aynı kaynak olasılığı split öncesinde çözülmelidir. V028 ve V029 için mevcut eşik altında benzerlik adayı kaydı yoktur.

## Çok yüzücülü aynı-stil grubu

- Not aynı stili destekliyor: **{evidence['note_supports_same_style']}**
- Yalnızca yapılandırılmış alanlar destekliyor: **{evidence['structured_fields_only']}**
- Not, `mixed_styles=no` alanıyla çelişiyor: **{evidence['note_conflicts_with_mixed_styles_no']}**

Bu grup otomatik onaylanmamalıdır. Çelişkili iki kayıt düzeltilmeden; notsuz kayıtlar da kısa bir kontrol yapılmadan güvenli kabul edilemez. Ayrıntılar `same_style_multiple_swimmers.csv` içindedir.

## Ayrı tutulan kayıtlar

Karma stilli ve boğulma adayı kayıtlar `deferred_complex_cases.csv` içine alındı. Bunlar kişi kırpma, zaman aralığı etiketi veya davranış doğrulaması gerektirir.

`aquamed_reviews.json` proje, Desktop, Downloads ve Documents altında bulunamadı. Bu nedenle karşılaştırma yalnızca mevcut `review_state.json` üzerinden yapılmıştır; eksik dosya adına veri varsayılmamıştır.
"""
    (args.output_dir / "label_eligibility_report.md").write_text(label_report, encoding="utf-8")

    split_report = f"""# Güvenilir train/validation/test ayrımı raporu

- Etiket bakımından halen kullanılabilir: **{len(verified_style)}**
- Kullanıcı onayı bekleyen temiz etiket adayı: **{len(groups['bulk_confirmation_candidate'])}**
- Kaynak veya oturum bilgisi eksik kayıt: **{missing_group}/{len(rows)}**
- Bilinen güvenilir grup: **0**
- Güvenilir değerlendirme split'ine hazır: **0**

Etiket uygunluğu ile değerlendirme uygunluğu aynı şey değildir. Stil etiketi doğru olsa bile gerçek `source_id` ve `session_id` bilinmeden aynı çekimin veya aynı kişinin train ve test tarafına sızmadığı gösterilemez. Video başına rastgele ya da benzersiz grup kimliği üretilmemiştir.

Asgari eksik işlem, gerçek kaynak/oturum ilişkilerini bilinen kayıt geçmişinden veya görsel benzerlik kümelerinden kanıtlayarak doldurmak ve split'i bu gruplar üzerinde yapmaktır. Dört sınıflı eğitim için ayrıca doğrulanmış backstroke gerekir; V028/V029/V030 ve diğer temiz adaylar kullanıcı tarafından onaylanırsa bu sınıf açılabilir, fakat grup sorunu yine çözülmez.
"""
    (args.output_dir / "split_eligibility_report.md").write_text(split_report, encoding="utf-8")
    print(json.dumps({"bulk_confirmation_candidates": len(groups["bulk_confirmation_candidate"]),
                      "same_style_multiple_swimmers": len(groups["same_style_multiple_swimmers"]),
                      "deferred_complex": len(groups["deferred_complex"]),
                      "reliable_split_ready": 0}, indent=2))


if __name__ == "__main__":
    main()
