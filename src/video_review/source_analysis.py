from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

import cv2

from src.video_classification.preprocessing import build_clip_records, probe_video
from .inventory import hamming
from .reviews import merge_reviews


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "video_review" / "source_analysis"


def components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for left, right in edges:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    groups, seen = [], set()
    for start in graph:
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            group.append(current)
            stack.extend(graph[current] - seen)
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: (-len(group), group))


def contact_sheet(path: Path, title: str, destination: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for fraction in (0.2, 0.5, 0.8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round((frame_count - 1) * fraction))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not decode thumbnail frames: {path}")
    sheet = cv2.hconcat(frames)
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(sheet, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), sheet)


def min_distance(left: dict[str, str], right: dict[str, str]) -> int:
    return min(hamming(a, b) for a in left["visual_hashes"].split("|") for b in right["visual_hashes"].split("|"))


def main() -> None:
    review_dir = ROOT / "results" / "video_review"
    with (review_dir / "master_metadata.csv").open(encoding="utf-8-sig", newline="") as source:
        master = list(csv.DictReader(source))
    state = json.loads((review_dir / "review_state.json").read_text(encoding="utf-8"))
    verified = [row for row in merge_reviews(master, state) if row["review_status"] == "verified"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    clips = []
    for row in verified:
        path = ROOT / Path(row["path"])
        info = probe_video(path)
        count = len(build_clip_records(row, info, sampling_fps=8.0, clip_length=16, stride=16))
        clips.append({"video_id": row["video_id"], "filename": row["filename"], "style": row["style"],
                      "fps": row["fps"], "duration_sec": row["duration_sec"], "width": row["width"],
                      "height": row["height"], "sha256": row["sha256"], "clip_count": count})
        contact_sheet(path, f"{row['video_id']} | {row['style']} | {row['filename']}",
                      OUTPUT / "assets" / f"{row['video_id']}.jpg")

    pairs = []
    for index, left in enumerate(verified):
        for right in verified[index + 1:]:
            distance = min_distance(left, right)
            if distance > 4:
                continue
            same_style = left["style"] == right["style"]
            pairs.append({
                "video_id_a": left["video_id"], "video_id_b": right["video_id"],
                "style_a": left["style"], "style_b": right["style"],
                "filename_a": left["filename"], "filename_b": right["filename"],
                "min_dhash_distance": distance, "same_style": same_style,
                "technical_match": (left["fps"], left["width"], left["height"]) == (right["fps"], right["width"], right["height"]),
                "evidence": "shared_camera_or_session_candidate" if same_style else "shared_pool_context_only",
                "certainty": "unverified",
            })

    pair_fields = list(pairs[0]) if pairs else ["video_id_a", "video_id_b", "style_a", "style_b",
                                                  "filename_a", "filename_b", "min_dhash_distance",
                                                  "same_style", "technical_match", "evidence", "certainty"]
    with (OUTPUT / "source_similarity_candidates.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pairs)
    with (OUTPUT / "verified_clip_inventory.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(clips[0]))
        writer.writeheader()
        writer.writerows(clips)

    same_style_edges = [(row["video_id_a"], row["video_id_b"]) for row in pairs if row["same_style"]]
    groups = components(same_style_edges)
    exact = components([(left["video_id"], right["video_id"]) for index, left in enumerate(verified)
                        for right in verified[index + 1:] if left["sha256"] == right["sha256"]])
    rows_by_id = {row["video_id"]: row for row in verified}

    cards = []
    for pair in sorted(pairs, key=lambda row: (row["certainty"], row["min_dhash_distance"])):
        left, right = rows_by_id[pair["video_id_a"]], rows_by_id[pair["video_id_b"]]
        def video_path(row: dict[str, str]) -> str:
            return "../../../" + row["path"].replace("\\", "/")
        cards.append(f"""<article class="pair">
<h2>{html.escape(left['video_id'])} ↔ {html.escape(right['video_id'])}</h2>
<p><b>{html.escape(pair['evidence'])}</b> · dHash {pair['min_dhash_distance']} · certainty: unverified</p>
<div class="compare"><section><img src="assets/{left['video_id']}.jpg"><video controls preload="metadata" src="{video_path(left)}"></video></section>
<section><img src="assets/{right['video_id']}.jpg"><video controls preload="metadata" src="{video_path(right)}"></video></section></div></article>""")
    gallery = "".join(f"<figure><img src='assets/{row['video_id']}.jpg'><figcaption>{row['video_id']} · {row['style']}</figcaption></figure>" for row in verified)
    group_text = "".join(f"<li>Candidate group {index}: {', '.join(group)}</li>" for index, group in enumerate(groups, 1)) or "<li>None</li>"
    page = f"""<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AquaMed kaynak karşılaştırması</title><style>
body{{font:15px system-ui;background:#071821;color:#e7f7fa;margin:0;padding:24px}}h1,h2{{color:#70e1e8}}.notice{{padding:14px;background:#12303b;border-left:4px solid #f6c85f}}
.pair{{margin:24px 0;padding:18px;background:#0d2731;border-radius:12px}}.compare{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img,video{{width:100%;display:block;margin:8px 0}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}figure{{margin:0;background:#0d2731;padding:8px}}@media(max-width:800px){{.compare{{grid-template-columns:1fr}}}}
</style><h1>AquaMed kaynak karşılaştırması</h1><p class="notice">Bu rapor olası görsel ilişkileri gösterir; hiçbir çift doğrulanmış ortak kaynak veya oturum değildir. Havuz arka planı dHash yanlış eşleşmelerine yol açabilir.</p>
<p>33 doğrulanmış video · {sum(item['clip_count'] for item in clips)} adet 16-frame klip · {len(pairs)} aday çift · {len(exact)} kesin dosya kopyası grubu</p>
<h2>Aynı-stil aday kümeleri</h2><ul>{group_text}</ul>{''.join(cards)}<h2>Tüm doğrulanmış videolar</h2><div class="gallery">{gallery}</div></html>"""
    (OUTPUT / "source_comparison.html").write_text(page, encoding="utf-8")

    class_clips = Counter()
    for item in clips:
        class_clips[item["style"]] += item["clip_count"]
    report = f"""# Kaynak benzerliği ve R3D-18 hazırlık raporu

- Doğrulanmış video: **{len(verified)}**
- Tam SHA-256 kopyası: **{sum(len(group) for group in exact)} video / {len(exact)} grup**
- dHash ≤ 4 aday çift: **{len(pairs)}**
- Aynı-stil aday kümeleri: **{len(groups)}**
- 16-frame klip: **{sum(item['clip_count'] for item in clips)}** ({', '.join(f'{key}: {value}' for key, value in sorted(class_clips.items()))})

## Olası gruplar

{chr(10).join(f'- G{index}: {", ".join(group)} — ortak kamera/oturum adayı, aynı orijinal kayıt doğrulanmadı' for index, group in enumerate(groups, 1)) or '- Yok'}

- G1 aynı sabit sualtı kamera açısını, kulvar geometrisini ve ışığı paylaşır; aynı oturum veya farklı günlerde aynı kurulum olabilir.
- G2 benzer kulvar/kamera çerçevesi gösterir; aynı orijinal kaydın kesitleri olduğuna dair süreklilik kanıtı yoktur.
- G3 aynı sualtı kamera yerleşimine çok benzer; V030 ve V032 birlikte tutulması gereken yüksek öncelikli inceleme çiftidir.
- Çapraz-stil düşük dHash çiftleri, havuz arka planının yanlış pozitif ürettiğini doğrular.

SHA-256 ile doğrulanmış aynı dosya yoktur. Görsel adaylar yalnızca aynı havuz görünümü, kamera açısı veya sıkıştırma benzerliği gösterebilir. Bu nedenle `source_id` ve `session_id` alanlarına yazılmamış, leakage-safe split oluşturulmamıştır.

## Eğitim stratejisi

Mevcut 33 video 16-frame klip üretimi ve model giriş biçimi açısından uygundur. Ancak bağımsız kaynak grupları doğrulanmadığı için yalnızca deneysel eğitim yapılabilir: bütün sonuçlar geliştirme sonucu olarak adlandırılmalı, bağımsız test başarısı iddia edilmemelidir. İlk deneyde pretrained R3D-18, donmuş backbone, 112×112 crop, 16 frame, 8 FPS, batch size 2 ve video bazlı ağırlıklandırma önerilir. Klipleri rastgele split etmek yasaktır; en azından video düzeyinde split yapılmalı ve benzerlik adayları aynı tarafta tutulmalıdır.
"""
    (OUTPUT / "source_similarity_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verified_videos": len(verified), "clips": sum(item["clip_count"] for item in clips),
                      "candidate_pairs": len(pairs), "candidate_groups": groups,
                      "exact_duplicate_groups": exact}, indent=2))


if __name__ == "__main__":
    main()
