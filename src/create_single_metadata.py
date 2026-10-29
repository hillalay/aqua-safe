
from pathlib import Path
import pandas as pd
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "single"
OUTPUT = PROJECT_ROOT / "results" / "single_metadata.csv"

classes = ["kelebek", "kurbagalama", "serbest", "sirtustu"]
records = []

for style in classes:
    for video_path in sorted((DATA_DIR / style).glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Video acilamadi: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        records.append({
            "video_id": f"V{len(records)+1:03d}",
            "filename": video_path.name,
            "path": str(video_path.relative_to(PROJECT_ROOT)),
            "style": style,
            "duration_sec": round(frames / fps, 2) if fps > 0 else 0,
            "swimmer_id": "unknown",
            "session_id": "unknown",
            "style_verified": "pending",
            "split": "unassigned"
        })

df = pd.DataFrame(records)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(df[["video_id", "filename", "style", "duration_sec"]])
print(f"\nToplam video: {len(df)}")
print(f"Kaydedildi: {OUTPUT}")
