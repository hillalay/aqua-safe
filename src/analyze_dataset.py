
from pathlib import Path
import cv2
import pandas as pd

ROOT = Path("data/raw")
CLASSES = ["serbest", "sirtustu", "kurbagalama", "kelebek", "bogulma"]

records = []

for class_name in CLASSES:
    folder = ROOT / class_name

    if not folder.exists():
        print(f"Klasor bulunamadi: {folder}")
        continue

    for path in sorted(folder.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            records.append({
                "filename": path.name,
                "class": class_name,
                "status": "ERROR"
            })
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        success, _ = cap.read()
        cap.release()

        duration = frames / fps if fps > 0 else 0

        records.append({
            "filename": path.name,
            "class": class_name,
            "fps": round(fps, 2),
            "frames": frames,
            "duration_sec": round(duration, 2),
            "width": width,
            "height": height,
            "status": "OK" if success and fps > 0 else "ERROR"
        })

df = pd.DataFrame(records)

Path("results").mkdir(exist_ok=True)
df.to_csv("results/dataset_inventory.csv", index=False)

print("\nVIDEO ENVANTERI")
print(df.to_string(index=False))

print("\nSINIF OZETI")
if not df.empty:
    print(df.groupby("class", dropna=False).agg(
        video_count=("filename", "count"),
        total_duration_sec=("duration_sec", "sum"),
        avg_duration_sec=("duration_sec", "mean")
    ).round(2))

    print("\nHATALI VIDEO SAYISI:",
          (df["status"] != "OK").sum())

print("\nKaydedildi: results/dataset_inventory.csv")
