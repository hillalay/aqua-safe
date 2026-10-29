from pathlib import Path
from ultralytics import YOLO

root = Path.cwd()
source_root = root / "data" / "raw"
output_root = root / "results" / "detect_all"
model = YOLO(str(root / "yolov8n.pt"))

videos = sorted(
    p for p in source_root.rglob("*")
    if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
)

print(f"Toplam kaynak video: {len(videos)}")
done = 0
failed = []

for i, video in enumerate(videos, 1):
    category = video.parent.name
    target = output_root / category / video.stem

    if target.exists() and any(target.glob("*.avi")):
        print(f"[{i}/{len(videos)}] ATLANDI: {video.name}")
        continue

    try:
        model.predict(
            source=str(video),
            classes=[0],
            conf=0.15,
            imgsz=960,
            save=True,
            project=str(output_root / category),
            name=video.stem,
            exist_ok=True,
            verbose=False
        )
        done += 1
        print(f"[{i}/{len(videos)}] TAMAM: {category}/{video.name}")
    except Exception as e:
        failed.append((str(video), str(e)))
        print(f"[{i}/{len(videos)}] HATA: {video.name}: {e}")

print("\n=== OZET ===")
print(f"Kaynak video: {len(videos)}")
print(f"Bu calistirmada islenen: {done}")
print(f"Hata: {len(failed)}")
for video, error in failed:
    print(f"HATA: {video} -> {error}")
