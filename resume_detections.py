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

done = 0
skipped = 0
failed = []

for i, video in enumerate(videos, 1):
    category = video.parent.name
    target = output_root / category / video.stem

    if target.exists() and any(target.glob("*.avi")):
        print(f"[{i}/{len(videos)}] VAR, ATLANDI: {video.name}", flush=True)
        skipped += 1
        continue

    try:
        print(f"[{i}/{len(videos)}] ISLENIYOR: {category}/{video.name}", flush=True)
        for _ in model.predict(
            source=str(video),
            classes=[0],
            conf=0.15,
            imgsz=640,
            save=True,
            project=str(output_root / category),
            name=video.stem,
            exist_ok=True,
            stream=True,
            verbose=False
        ):
            pass
        done += 1
    except Exception as e:
        failed.append((str(video), str(e)))
        print(f"HATA: {video.name}: {e}", flush=True)

print("\n=== OZET ===")
print("Toplam:", len(videos))
print("Bu sefer tamamlanan:", done)
print("Mevcut olduğu için atlanan:", skipped)
print("Hatalı:", len(failed))
for video, error in failed:
    print("HATA:", video, error)
