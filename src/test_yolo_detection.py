
from pathlib import Path

import torch
from ultralytics import YOLO

# Proje kok dizinini dosyanin konumundan bul
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "test_videos" / "1020(3)-1.mp4"
OUTPUT_DIR = PROJECT_ROOT / "results"

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Video bulunamadi: {VIDEO_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"

print(f"Kullanilan cihaz: {device}")
print(f"Proje klasoru: {PROJECT_ROOT}")

# YOLOv8 insan tespiti
model = YOLO("yolov8s.pt")

results = model.predict(
    source=str(VIDEO_PATH),
    device=device,
    classes=[0],
    conf=0.10,
    imgsz=960,
    save=True,
    project=str(OUTPUT_DIR),
    name="yolov8s_detection_test",
    exist_ok=True,
    stream=True,
    verbose=False,
)

detected_frames = 0
total_detections = 0

for result in results:
    count = len(result.boxes)
    total_detections += count

    if count > 0:
        detected_frames += 1

print("\nYOLO TEST SONUCLARI")
print("----------------------------")
print(f"Insan tespit edilen kare: {detected_frames}")
print(f"Toplam tespit: {total_detections}")
print(f"Kayit klasoru: {OUTPUT_DIR / 'yolov8s_detection_test'}")
