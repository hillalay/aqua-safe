
from pathlib import Path
import cv2

video_path = Path("data/raw/serbest/serbest_01.mp4")

if not video_path.is_file():
    raise FileNotFoundError(f"Video bulunamadi: {video_path}")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Video acilamadi.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("VIDEO BILGILERI")
print("------------------------")
print(f"FPS: {fps}")
print(f"Frame sayisi: {frame_count}")
print(f"Cozunurluk: {width}x{height}")

if fps > 0:
    print(f"Sure: {frame_count / fps:.2f} saniye")

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Ilk frame okunamadi.")

output_path = Path("data/processed/first_frame.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output_path), frame):
    raise RuntimeError("Frame kaydedilemedi.")

print(f"Ilk frame kaydedildi: {output_path}")
