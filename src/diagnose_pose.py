from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pose_pipeline.extractors import MEDIAPIPE_NAMES, YOLO_NAMES, PoseExtractor, _common
from pose_pipeline.quality import landmark_quality, pose_is_usable, roi_to_original
from pose_pipeline.schema import Landmark
from pose_pipeline.video import draw_pose


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "pose_diagnostics"
VIDEOS = {
    "V001_single": ROOT / "data" / "single" / "kelebek" / "kelebek1.mp4",
    "kelebek_multi": ROOT / "data" / "raw" / "kelebek" / "kelebek.mp4",
}
V001_ROI = (0.30, 0.00, 0.75, 0.55)
QUALITY_THRESHOLD = 0.5


def sample_frames(path: Path, count: int = 20) -> list[dict[str, object]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    samples = []
    try:
        for position in (np.arange(count) + 0.5) / count:
            frame_number = round((frame_count - 1) * position)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Frame {frame_number} could not be read from {path}")
            samples.append({"frame_number": frame_number, "timestamp_sec": frame_number / fps, "frame": frame})
    finally:
        cap.release()
    return samples


def crop_frame(frame: np.ndarray, roi: tuple[float, float, float, float] | None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    if roi is None:
        return frame, (0, 0, width, height)
    x1, y1, x2, y2 = roi
    pixels = (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))
    left, top, right, bottom = pixels
    return frame[top:bottom, left:right], pixels


def save_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def contact_sheet(path: Path, samples: list[dict[str, object]], instances: list[list[tuple[dict[str, Landmark | None], bool]]] | None = None) -> None:
    tiles = []
    for index, sample in enumerate(samples):
        frame = sample["frame"].copy()
        if instances is not None:
            for common, usable in instances[index]:
                if usable:
                    frame = draw_pose(frame, common, PoseExtractor, QUALITY_THRESHOLD)
                else:
                    height, width = frame.shape[:2]
                    for point in common.values():
                        if point is not None and np.isfinite((point.x, point.y)).all():
                            cv2.circle(frame, (round(point.x * width), round(point.y * height)), 3, (0, 140, 255), -1)
                label = "usable" if usable else "raw only"
                cv2.putText(frame, label, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0) if usable else (0, 140, 255), 2)
        cv2.putText(frame, f"frame {sample['frame_number']}  t={sample['timestamp_sec']:.2f}s", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiles.append(cv2.resize(frame, (384, 216)))
    sheet = np.vstack([np.hstack(tiles[row:row + 5]) for row in range(0, len(tiles), 5)])
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Could not write {path}")


def summarize(experiment: dict[str, object], frame_instances: list[list[tuple[dict[str, Landmark | None], bool]]], elapsed: float) -> dict[str, object]:
    raw_frames = sum(bool(instances) for instances in frame_instances)
    raw_instances = sum(len(instances) for instances in frame_instances)
    usable_frames = sum(any(usable for _, usable in instances) for instances in frame_instances)
    selected_usable = sum(
        (instances[0][1] if experiment["video_type"] == "single" else any(usable for _, usable in instances))
        for instances in frame_instances if instances
    )
    return {
        **experiment,
        "sampled_frames": len(frame_instances),
        "raw_detected_frames": raw_frames,
        "raw_pose_instances": raw_instances,
        "raw_detection_rate": raw_frames / len(frame_instances),
        "usable_pose_frames": usable_frames,
        "usable_pose_rate": usable_frames / len(frame_instances),
        "selected_usable_pose_frames": selected_usable,
        "selected_usable_pose_rate": selected_usable / len(frame_instances),
        "processing_seconds": round(elapsed, 3),
    }


def run_yolo(model: YOLO, video_id: str, samples: list[dict[str, object]], conf: float, imgsz: int,
             roi: tuple[float, float, float, float] | None = None) -> tuple[dict[str, object], list[list[tuple[dict[str, Landmark | None], bool]]]]:
    experiment = {"video_id": video_id, "video_type": "single" if video_id.startswith("V001") else "multi",
                  "model": "yolov8n-pose", "confidence": conf, "imgsz": imgsz,
                  "roi": "V001_top_center" if roi else "full", "num_poses": 4,
                  "quality_threshold": QUALITY_THRESHOLD}
    slug = f"yolo_conf{conf:g}_imgsz{imgsz}_{experiment['roi']}"
    folder = OUTPUT / video_id / slug
    folder.mkdir(parents=True, exist_ok=True)
    raw_records, usable_records, all_instances = [], [], []
    started = time.perf_counter()
    for sample in samples:
        frame = sample["frame"]
        input_frame, crop = crop_frame(frame, roi)
        result = model.predict(input_frame, conf=conf, imgsz=imgsz, max_det=4,
                               device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
        instances = []
        if result.keypoints is not None:
            xyn = result.keypoints.xyn.cpu().numpy()
            scores = result.keypoints.conf
            keypoint_scores = scores.cpu().numpy() if scores is not None else np.ones(xyn.shape[:2])
            box_scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(xyn))
            for person, (coords, point_scores) in enumerate(zip(xyn, keypoint_scores)):
                points = tuple(
                    roi_to_original(Landmark(name, float(xy[0]), float(xy[1]), confidence=float(score)), crop, frame.shape[1], frame.shape[0])
                    for name, xy, score in zip(YOLO_NAMES, coords, point_scores)
                )
                common = _common(points)
                usable = pose_is_usable(common, QUALITY_THRESHOLD)
                instance = {"box_confidence": float(box_scores[person]), "landmarks": [asdict(point) for point in points],
                            "usable": usable, "usable_common_landmarks": sum(landmark_quality(point) >= QUALITY_THRESHOLD for point in common.values() if point)}
                instances.append((common, usable))
                if usable:
                    usable_records.append({"frame_number": sample["frame_number"], "timestamp_sec": sample["timestamp_sec"], **instance})
                raw_records.append({"frame_number": sample["frame_number"], "timestamp_sec": sample["timestamp_sec"], **instance})
        all_instances.append(instances)
    row = summarize(experiment, all_instances, time.perf_counter() - started)
    save_jsonl(folder / "raw_outputs.jsonl", raw_records)
    save_jsonl(folder / "usable_outputs.jsonl", usable_records)
    contact_sheet(folder / "visual_comparison.jpg", samples, all_instances)
    return row, all_instances


def run_mediapipe(video_id: str, samples: list[dict[str, object]], threshold: float, num_poses: int,
                  roi: tuple[float, float, float, float] | None = None) -> tuple[dict[str, object], list[list[tuple[dict[str, Landmark | None], bool]]]]:
    import mediapipe as mp

    experiment = {"video_id": video_id, "video_type": "single" if video_id.startswith("V001") else "multi",
                  "model": "mediapipe_pose_landmarker", "confidence": threshold, "imgsz": "native",
                  "roi": "V001_top_center" if roi else "full", "num_poses": num_poses,
                  "quality_threshold": QUALITY_THRESHOLD}
    slug = f"mediapipe_threshold{threshold:g}_poses{num_poses}_{experiment['roi']}"
    folder = OUTPUT / video_id / slug
    folder.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models" / "pose_landmarker_full.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=threshold,
        min_pose_presence_confidence=threshold,
        min_tracking_confidence=threshold,
    )
    raw_records, usable_records, all_instances = [], [], []
    started = time.perf_counter()
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for sample in samples:
            frame = sample["frame"]
            input_frame, crop = crop_frame(frame, roi)
            rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(input_frame[:, :, ::-1]))
            result = landmarker.detect_for_video(rgb, round(sample["timestamp_sec"] * 1000))
            instances = []
            for pose in result.pose_landmarks:
                points = tuple(
                    roi_to_original(Landmark(MEDIAPIPE_NAMES[index], float(point.x), float(point.y), float(point.z),
                                             visibility=float(point.visibility), presence=float(point.presence)),
                                    crop, frame.shape[1], frame.shape[0])
                    for index, point in enumerate(pose)
                )
                common = _common(points)
                usable = pose_is_usable(common, QUALITY_THRESHOLD)
                instance = {"landmarks": [asdict(point) for point in points], "usable": usable,
                            "usable_common_landmarks": sum(landmark_quality(point) >= QUALITY_THRESHOLD for point in common.values() if point)}
                instances.append((common, usable))
                if usable:
                    usable_records.append({"frame_number": sample["frame_number"], "timestamp_sec": sample["timestamp_sec"], **instance})
                raw_records.append({"frame_number": sample["frame_number"], "timestamp_sec": sample["timestamp_sec"], **instance})
            all_instances.append(instances)
    row = summarize(experiment, all_instances, time.perf_counter() - started)
    save_jsonl(folder / "raw_outputs.jsonl", raw_records)
    save_jsonl(folder / "usable_outputs.jsonl", usable_records)
    contact_sheet(folder / "visual_comparison.jpg", samples, all_instances)
    return row, all_instances


def write_report(rows: list[dict[str, object]]) -> None:
    best = {}
    for row in rows:
        key = (row["video_id"], row["model"])
        score = (int(row["selected_usable_pose_frames"]), int(row["raw_detected_frames"]),
                 -int(row["imgsz"]) if str(row["imgsz"]).isdigit() else 0)
        if key not in best or score > best[key][0]:
            best[key] = (score, row)
    lines = [
        "# AquaMed pose detection pilot hata analizi", "",
        "## Doğrulanan ana neden", "",
        "Önceki `%0 pose_detection_rate`, 13-eklem dönüşümü veya 0.5 landmark kalite filtresinden kaynaklanmıyordu. "
        "Üretim kodu `detected=bool(native_landmarks)` kullandığı için model API'leri V001'de hiç ham pose döndürmemişti. "
        "V001 görsel örneklerinde yüzücü çoğunlukla çok küçük, uzak, yatay, bulanık ve köpük/ışık kırılmasıyla kısmen örtülü. "
        "COCO tabanlı YOLOv8-Pose ile genel amaçlı MediaPipe'ın alan dışı sualtı görüntüsünde başarısızlığı bu gözlemle tutarlı.", "",
        "Varsayılan V001 deneyinde YOLO (`conf=0.25`, `imgsz=640`) ve MediaPipe (üç eşik de 0.5) 20/20 karede sıfır "
        "ham aday üretti. Bu, kaybın ortak-13 dönüşümünden önce oluştuğunu doğrudan doğrular.", "",
        "## Boru hattı doğrulaması", "",
        "- MediaPipe girişi BGR'den contiguous RGB'ye çevrildi.",
        "- `VIDEO` modu kullanıldı; 20 örnek artan gerçek video timestamp'leriyle işlendi.",
        "- Tek yüzücü için `num_poses=1`, çok yüzücülü deney için `num_poses=4` kullanıldı.",
        "- Ham model pose'ları kalite filtresinden önce `raw_outputs.jsonl`, kullanılabilir olanlar sonra `usable_outputs.jsonl` içine yazıldı.",
        "- ROI sonuçları orijinal görüntünün normalize koordinatlarına geri dönüştürüldü.",
        "- Kullanılabilir pose: kalite >= 0.5 olan en az 6/13 ortak eklem ve en az bir omuz + bir kalça.", "",
        "## En iyi gözlenen ayarlar", "",
        "| Video | Model | conf | imgsz | ROI | Ham kare | Seçilen kullanılabilir kare | Oran |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for (_, _), (_, row) in sorted(best.items()):
        lines.append(f"| {row['video_id']} | {row['model']} | {row['confidence']} | {row['imgsz']} | {row['roi']} | "
                     f"{row['raw_detected_frames']}/20 | {row['selected_usable_pose_frames']}/20 | {float(row['selected_usable_pose_rate']):.1%} |")
    lines += [
        "", "Seçim sırası: önce kullanılabilir kare sayısı, sonra ham tespitli kare sayısı, eşitlikte daha küçük `imgsz`. "
        "Tek-yüzücülü videoda yalnızca en yüksek box-confidence adayı sayılmıştır. Bu hâlâ doğruluk ölçümü değildir.", "",
        "## Görsel doğrulama sonucu", "",
        "Düşük eşik ve büyük giriş, gerçek yüzücü üzerinde bazı doğru iskeletler üretirken kulvar ipi, yansıma ve köpük üzerinde "
        "çok sayıda insan-benzeri iskelet de üretti. Özellikle `conf=0.01` deneylerindeki yüksek kullanılabilir-pose oranları "
        "gerçek başarı değildir. Mevcut otomatik eklem-kalite kuralı bu yanlış pozitifleri reddedemediği için güvenilir bir "
        "üretim parametresi seçilmedi. Tam kare `conf=0.01, imgsz=1280` yalnızca yüksek-recall teşhis adayıdır. Sabit V001 "
        "ROI'si ham adayı 20/20'ye çıkardı ama en yüksek-confidence kullanılabilir adayı 1/20'ye düşürdü; çözüm olmadı.", "",
        "## GRU veri kalitesi", "",
        "Önceki V001 pilotunda iki model de 206 karenin hiçbirinde ham pose üretmedi. Bu nedenle her modeldeki 6 tam, "
        "30-adımlı pencerenin eksik landmark oranı %100 ve `training_usable=False` olmalıdır. NPZ üretimi artık düşük "
        "confidence/presence noktalarını maskeler ve pencere başına `missing_rate` ile `training_usable` kaydeder.", "",
        "## Öneri", "",
        "Tam veri setine geçmeyin. ROI yaklaşımı için yüzücüyü kulvar ipinden ayıran bir detector/tracker ve küçük bir doğrulama "
        "etiket seti gerekir; elle sabit V001 ROI'si genellenebilir değildir. Sonraki kontrollü adım, bu 40 karede insan bbox/pose "
        "ground truth işaretleyip precision/recall ölçmek; ardından gerekirse sualtı verisiyle detector veya pose modelini fine-tune etmektir.", "",
    ]
    (OUTPUT / "pose_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = {video_id: sample_frames(path) for video_id, path in VIDEOS.items()}
    for video_id, video_samples in samples.items():
        folder = OUTPUT / video_id
        folder.mkdir(parents=True, exist_ok=True)
        contact_sheet(folder / "source_samples.jpg", video_samples)

    model = YOLO(str(ROOT / "models" / "yolov8n-pose.pt"))
    rows = []
    for video_id, video_samples in samples.items():
        for conf in (0.25, 0.05, 0.01):
            for imgsz in (640, 960, 1280):
                row, _ = run_yolo(model, video_id, video_samples, conf, imgsz)
                rows.append(row)
    full_best = max((row for row in rows if row["video_id"] == "V001_single"),
                    key=lambda row: (row["selected_usable_pose_frames"], row["raw_detected_frames"], -row["imgsz"]))
    row, _ = run_yolo(model, "V001_single", samples["V001_single"], full_best["confidence"], full_best["imgsz"], V001_ROI)
    rows.append(row)

    for video_id, video_samples in samples.items():
        num_poses = 1 if video_id == "V001_single" else 4
        for threshold in (0.5, 0.1):
            row, _ = run_mediapipe(video_id, video_samples, threshold, num_poses)
            rows.append(row)
    row, _ = run_mediapipe("V001_single", samples["V001_single"], 0.1, 1, V001_ROI)
    rows.append(row)

    fields = list(rows[0])
    with (OUTPUT / "parameter_comparison.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows)
    print(f"Saved {len(rows)} experiments to {OUTPUT}")


if __name__ == "__main__":
    main()
