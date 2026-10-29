from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.video_classification.binary_drowning import CLASSES
from src.video_classification.mil_train import aggregate_bag_logits
from src.video_classification.model import build_r3d18
from src.video_classification.preprocessing import R3D_MEAN, R3D_STD, transform_clip

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "results/r3d18_binary_drowning/mil/fold_3/best_checkpoint.pt"
OUTPUT_ROOT = ROOT / "results/drowning_demo"
WARNING = (
    "Skorlar kalibre edilmiş boğulma olasılığı değildir. Klip skorları kesin olay "
    "konumları değildir; model zayıf video etiketleriyle eğitilmiştir. Skor modelin "
    "merkez kırpımı üzerinden video bölgesine aittir; kişi tespiti ve takibi yoktur. "
    "Bu demo gerçek can güvenliği alarm sistemi değildir."
)
LABELS = ["Normal yüzme", "Boğulma belirtisi"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_checkpoints() -> list[Path]:
    return sorted((ROOT / "results/r3d18_binary_drowning/mil").glob("fold_*/best_checkpoint.pt"))


def new_output_dir() -> Path:
    path = OUTPUT_ROOT / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_model(checkpoint_path: Path, device: str):
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Yalnız CPU veya CUDA seçilebilir.")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA bu ortamda kullanılamıyor; CPU seçin.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    expected = {"model": "torchvision.models.video.r3d_18", "clip_length": 16,
                "sampling_fps": 8.0, "input_size": [112, 112]}
    if checkpoint.get("classes") != CLASSES or config.get("classes") != CLASSES:
        raise ValueError("Checkpoint sınıf sırası eğitim sözleşmesiyle uyuşmuyor.")
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint mimarisi veya preprocessing ayarları desteklenmiyor.")
    strategy = config.get("training_strategy", {})
    if strategy.get("name") != "multiple_instance_learning" or strategy.get("pooling") != "top_k_mean_clip_logits":
        raise ValueError("Checkpoint beklenen top-k MIL modeli değil.")
    if not isinstance(config.get("top_k"), int) or config["top_k"] <= 0:
        raise ValueError("Checkpoint top_k değeri geçersiz.")
    model = build_r3d18(2, pretrained=False, freeze_backbone=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(target).eval()
    return model, config


def decoded_frames(path: Path):
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Dosyada video akışı yok.")
        stream = container.streams.video[0]
        stream.codec_context.options = {"err_detect": "explode"}
        yield from container.decode(stream)


def inspect_video(path: Path) -> dict:
    timestamps = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Dosyada video akışı yok.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or not math.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError("Geçerli video FPS bilgisi bulunamadı.")
        declared_frames = stream.frames
        has_audio = bool(container.streams.audio)
    width = height = 0
    last_duration = 0.0
    for frame in decoded_frames(path):
        if frame.pts is None or frame.time_base is None:
            raise ValueError("Video zaman damgaları eksik; güvenilir zaman hizalaması yapılamıyor.")
        timestamp = float(frame.pts * frame.time_base)
        if not math.isfinite(timestamp) or (timestamps and timestamp <= timestamps[-1]):
            raise ValueError("Video zaman damgaları artan sırada değil.")
        if timestamps and (width, height) != (frame.width, frame.height):
            raise ValueError("Video içinde çözünürlük değişiyor; sabit çözünürlüklü MP4 kullanın.")
        width, height = frame.width, frame.height
        timestamps.append(timestamp)
        last_duration = float(frame.duration * frame.time_base) if frame.duration else 0.0
    if not timestamps:
        raise ValueError("Video boş veya kareleri çözülemiyor.")
    if declared_frames and len(timestamps) != declared_frames:
        raise ValueError(f"Video eksik/bozuk: {declared_frames} yerine {len(timestamps)} kare çözüldü.")
    origin = timestamps[0]
    timestamps = [value - origin for value in timestamps]
    tail = last_duration or (float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1 / float(rate))
    return {"fps": float(rate), "rate_numerator": rate.numerator, "rate_denominator": rate.denominator,
            "frame_count": len(timestamps), "width": width, "height": height,
            "duration_sec": timestamps[-1] + tail, "timestamps_sec": timestamps,
            "source_pts_origin_sec": origin, "has_audio": has_audio}


def plan_windows(info: dict) -> tuple[list[dict], int]:
    # Match training's rounded source-frame sampling, including non-integer FPS.
    step = max(1, round(info["fps"] / 8.0))
    span = 15 * step + 1
    count = info["frame_count"]
    last_start = max(0, count - span)
    starts = list(range(0, last_start + 1, 8 * step))
    if starts[-1] != last_start:
        starts.append(last_start)
    times = info["timestamps_sec"]
    records = []
    for index, start in enumerate(starts):
        indices = [min(start + offset * step, count - 1) for offset in range(16)]
        end = indices[-1]
        records.append({"clip_id": index, "start_frame": start, "end_frame": end,
                        "start_sec": times[start], "last_sample_sec": times[end],
                        "end_sec": times[end + 1] if end + 1 < count else info["duration_sec"],
                        "sample_frame_indices": indices, "sample_timestamps_sec": [times[i] for i in indices],
                        "padded_frames": len(indices) - len(set(indices))})
    return records, step


def infer_windows(path: Path, info: dict, records: list[dict], model, device: str, progress=None) -> torch.Tensor:
    needed = {index for record in records for index in record["sample_frame_indices"]}
    cache, logits = {}, []
    current, index = 0, -1
    with torch.inference_mode():
        for index, frame in enumerate(decoded_frames(path)):
            if index >= info["frame_count"]:
                raise ValueError("Analiz sırasında kaynak kare sayısı değişti.")
            timestamp = float(frame.pts * frame.time_base) - info["source_pts_origin_sec"]
            if abs(timestamp - info["timestamps_sec"][index]) > 1e-6:
                raise ValueError("Analiz sırasında kaynak zaman damgaları değişti.")
            if index in needed:
                # Transform once per sampled frame; identical deterministic training transform.
                cache[index] = transform_clip(frame.to_ndarray(format="rgb24")[None])[:, 0]
            while current < len(records) and records[current]["end_frame"] == index:
                record = records[current]
                clip = torch.stack([cache[i] for i in record["sample_frame_indices"]], dim=1)
                output = model(clip.unsqueeze(0).to(device)).float().cpu()
                if output.shape != (1, 2) or not torch.isfinite(output).all():
                    raise ValueError("Model geçersiz klip çıktısı üretti.")
                logits.append(output[0])
                current += 1
                keep_from = records[current]["start_frame"] if current < len(records) else index + 1
                cache = {i: value for i, value in cache.items() if i >= keep_from}
                if progress:
                    progress(0.15 + 0.55 * current / len(records), "Model pencereleri analiz ediyor…")
    if current != len(records) or index + 1 != info["frame_count"]:
        raise ValueError("Analiz sırasında tüm kareler çözülemedi.")
    return torch.stack(logits)


@lru_cache(maxsize=16)
def overlay_font(size: int):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise ValueError("Türkçe karakter destekleyen Arial veya DejaVu Sans fontu bulunamadı.")


def draw_overlay(frame: av.VideoFrame, record: dict, timestamp: float, duration: float) -> Image.Image:
    image = frame.to_image()
    width, height = image.size
    # Draw on a readable logical canvas, then scale to fit even small source videos.
    panel_width = max(760, width)
    font_size = max(18, round(panel_width / 70))
    line_height = round(font_size * 1.4)
    panel = Image.new("RGB", (panel_width, line_height * 6 + 30), (12, 25, 37))
    draw = ImageDraw.Draw(panel)
    font = overlay_font(font_size)
    color = (255, 169, 72) if record["drowning_behavior_score"] >= 0.5 else (90, 225, 195)
    lines = ["AquaMed | " + LABELS[int(record["drowning_behavior_score"] >= 0.5)],
             f"Boğulma davranışı skoru: {record['drowning_behavior_score']:.3f}",
             f"Analiz aralığı: {record['start_sec']:.2f}–{record['end_sec']:.2f} sn | Video: {timestamp:.2f} sn",
             "Video bölgesi skoru · Kişi tespiti / takibi yok",
             "Kalibre olasılık / kesin olay konumu değildir",
             "Deneysel demo · Can güvenliği alarmı değildir"]
    for row, text in enumerate(lines):
        draw.text((12, row * line_height), text, font=font, fill=color if row < 2 else "white")
    bar_y = panel.height - 19
    draw.rectangle((12, bar_y, panel_width - 12, bar_y + 8), fill=(60, 75, 85))
    left = 12 + int((panel_width - 24) * record["start_sec"] / duration)
    right = 12 + int((panel_width - 24) * record["end_sec"] / duration)
    draw.rectangle((left, bar_y, right, bar_y + 8), fill=color)
    cursor = 12 + int((panel_width - 24) * min(timestamp / duration, 1))
    draw.line((cursor, bar_y - 3, cursor, bar_y + 12), fill="white", width=3)
    scale = min(width / panel.width, height * 0.48 / panel.height)
    panel = panel.resize((max(1, round(panel.width * scale)), max(1, round(panel.height * scale))), Image.Resampling.LANCZOS)
    image.paste(panel, (0, 0))
    return image


def render_video(source: Path, target: Path, info: dict, records: list[dict], progress=None) -> None:
    centers = [(row["start_sec"] + row["end_sec"]) / 2 for row in records]
    boundaries = (np.array(centers[:-1]) + np.array(centers[1:])) / 2
    rate = Fraction(info["rate_numerator"], info["rate_denominator"])
    time_base = Fraction(1, 1000000)
    with av.open(str(target), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = info["width"], info["height"]
        stream.pix_fmt = "yuv420p" if info["width"] % 2 == info["height"] % 2 == 0 else "yuv444p"
        stream.time_base = stream.codec_context.time_base = time_base
        stream.options = {"crf": "20", "preset": "veryfast", "bf": "0"}
        for index, frame in enumerate(decoded_frames(source)):
            if index >= info["frame_count"]:
                raise ValueError("Dışa aktarım sırasında kaynak değişti.")
            timestamp = info["timestamps_sec"][index]
            actual = float(frame.pts * frame.time_base) - info["source_pts_origin_sec"]
            if abs(actual - timestamp) > 1e-6:
                raise ValueError("Dışa aktarım zaman hizalaması değişti.")
            record = records[int(np.searchsorted(boundaries, timestamp, side="right"))]
            annotated = av.VideoFrame.from_image(draw_overlay(frame, record, timestamp, info["duration_sec"]))
            annotated.pts = round(timestamp / time_base)
            annotated.time_base = time_base
            annotated.duration = max(1, round(((info["timestamps_sec"][index + 1] if index + 1 < info["frame_count"]
                                               else info["duration_sec"]) - timestamp) / time_base))
            for packet in stream.encode(annotated):
                container.mux(packet)
            if progress and index % 30 == 0:
                progress(0.7 + 0.25 * (index + 1) / info["frame_count"], "İşlenmiş MP4 oluşturuluyor…")
        for packet in stream.encode():
            container.mux(packet)
    exported = inspect_video(target)
    if any(exported[key] != info[key] for key in ("width", "height", "frame_count")):
        raise ValueError("MP4 doğrulaması başarısız: çözünürlük veya kare sayısı değişti.")
    if not np.allclose(exported["timestamps_sec"], info["timestamps_sec"], atol=0.0001, rtol=0):
        raise ValueError("MP4 doğrulaması başarısız: zaman damgaları kaydı.")


def save_timeline(path: Path, records: list[dict], mil_score: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 3.8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    centers = [(row["start_sec"] + row["end_sec"]) / 2 for row in records]
    scores = [row["drowning_behavior_score"] for row in records]
    axis.hlines(scores, [row["start_sec"] for row in records], [row["end_sec"] for row in records],
                alpha=0.3, color="teal", label="Pencere aralığı")
    axis.plot(centers, scores, "o-", color="teal", label="Klip skoru")
    axis.axhline(0.5, color="gray", linestyle=":", label="Demo eşiği 0.5 (kalibre değil)")
    axis.set(xlabel="Video zamanı (sn)", ylabel="Boğulma davranışı skoru", ylim=(-0.03, 1.03),
             title=f"AquaMed · Video MIL skoru: {mil_score:.3f} · Kesin olay konumu değildir")
    axis.legend(loc="best", fontsize=8)
    figure.savefig(path, dpi=140)


def analyze_video(source: Path, checkpoint: Path = DEFAULT_CHECKPOINT, device: str = "cpu",
                  output: Path | None = None, progress=None) -> dict:
    source, checkpoint = source.resolve(), checkpoint.resolve()
    output = new_output_dir() if output is None else output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or not output.is_dir():
        raise ValueError("Çıktı klasörü results/drowning_demo altında yeni bir analiz klasörü olmalı.")
    if any((output / name).exists() for name in ("analysis_summary.json", "clip_scores.csv", "annotated_video.mp4", "score_timeline.png")):
        raise ValueError("Mevcut analiz çıktılarının üzerine yazılmaz.")
    partial = output / "annotated_video.partial.mp4"
    if source in {partial, output / "annotated_video.mp4"}:
        raise ValueError("Kaynak video çıktı dosyası olamaz.")
    try:
        if progress:
            progress(0.02, "Checkpoint yükleniyor…")
        source_hash, checkpoint_hash = sha256(source), sha256(checkpoint)
        model, config = load_model(checkpoint, device)
        if progress:
            progress(0.07, "Video ve zaman damgaları doğrulanıyor…")
        info = inspect_video(source)
        records, frame_step = plan_windows(info)
        logits = infer_windows(source, info, records, model, device, progress)
        bag_logit, top_indices = aggregate_bag_logits(logits, config["top_k"])
        scores = (logits[:, 1] - logits[:, 0]).sigmoid().tolist()
        for row, values, score in zip(records, logits.tolist(), scores):
            row.update({"normal_logit": values[0], "drowning_logit": values[1],
                        "drowning_behavior_score": score, "predicted_label": CLASSES[int(score >= 0.5)],
                        "selected_for_mil": row["clip_id"] in top_indices.tolist()})
        mil_score = float(bag_logit.sigmoid())
        render_video(source, partial, info, records, progress)
        save_timeline(output / "score_timeline.png", records, mil_score)
        with (output / "clip_scores.csv").open("w", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
                            for row in records)
        summary = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "source": str(source), "source_sha256": source_hash, "checkpoint": str(checkpoint),
                   "checkpoint_sha256": checkpoint_hash, "device": device, "classes": CLASSES,
                   "video": info, "preprocessing": {"clip_length": 16, "sampling_fps": 8.0,
                   "frame_step": frame_step, "effective_sampling_fps": info["fps"] / frame_step,
                   "stride_sampled_frames": 8, "resize_short_side": 128, "center_crop": [112, 112],
                   "mean": R3D_MEAN, "std": R3D_STD, "color": "RGB",
                   "sampling_rule": "training: max(1, round(source_fps / 8)); short clips repeat last frame"},
                   "mil": {"drowning_behavior_score": mil_score, "bag_logit": float(bag_logit),
                   "top_k": config["top_k"], "selected_clip_ids": top_indices.tolist(),
                   "aggregation": "sigmoid(mean(top_k(drowning_logit - normal_logit)))",
                   "bag": "all demo overlapping windows; not the capped validation bag"},
                   "clip_count": len(records), "short_video_padded": any(row["padded_frames"] for row in records),
                   "display_threshold": 0.5, "overlay_selection": "nearest window center; no event localization",
                   "time_basis": "decoded presentation timestamps relative to first frame; end_sec exclusive",
                   "output_dir": str(output), "audio_preserved": False,
                   "browser_note": "Odd source dimensions require H.264 yuv444p; some browsers may not play it.",
                   "warning": WARNING, "versions": {"torch": str(torch.__version__), "av": av.__version__}}
        partial.replace(output / "annotated_video.mp4")
        (output / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress(1.0, "Analiz tamamlandı.")
        return summary
    except Exception as error:
        partial.unlink(missing_ok=True)
        (output / "error.json").write_text(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description="AquaMed inference-only video demo")
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    result = analyze_video(args.video, args.checkpoint, args.device,
                           progress=lambda fraction, text: print(f"{fraction:.0%} {text}", flush=True))
    print(result["output_dir"])


if __name__ == "__main__":
    main()
