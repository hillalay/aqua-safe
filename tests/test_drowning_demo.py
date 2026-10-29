import csv
import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import torch

from src.drowning_demo.analysis import (
    DEFAULT_CHECKPOINT, ROOT, analyze_video, available_checkpoints, decoded_frames,
    infer_windows, inspect_video, load_model, new_output_dir, plan_windows,
    render_video, sha256,
)
from src.video_classification.preprocessing import probe_video, transform_clip


def make_video(path, count=32, width=160, height=120, variable=False):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {"bf": "0"}
        pts = 0
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), index * 5 % 256, np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
            pts += 100 if variable and index % 2 else 150 if variable else 125
        for packet in stream.encode():
            container.mux(packet)


class MeanModel(torch.nn.Module):
    def forward(self, batch):
        mean = batch.mean(dim=(1, 2, 3, 4))
        return torch.stack((-mean, mean), dim=1)


class DemoTests(unittest.TestCase):
    def test_window_frame_and_timestamp_alignment(self):
        times = [index / 30 for index in range(255)]
        records, step = plan_windows({"fps": 30, "frame_count": 255,
                                     "timestamps_sec": times, "duration_sec": 8.5})
        self.assertEqual(step, 4)
        self.assertEqual(records[0]["sample_frame_indices"], list(range(0, 61, 4)))
        self.assertEqual(records[1]["start_frame"], 32)
        self.assertEqual(records[-1]["end_frame"], 254)
        self.assertEqual(records[-1]["end_sec"], 8.5)
        for record in records:
            self.assertEqual(record["sample_timestamps_sec"], [times[i] for i in record["sample_frame_indices"]])

    def test_short_one_frame_inference_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "short.mp4"
            make_video(source, count=1)
            info = inspect_video(source)
            records, _ = plan_windows(info)
            self.assertEqual(records[0]["padded_frames"], 15)
            logits = infer_windows(source, info, records, MeanModel(), "cpu")
            self.assertEqual(tuple(logits.shape), (1, 2))
            records[0]["drowning_behavior_score"] = 0.25
            target = Path(folder) / "out.mp4"
            render_video(source, target, info, records)
            result = inspect_video(target)
            self.assertEqual(result["frame_count"], 1)
            self.assertEqual((result["width"], result["height"]), (160, 120))

    def test_vfr_export_and_odd_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "vfr.mp4"
            make_video(source, count=18, width=161, height=121, variable=True)
            info = inspect_video(source)
            self.assertGreater(np.ptp(np.diff(info["timestamps_sec"])), 0.02)
            records, _ = plan_windows(info)
            for record in records:
                record["drowning_behavior_score"] = 0.7
            target = Path(folder) / "out.mp4"
            render_video(source, target, info, records)
            exported = inspect_video(target)
            self.assertEqual((exported["width"], exported["height"]), (161, 121))
            np.testing.assert_allclose(exported["timestamps_sec"], info["timestamps_sec"], atol=0.0001)

    def test_streaming_transform_and_logits_equal_training_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "normal.mp4"
            make_video(source)
            info = inspect_video(source)
            records, _ = plan_windows(info)
            model = MeanModel()
            actual = infer_windows(source, info, records, model, "cpu")
            frames = [frame.to_ndarray(format="rgb24") for frame in decoded_frames(source)]
            expected = torch.cat([model(transform_clip(np.stack([frames[i] for i in row["sample_frame_indices"]])).unsqueeze(0))
                                  for row in records])
            torch.testing.assert_close(actual, expected)

    def test_bad_and_truncated_videos(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "bad.mp4"
            for content in (b"", b"this is not an mp4"):
                source.write_bytes(content)
                with self.assertRaises((ValueError, av.FFmpegError)):
                    inspect_video(source)
            make_video(source)
            source.write_bytes(source.read_bytes()[:100])
            with self.assertRaises((ValueError, av.FFmpegError)):
                inspect_video(source)

    def test_checkpoint_contract_rejects_wrong_classes_and_preprocessing(self):
        base = {"classes": ["normal_swimming", "drowning_behavior"],
                "config": {"classes": ["normal_swimming", "drowning_behavior"],
                           "model": "torchvision.models.video.r3d_18", "clip_length": 16,
                           "sampling_fps": 8., "input_size": [112, 112]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pt"
            torch.save({**base, "classes": list(reversed(base["classes"]))}, path)
            with self.assertRaisesRegex(ValueError, "sınıf"):
                load_model(path, "cpu")
            torch.save({**base, "config": {**base["config"], "clip_length": 8}}, path)
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                load_model(path, "cpu")

    def test_failure_has_no_finished_mp4_and_never_overwrites_input(self):
        output = new_output_dir()
        source = output / "input.mp4"
        source.write_bytes(b"invalid input")
        before = sha256(source)
        with patch("src.drowning_demo.analysis.load_model", return_value=(MeanModel(), {"top_k": 2})):
            with self.assertRaises((ValueError, av.FFmpegError)):
                analyze_video(source, DEFAULT_CHECKPOINT, output=output)
        self.assertEqual(before, sha256(source))
        self.assertFalse((output / "annotated_video.mp4").exists())
        self.assertEqual(json.loads((output / "error.json").read_text(encoding="utf-8"))["status"], "failed")

@unittest.skipUnless(os.environ.get("AQUAMED_SMOKE_VIDEO"), "Set AQUAMED_SMOKE_VIDEO for real checkpoint/video smoke")
class RealCheckpointSmoke(unittest.TestCase):
    def test_real_video_all_folds_cpu_cuda_and_exports(self):
        source = ROOT / os.environ["AQUAMED_SMOKE_VIDEO"]
        checkpoints = available_checkpoints()
        self.assertEqual(len(checkpoints), 5)
        before = {str(path): sha256(path) for path in [source, *checkpoints]}
        label_paths = [ROOT / "results/r3d18_binary_drowning/binary_metadata.csv",
                       ROOT / "results/video_review/review_state.json"]
        before_labels = {str(path): sha256(path) for path in label_paths if path.exists()}
        info = inspect_video(source)
        self.assertAlmostEqual(info["fps"], probe_video(source).fps, places=3)
        records, _ = plan_windows(info)
        indices = records[0]["sample_frame_indices"]
        frames = {i: frame.to_ndarray(format="rgb24") for i, frame in enumerate(decoded_frames(source)) if i in indices}
        tensor = transform_clip(np.stack([frames[i] for i in indices])).unsqueeze(0)
        fold_scores = {}
        cuda_logit_max_difference = None
        for checkpoint in checkpoints:
            model, _ = load_model(checkpoint, "cpu")
            with torch.inference_mode():
                cpu = model(tensor)
                self.assertTrue(torch.isfinite(cpu).all())
                fold_scores[checkpoint.parent.name] = float((cpu[0, 1] - cpu[0, 0]).sigmoid())
                if checkpoint == DEFAULT_CHECKPOINT and torch.cuda.is_available():
                    cuda = model.to("cuda")(tensor.to("cuda")).cpu()
                    cuda_logit_max_difference = float((cpu - cuda).abs().max())
                    torch.testing.assert_close(cpu, cuda, atol=1e-3, rtol=1e-3)
        self.assertGreater(len(set(fold_scores.values())), 1)
        result = analyze_video(source, DEFAULT_CHECKPOINT, "cpu")
        output = Path(result["output_dir"])
        for name in ("annotated_video.mp4", "analysis_summary.json", "clip_scores.csv", "score_timeline.png"):
            self.assertGreater((output / name).stat().st_size, 0)
        exported = inspect_video(output / "annotated_video.mp4")
        for key in ("width", "height", "frame_count"):
            self.assertEqual(exported[key], info[key])
        np.testing.assert_allclose(exported["timestamps_sec"], info["timestamps_sec"], atol=0.0001, rtol=0)
        with (output / "clip_scores.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), result["clip_count"])
        differences = sorted((float(row["drowning_logit"]) - float(row["normal_logit"]) for row in rows), reverse=True)
        expected = torch.tensor(differences[:result["mil"]["top_k"]]).mean().sigmoid().item()
        self.assertAlmostEqual(expected, result["mil"]["drowning_behavior_score"], places=6)
        for path, digest in {**before, **before_labels}.items():
            self.assertEqual(sha256(Path(path)), digest)
        report = {"real_video": str(source), "output_dir": str(output), "fold_cpu_scores": fold_scores,
                  "cuda_checked": torch.cuda.is_available(), "frame_count": info["frame_count"],
                  "cpu_cuda_max_logit_difference": cuda_logit_max_difference,
                  "resolution": [info["width"], info["height"]], "mil_score": expected,
                  "source_checkpoints_and_labels_unchanged": True,
                  "direct_mp4_export": "passed"}
        (output / "smoke_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("REAL_SMOKE", json.dumps(report), flush=True)


if __name__ == "__main__":
    unittest.main()
