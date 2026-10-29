from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .schema import COMMON_LANDMARKS, Landmark


MEDIAPIPE_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)

YOLO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


class PoseExtractor(ABC):
    name: str
    version: str
    connections: tuple[tuple[str, str], ...] = (
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    )

    @abstractmethod
    def extract(self, bgr_frame: np.ndarray, timestamp_ms: int) -> tuple[tuple[Landmark, ...], dict[str, Landmark | None]]:
        """Return native landmarks and the shared subset; empty native means no pose."""

    def close(self) -> None:
        pass

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _common(points: tuple[Landmark, ...]) -> dict[str, Landmark | None]:
    by_name = {point.name: point for point in points}
    return {name: by_name.get(name) for name in COMMON_LANDMARKS}


class MediaPipeExtractor(PoseExtractor):
    name = "mediapipe_pose_landmarker"

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency: pip install mediapipe") from exc

        self._mp = mp
        self.version = getattr(mp, "__version__", "unknown")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def extract(self, bgr_frame: np.ndarray, timestamp_ms: int) -> tuple[tuple[Landmark, ...], dict[str, Landmark | None]]:
        rgb = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(bgr_frame[:, :, ::-1]),
        )
        result = self._landmarker.detect_for_video(rgb, timestamp_ms)
        if not result.pose_landmarks:
            return (), _common(())
        points = tuple(
            Landmark(
                name=MEDIAPIPE_NAMES[index], x=float(point.x), y=float(point.y),
                z=float(point.z), visibility=float(point.visibility), presence=float(point.presence),
            )
            for index, point in enumerate(result.pose_landmarks[0])
        )
        return points, _common(points)

    def close(self) -> None:
        self._landmarker.close()


class YoloPoseExtractor(PoseExtractor):
    name = "ultralytics_yolov8_pose"

    def __init__(self, weights: Path, device: str = "0"):
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO pose weights not found: {weights}")
        import ultralytics
        from ultralytics import YOLO

        self.version = ultralytics.__version__
        self._model = YOLO(str(weights))
        self._device = device

    def extract(self, bgr_frame: np.ndarray, timestamp_ms: int) -> tuple[tuple[Landmark, ...], dict[str, Landmark | None]]:
        del timestamp_ms
        result = self._model.predict(bgr_frame, device=self._device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return (), _common(())
        person = int(result.boxes.conf.argmax().item()) if result.boxes is not None else 0
        xy = result.keypoints.xyn[person].cpu().numpy()
        confidence = result.keypoints.conf
        conf = confidence[person].cpu().numpy() if confidence is not None else np.ones(len(xy))
        points = tuple(
            Landmark(name=name, x=float(coords[0]), y=float(coords[1]), confidence=float(score))
            for name, coords, score in zip(YOLO_NAMES, xy, conf)
        )
        return points, _common(points)
