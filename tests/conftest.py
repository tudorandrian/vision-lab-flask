"""Shared fixtures. The fake models make the web layer testable in milliseconds, offline."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient
from PIL import Image as PilImage

from vision_lab.app import create_app
from vision_lab.config import Settings
from vision_lab.inference import Detection, FaceEmotion, ModelRegistry

VOC = ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
       "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
       "sheep", "sofa", "train", "tvmonitor"]  # fmt: skip


class FakeDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        return [Detection("person", 0.91, (2, 2, width // 2, height // 2))]


class FakeSegmenter:
    class_names = VOC

    def segment(self, image: np.ndarray) -> np.ndarray:
        class_map = np.zeros(image.shape[:2], dtype=np.uint8)
        class_map[: image.shape[0] // 2, :] = 15  # top half is "person"
        return class_map


class FakeEmotionAnalyzer:
    def analyze(self, image: np.ndarray) -> list[FaceEmotion]:
        return [FaceEmotion((4, 4, 20, 20), "neutral", {"neutral": 88.5, "happy": 11.5})]


class FakeRegistry(ModelRegistry):
    """Same interface as the real registry, no torch import, no downloads."""

    def __init__(self, weights_dir: Path, *, emotion: bool = False, fail: bool = False) -> None:
        super().__init__(weights_dir, enable_emotion=emotion)
        self._fail = fail

    def available_detectors(self) -> list[str]:
        return ["fasterrcnn"]

    def emotion_available(self) -> bool:
        return self.enable_emotion

    def detector(self, name: str) -> FakeDetector:
        if self._fail:
            raise RuntimeError("model exploded")
        return FakeDetector()

    def segmenter(self) -> FakeSegmenter:
        return FakeSegmenter()

    def emotion_analyzer(self) -> FakeEmotionAnalyzer | None:
        return FakeEmotionAnalyzer() if self.enable_emotion else None


def png_bytes(
    width: int = 64, height: int = 48, color: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    buffer = io.BytesIO()
    PilImage.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        max_upload_bytes=256 * 1024,
        max_pixels=1_000_000,
        max_side=320,
        queue_seconds=0.2,
    )


@pytest.fixture
def app(settings: Settings) -> Flask:
    flask_app = create_app(settings, FakeRegistry(settings.weights_dir))
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app: Flask) -> Iterator[FlaskClient]:
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def upload() -> dict[str, object]:
    return {"image": (io.BytesIO(png_bytes()), "photo.png")}
