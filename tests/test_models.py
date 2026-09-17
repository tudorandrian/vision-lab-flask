"""Real weights, real inference. Opt in with: pytest -m models (downloads about 120 MB once)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_lab.config import Settings
from vision_lab.inference import ModelRegistry

pytestmark = pytest.mark.models

SAMPLE = Path(__file__).parent.parent / "samples" / "astronaut.jpg"


def _weights_dir() -> Path:
    """Where the real registry below stores weights.

    Reads VISION_LAB_DATA_DIR the same way the application does, so setting
    it to reuse a weights cache actually works for this test module too,
    instead of always resolving ./instance/weights against the working
    directory regardless of that variable.
    """
    return Settings.from_env().weights_dir


def test_weights_dir_honours_vision_lab_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISION_LAB_DATA_DIR", str(tmp_path))
    assert _weights_dir() == tmp_path.resolve() / "weights"


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry(_weights_dir())


@pytest.fixture(scope="module")
def astronaut() -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(SAMPLE.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_every_installed_detector_finds_the_person(
    registry: ModelRegistry, astronaut: np.ndarray
) -> None:
    for name in registry.available_detectors():
        people = [d for d in registry.detector(name).detect(astronaut) if d.label == "person"]
        assert people, f"{name} found no person"
        best = max(people, key=lambda d: d.confidence)
        assert best.confidence > 0.8
        x1, y1, x2, y2 = best.box
        assert (x2 - x1) * (y2 - y1) > 0.25 * astronaut.shape[0] * astronaut.shape[1]


def test_segmentation_labels_a_large_person_region(
    registry: ModelRegistry, astronaut: np.ndarray
) -> None:
    segmenter = registry.segmenter()
    class_map = segmenter.segment(astronaut)
    assert class_map.shape == astronaut.shape[:2]
    assert class_map.dtype == np.uint8
    person = list(segmenter.class_names).index("person")
    assert 0.25 < float(np.mean(class_map == person)) < 0.9


def test_models_are_built_once(registry: ModelRegistry) -> None:
    assert registry.segmenter() is registry.segmenter()
    assert registry.detector("fasterrcnn") is registry.detector("fasterrcnn")


def test_yolo_detector_disables_autoinstall_and_unsafe_pickle_load(
    registry: ModelRegistry,
) -> None:
    """A tampered or substituted checkpoint must not run arbitrary code, and a
    Pillow decode failure during a request must never trigger a PyPI install."""
    registry.detector("yolov5nu")
    import ultralytics.utils as ultra_utils

    assert ultra_utils.AUTOINSTALL is False
    assert ultra_utils.SAFE_LOAD is True
