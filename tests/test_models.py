"""Real weights, real inference. Opt in with: pytest -m models (downloads about 120 MB once)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_lab.inference import ModelRegistry

pytestmark = pytest.mark.models

SAMPLE = Path(__file__).parent.parent / "samples" / "astronaut.jpg"


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry(Path("instance/weights").resolve())


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
