"""Real weights, real inference. Opt in with: pytest -m models (downloads about 120 MB once)."""

from __future__ import annotations

import subprocess
import sys
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


@pytest.mark.models
def test_deepface_imports_and_reports_no_face_on_a_blank_image(tmp_path: Path) -> None:
    """The emotion extra is covered by a fake elsewhere. This runs the real
    DeepFace once: it proves the import still works and that a frame with no
    face yields an empty list (enforce_detection=False reports the whole frame
    with face_confidence 0, which the analyzer must drop). No face fixture is
    used: the repository has no image with consent for that."""
    pytest.importorskip("deepface")
    from vision_lab.inference import DeepFaceEmotionAnalyzer

    blank = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert DeepFaceEmotionAnalyzer(tmp_path).analyze(blank) == []


def test_environment_cannot_re_enable_autoinstall_or_unsafe_pickle_loading() -> None:
    """These must be forced unconditionally at import time of vision_lab.inference,
    not merely defaulted: a careless or hostile environment that presets
    YOLO_AUTOINSTALL=true or ULTRALYTICS_SAFE_LOAD=0 before the process starts
    must not be able to switch the guard off. Run in a fresh interpreter,
    since ultralytics.utils keeps its module-level constants for the rest of
    any one process once imported anywhere in it.
    """
    script = (
        "import os\n"
        "os.environ['YOLO_AUTOINSTALL'] = 'true'\n"
        "os.environ['ULTRALYTICS_SAFE_LOAD'] = '0'\n"
        "import vision_lab.inference\n"
        "import ultralytics.utils as u\n"
        "print(u.AUTOINSTALL, u.SAFE_LOAD)\n"
    )
    result = subprocess.run(  # noqa: S603 -- sys.executable and a literal script, no user input
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False True"
