from __future__ import annotations

import numpy as np

from vision_lab import render
from vision_lab.inference import Detection, FaceEmotion


def test_metrics_are_measured_on_the_class_map_and_sum_to_100() -> None:
    class_map = np.zeros((10, 10), dtype=np.uint8)
    class_map[:3, :] = 2
    metrics = render.segmentation_metrics(class_map, ["__background__", "aeroplane", "bicycle"])
    assert metrics == [
        {"class_id": 0, "label": "background", "pixels": 70, "percentage": 70.0},
        {"class_id": 2, "label": "bicycle", "pixels": 30, "percentage": 30.0},
    ]


def test_overlay_tints_foreground_only() -> None:
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    class_map = np.zeros((10, 10), dtype=np.uint8)
    class_map[:5, :] = 15
    overlay = render.overlay_segmentation(image, class_map, 21)
    assert np.array_equal(overlay[5:], image[5:])
    assert not np.array_equal(overlay[:5], image[:5])
    assert np.array_equal(image, np.full((10, 10, 3), 100, dtype=np.uint8)), "input is not mutated"


def test_palette_is_deterministic_and_distinct() -> None:
    palette = render.class_palette(21)
    assert tuple(palette[0]) == (0, 0, 0)
    assert len({tuple(colour) for colour in palette}) == 21
    assert np.array_equal(palette, render.class_palette(21))


def test_drawing_returns_a_copy_with_marks() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    boxed = render.draw_detections(image, [Detection("cat", 0.9, (10, 30, 60, 70))])
    faces = render.draw_faces(image, [FaceEmotion((10, 30, 20, 20), "happy", {"happy": 99.0})])
    assert image.max() == 0
    assert boxed.max() > 0
    assert faces.max() > 0
