"""Drawing model output onto images, and the numbers reported next to them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_lab.imaging import Image
from vision_lab.inference import Detection, FaceEmotion

_GREEN = (0, 200, 0)
_BLUE = (255, 128, 0)  # BGR


class SegmentMetric(TypedDict):
    class_id: int
    label: str
    pixels: int
    percentage: float


def _label(image: Image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, 0.5, 1)
    top = max(y - height - baseline - 4, 0)
    cv2.rectangle(image, (x, top), (x + width + 4, top + height + baseline + 4), color, -1)
    cv2.putText(image, text, (x + 2, top + height + 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_detections(image: Image, detections: Sequence[Detection]) -> Image:
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _GREEN, 2)
        _label(canvas, f"{detection.label} {detection.confidence:.2f}", x1, y1, _GREEN)
    return canvas


def draw_faces(image: Image, faces: Sequence[FaceEmotion]) -> Image:
    canvas = image.copy()
    for face in faces:
        x, y, width, height = face.box
        cv2.rectangle(canvas, (x, y), (x + width, y + height), _BLUE, 2)
        _label(canvas, face.dominant, x, y, _BLUE)
    return canvas


def class_palette(class_count: int) -> NDArray[np.uint8]:
    """Deterministic, well separated BGR colours. Class 0 (background) stays black."""
    palette = np.zeros((class_count, 3), dtype=np.uint8)
    for class_id in range(1, class_count):
        hue = int(180 * ((class_id * 0.618033988749895) % 1.0))
        hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
        palette[class_id] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return palette


def overlay_segmentation(image: Image, class_map: Image, class_count: int) -> Image:
    """Blend a per-pixel class map over the image; background pixels are left untouched."""
    colours = class_palette(class_count)[class_map]
    blended = cv2.addWeighted(image, 0.5, colours, 0.5, 0)
    foreground = class_map > 0
    canvas = image.copy()
    canvas[foreground] = blended[foreground]
    return canvas


def segmentation_metrics(class_map: Image, class_names: Sequence[str]) -> list[SegmentMetric]:
    """Pixel count and share per class, measured on the class map itself."""
    class_ids, counts = np.unique(class_map, return_counts=True)
    total = int(class_map.size)
    metrics: list[SegmentMetric] = [
        {
            "class_id": int(class_id),
            "label": class_names[int(class_id)].strip("_"),
            "pixels": int(count),
            "percentage": round(100.0 * int(count) / total, 2),
        }
        for class_id, count in zip(class_ids, counts, strict=True)
    ]
    return sorted(metrics, key=lambda metric: metric["pixels"], reverse=True)
