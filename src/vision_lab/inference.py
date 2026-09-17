"""Pretrained models behind small protocols.

Heavy libraries are imported inside the classes that need them, so importing
this module costs nothing and the web layer can be tested with fakes. Each model
is loaded once per process, on first use, and then reused.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2

from vision_lab.imaging import Image


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class FaceEmotion:
    box: tuple[int, int, int, int]  # x, y, width, height
    dominant: str
    scores: dict[str, float]


class Detector(Protocol):
    def detect(self, image: Image) -> list[Detection]: ...


class Segmenter(Protocol):
    class_names: Sequence[str]

    def segment(self, image: Image) -> Image: ...


class EmotionAnalyzer(Protocol):
    def analyze(self, image: Image) -> list[FaceEmotion]: ...


def _to_tensor(image: Image) -> Any:
    import torch

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)


class TorchvisionDetector:
    """Faster R-CNN, MobileNetV3-Large FPN backbone, COCO labels. BSD-3-Clause."""

    def __init__(self, weights_dir: Path, score_threshold: float = 0.5) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        torch.hub.set_dir(str(weights_dir))
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self._labels: Sequence[str] = weights.meta["categories"]
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, box_score_thresh=score_threshold
        ).eval()

    def detect(self, image: Image) -> list[Detection]:
        import torch

        with torch.inference_mode():
            output = self._model([_to_tensor(image)])[0]
        return [
            Detection(self._labels[int(label)], float(score), tuple(int(v) for v in box))  # type: ignore[arg-type]
            for box, label, score in zip(
                output["boxes"].tolist(),
                output["labels"].tolist(),
                output["scores"].tolist(),
                strict=True,
            )
        ]


class YoloDetector:
    """Ultralytics YOLOv5u. AGPL-3.0; installed only with the 'yolo' extra.

    A YOLO instance keeps per-call predictor state, so calls are serialised
    with a lock; the torchvision models above are stateless under
    inference_mode and need no lock.
    """

    def __init__(self, weights_dir: Path, variant: str, score_threshold: float = 0.5) -> None:
        from ultralytics import YOLO, settings

        # Ultralytics reports anonymous usage analytics unless told not to.
        settings.update({"sync": False, "weights_dir": str(weights_dir)})
        self._threshold = score_threshold
        self._model = YOLO(str(weights_dir / f"{variant}.pt"))
        self._lock = threading.Lock()

    def detect(self, image: Image) -> list[Detection]:
        with self._lock:
            result = self._model.predict(image, conf=self._threshold, verbose=False)[0]
        names = result.names
        return [
            Detection(names[int(label)], float(score), tuple(int(v) for v in box))  # type: ignore[arg-type]
            for box, label, score in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
                strict=True,
            )
        ]


class DeepLabSegmenter:
    """DeepLabV3, MobileNetV3-Large backbone, the 21 Pascal VOC classes. BSD-3-Clause."""

    def __init__(self, weights_dir: Path) -> None:
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        torch.hub.set_dir(str(weights_dir))
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.class_names: Sequence[str] = weights.meta["categories"]
        self._model = deeplabv3_mobilenet_v3_large(weights=weights).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def segment(self, image: Image) -> Image:
        import torch

        batch = ((_to_tensor(image) - self._mean) / self._std).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(batch)["out"][0]
        class_map: Image = logits.argmax(0).to(torch.uint8).cpu().numpy()
        return class_map


class DeepFaceEmotionAnalyzer:
    """DeepFace emotion model. Installed only with the 'emotion' extra, off by default."""

    def __init__(self, weights_dir: Path) -> None:
        self._weights_dir = weights_dir
        # DeepFace reads DEEPFACE_HOME at import time and stores weights under
        # <home>/.deepface/weights, so this must be set before the import below.
        os.environ.setdefault("DEEPFACE_HOME", str(weights_dir))

    def analyze(self, image: Image) -> list[FaceEmotion]:
        from deepface import DeepFace

        results = DeepFace.analyze(
            img_path=image, actions=["emotion"], enforce_detection=False, silent=True
        )
        faces = []
        for entry in results:
            if float(entry.get("face_confidence", 0.0)) <= 0.0:
                continue  # enforce_detection=False reports the whole frame when no face is found
            region = entry["region"]
            faces.append(
                FaceEmotion(
                    box=(int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])),
                    dominant=str(entry["dominant_emotion"]),
                    scores={name: round(float(v), 2) for name, v in entry["emotion"].items()},
                )
            )
        return faces


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass
class ModelRegistry:
    """Builds each model at most once and hands out the shared instance."""

    weights_dir: Path
    enable_emotion: bool = False
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def available_detectors(self) -> list[str]:
        names = ["fasterrcnn"]
        if _installed("ultralytics"):
            names += ["yolov5nu", "yolov5su"]
        return names

    def emotion_available(self) -> bool:
        return self.enable_emotion and _installed("deepface")

    def _get(self, key: str, build: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._cache:
                self.weights_dir.mkdir(parents=True, exist_ok=True)
                self._cache[key] = build()
            return self._cache[key]

    def detector(self, name: str) -> Detector:
        if name not in self.available_detectors():
            raise KeyError(name)
        if name == "fasterrcnn":
            return self._get(name, lambda: TorchvisionDetector(self.weights_dir))  # type: ignore[no-any-return]
        return self._get(name, lambda: YoloDetector(self.weights_dir, name))  # type: ignore[no-any-return]

    def segmenter(self) -> Segmenter:
        return self._get("deeplabv3", lambda: DeepLabSegmenter(self.weights_dir))  # type: ignore[no-any-return]

    def emotion_analyzer(self) -> EmotionAnalyzer | None:
        if not self.emotion_available():
            return None
        return self._get("deepface", lambda: DeepFaceEmotionAnalyzer(self.weights_dir))  # type: ignore[no-any-return]
