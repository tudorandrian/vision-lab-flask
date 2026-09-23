"""Pretrained models behind small protocols.

Heavy libraries are imported inside the classes that need them, so importing
this module costs nothing and the web layer can be tested with fakes. Each model
is loaded once per process, on first use, and then reused.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import cv2

from vision_lab.imaging import Image

# Forced unconditionally (assignment, not setdefault), at module import time
# rather than inside YoloDetector.__init__, so this is structural rather than
# dependent on nothing else importing ultralytics first, and so a careless or
# hostile environment cannot switch either guard back off. Both are read as
# module-level constants the first time ultralytics.utils is imported
# (AUTOINSTALL: ultralytics/utils/__init__.py:71; SAFE_LOAD:
# ultralytics/utils/__init__.py:73, consumed at nn/tasks.py:1824), so this
# must run before that happens, wherever the import happens from. Unset, a
# decode failure on a hostile upload (Pillow's opener is patched by
# ultralytics on import) triggers `uv pip install` from PyPI mid-request
# (AUTOINSTALL defaults True), and a YOLO checkpoint is unpickled with
# weights_only=False (SAFE_LOAD defaults False): neither an unlocked package
# nor code execution from a tampered checkpoint is acceptable while serving.
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"

# The confidence below which a detection is discarded. Shared by every detector
# so the result page can report the one number that was actually applied.
DEFAULT_SCORE_THRESHOLD = 0.5


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
    """Faster R-CNN, MobileNetV3-Large 320 FPN backbone, COCO labels. BSD-3-Clause.

    Images are resized so the short side is 320 px internally, trading
    small-object accuracy for CPU speed.
    """

    def __init__(self, weights_dir: Path, score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        torch.hub.set_dir(str(weights_dir))
        # Pinned rather than .DEFAULT, which is defined to track "the best
        # available weights": an upstream torchvision upgrade could otherwise
        # change detection results with no change to this file. This is the
        # weights enum .DEFAULT currently resolves to.
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        self._labels: Sequence[str] = weights.meta["categories"]
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, box_score_thresh=score_threshold
        ).eval()

    def detect(self, image: Image) -> list[Detection]:
        import torch

        with torch.inference_mode():
            output = self._model([_to_tensor(image)])[0]
        detections = []
        for box, label, score in zip(
            output["boxes"].tolist(),
            output["labels"].tolist(),
            output["scores"].tolist(),
            strict=True,
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(Detection(self._labels[int(label)], float(score), (x1, y1, x2, y2)))
        return detections


class YoloDetector:
    """Ultralytics YOLOv5u. AGPL-3.0; installed only with the 'yolo' extra.

    A YOLO instance keeps per-call predictor state, so calls are serialised
    with a lock; the torchvision models above are stateless under
    inference_mode and need no lock.
    """

    def __init__(
        self, weights_dir: Path, variant: str, score_threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> None:
        # YOLO_AUTOINSTALL and ULTRALYTICS_SAFE_LOAD are forced at module
        # import time, above; see the comment there.
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
        detections = []
        for box, label, score in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            strict=True,
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(Detection(names[int(label)], float(score), (x1, y1, x2, y2)))
        return detections


class DeepLabSegmenter:
    """DeepLabV3, MobileNetV3-Large backbone, the 21 Pascal VOC classes. BSD-3-Clause."""

    def __init__(self, weights_dir: Path) -> None:
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        torch.hub.set_dir(str(weights_dir))
        # Pinned rather than .DEFAULT, for the same reason as the detector
        # above: this is what .DEFAULT currently resolves to.
        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
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
        # DeepFace reads DEEPFACE_HOME at import time and stores weights under
        # <home>/.deepface/weights, so this must be set before the import below.
        os.environ.setdefault("DEEPFACE_HOME", str(weights_dir))
        # DeepFace prints a backend deprecation banner on every import, and
        # TensorFlow prints a oneDNN notice before its log level is read, so
        # TF_CPP_MIN_LOG_LEVEL cannot hide it; only disabling oneDNN does. For
        # this small model that changes neither the scores nor the speed.
        # Defaults only: a user can turn all of it back on.
        os.environ.setdefault("DEEPFACE_LOG_LEVEL", str(logging.ERROR))
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        tensorflow_logger = logging.getLogger("tensorflow")
        if tensorflow_logger.level == logging.NOTSET:
            tensorflow_logger.setLevel(logging.ERROR)

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


T = TypeVar("T")


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

    def _get(self, key: str, build: Callable[[], T]) -> T:
        with self._lock:
            if key not in self._cache:
                self.weights_dir.mkdir(parents=True, exist_ok=True)
                self._cache[key] = build()
            # The cache is keyed by model name and only ever holds what build()
            # returned for that key, so this is a real invariant, not a lie to
            # the type checker: build() itself is what fixes T at each call site.
            return cast(T, self._cache[key])

    def detector(self, name: str) -> Detector:
        if name not in self.available_detectors():
            raise KeyError(name)
        if name == "fasterrcnn":
            return self._get(name, lambda: TorchvisionDetector(self.weights_dir))
        return self._get(name, lambda: YoloDetector(self.weights_dir, name))

    def segmenter(self) -> Segmenter:
        return self._get("deeplabv3", lambda: DeepLabSegmenter(self.weights_dir))

    def emotion_analyzer(self) -> EmotionAnalyzer | None:
        if not self.emotion_available():
            return None
        return self._get("deepface", lambda: DeepFaceEmotionAnalyzer(self.weights_dir))
