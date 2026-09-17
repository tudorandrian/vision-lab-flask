"""Runs every stage for one upload and records what was produced.

The result is a plain dictionary saved as result.json, so showing a job again
is a file read: models run once per upload, never per page view.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from vision_lab import catalog, ops, render
from vision_lab.imaging import Image
from vision_lab.inference import ModelRegistry
from vision_lab.params import (
    COLOR_SPACES,
    EDGE_ALGORITHMS,
    ENHANCEMENTS,
    EQUALIZATIONS,
    FILTERS,
    ProcessingParams,
)
from vision_lab.storage import JobStore


def _variants(selected: str, every: Sequence[str], result_option: str) -> Sequence[str]:
    return every if result_option == "all" else (selected,)


def _section(
    store: JobStore,
    job_id: str,
    group_key: str,
    keys: Sequence[str],
    produce: Callable[[str], Image],
) -> dict[str, Any]:
    group = catalog.group(group_key)
    items = []
    for key in keys:
        entry = group.get(key)
        name = store.save_image(job_id, f"{group_key}_{key.lower()}.jpg", produce(key))
        items.append({"title": entry.title, "summary": entry.summary, "file": name})
    return {"key": group_key, "title": group.title, "items": items}


def run_job(
    store: JobStore,
    models: ModelRegistry,
    job_id: str,
    image: Image,
    params: ProcessingParams,
    max_side: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    option = params.result_option
    store.save_image(job_id, "original.jpg", image)

    transformed = ops.transform(
        image,
        rotation_angle=params.rotation_angle,
        scale_factor=params.scale_factor,
        crop=(params.crop_x, params.crop_y, params.crop_width, params.crop_height),
        flip=params.flip,
        max_side=max_side,
    )
    store.save_image(job_id, "transformed.jpg", transformed)

    detections = models.detector(params.detector).detect(image)
    store.save_image(job_id, "detections.jpg", render.draw_detections(image, detections))

    segmenter = models.segmenter()
    class_map = segmenter.segment(image)
    class_names = list(segmenter.class_names)
    overlay = render.overlay_segmentation(image, class_map, len(class_names))
    store.save_image(job_id, "segmentation.jpg", overlay)

    faces: list[dict[str, Any]] | None = None
    analyzer = models.emotion_analyzer()
    if analyzer is not None and any(item.label == "person" for item in detections):
        found = analyzer.analyze(image)
        store.save_image(job_id, "faces.jpg", render.draw_faces(image, found))
        faces = [{"dominant": face.dominant, "scores": face.scores} for face in found]

    sections = [
        _section(
            store,
            job_id,
            "filter_type",
            _variants(params.filter_type, FILTERS, option),
            lambda key: ops.smooth(image, key, params.kernel_size),
        ),
        _section(
            store,
            job_id,
            "edge_algorithm",
            _variants(params.edge_algorithm, EDGE_ALGORITHMS, option),
            lambda key: ops.detect_edges(image, key, params.threshold1, params.threshold2),
        ),
        _section(
            store,
            job_id,
            "equalization_type",
            _variants(params.equalization_type, EQUALIZATIONS, option),
            lambda key: ops.equalize(image, key, params.clip_limit, params.tile_grid_size),
        ),
        _section(
            store,
            job_id,
            "enhancement_type",
            _variants(params.enhancement_type, ENHANCEMENTS, option),
            lambda key: ops.enhance(image, key, params.enhancement_value),
        ),
        _section(
            store,
            job_id,
            "color_space",
            _variants(params.color_space, COLOR_SPACES, option),
            lambda key: ops.channel_montage(image, key),
        ),
    ]

    result: dict[str, Any] = {
        "detector": catalog.group("models").get(params.detector).title,
        "detections": [
            {"label": item.label, "confidence": round(item.confidence, 2)} for item in detections
        ],
        "segments": render.segmentation_metrics(class_map, class_names),
        "faces": faces,
        "sections": sections,
        "size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "seconds": round(time.perf_counter() - started, 2),
    }
    store.save_result(job_id, result)
    return result
