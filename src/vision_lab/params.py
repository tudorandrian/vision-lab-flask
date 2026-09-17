"""Form parsing and validation. Nothing reaches OpenCV without passing through here."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

COLOR_SPACES = ("HSV", "LAB", "GRAY", "RGB")
FLIPS = ("none", "horizontal", "vertical", "both")
FILTERS = ("gaussian", "median")
EDGE_ALGORITHMS = ("canny", "sobel", "scharr", "roberts", "log")
EQUALIZATIONS = ("he", "ahe", "clahe")
ENHANCEMENTS = ("sharpen", "denoise", "brightness", "contrast")
RESULT_OPTIONS = ("single", "all")


class ParamError(ValueError):
    """Raised with one message per offending field."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{name}: {message}" for name, message in errors.items()))
        self.errors = errors


@dataclass(frozen=True)
class ProcessingParams:
    detector: str
    result_option: str = "single"
    color_space: str = "HSV"
    rotation_angle: int = 0
    scale_factor: float = 1.0
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    flip: str = "none"
    filter_type: str = "gaussian"
    kernel_size: int = 7
    edge_algorithm: str = "canny"
    threshold1: int = 100
    threshold2: int = 200
    equalization_type: str = "clahe"
    clip_limit: float = 2.0
    tile_grid_size: int = 8
    enhancement_type: str = "sharpen"
    enhancement_value: float = 1.0


def parse_params(form: Mapping[str, str], detectors: Sequence[str]) -> ProcessingParams:
    """Build ProcessingParams from submitted form fields, or raise ParamError.

    Missing fields fall back to the dataclass defaults, so a bare upload is valid.
    """
    errors: dict[str, str] = {}
    defaults = ProcessingParams(detector=detectors[0])

    def choice(name: str, allowed: Sequence[str]) -> str:
        raw = form.get(name, "").strip()
        if not raw:
            return str(getattr(defaults, name))
        if raw not in allowed:
            errors[name] = f"must be one of: {', '.join(allowed)}"
            return str(getattr(defaults, name))
        return raw

    def integer(name: str, low: int, high: int) -> int:
        raw = form.get(name, "").strip()
        if not raw:
            return int(getattr(defaults, name))
        try:
            value = int(raw)
        except ValueError:
            errors[name] = "must be a whole number"
            return int(getattr(defaults, name))
        if not low <= value <= high:
            errors[name] = f"must be between {low} and {high}"
        return value

    def decimal(name: str, low: float, high: float) -> float:
        raw = form.get(name, "").strip()
        if not raw:
            return float(getattr(defaults, name))
        try:
            value = float(raw)
        except ValueError:
            errors[name] = "must be a number"
            return float(getattr(defaults, name))
        if math.isnan(value) or not low <= value <= high:
            errors[name] = f"must be between {low} and {high}"
        return value

    params = ProcessingParams(
        detector=choice("detector", detectors),
        result_option=choice("result_option", RESULT_OPTIONS),
        color_space=choice("color_space", COLOR_SPACES),
        rotation_angle=integer("rotation_angle", -360, 360),
        scale_factor=decimal("scale_factor", 0.1, 4.0),
        crop_x=integer("crop_x", 0, 10_000),
        crop_y=integer("crop_y", 0, 10_000),
        crop_width=integer("crop_width", 0, 10_000),
        crop_height=integer("crop_height", 0, 10_000),
        flip=choice("flip", FLIPS),
        filter_type=choice("filter_type", FILTERS),
        kernel_size=integer("kernel_size", 3, 31),
        edge_algorithm=choice("edge_algorithm", EDGE_ALGORITHMS),
        threshold1=integer("threshold1", 0, 1000),
        threshold2=integer("threshold2", 0, 1000),
        equalization_type=choice("equalization_type", EQUALIZATIONS),
        clip_limit=decimal("clip_limit", 0.1, 40.0),
        tile_grid_size=integer("tile_grid_size", 1, 32),
        enhancement_type=choice("enhancement_type", ENHANCEMENTS),
        enhancement_value=decimal("enhancement_value", 0.0, 4.0),
    )
    if "kernel_size" not in errors and params.kernel_size % 2 == 0:
        errors["kernel_size"] = "must be odd (OpenCV kernels need a centre pixel)"
    thresholds_valid = "threshold1" not in errors and "threshold2" not in errors
    if thresholds_valid and params.threshold1 > params.threshold2:
        errors["threshold1"] = "must not exceed threshold2"
    if errors:
        raise ParamError(errors)
    return params
