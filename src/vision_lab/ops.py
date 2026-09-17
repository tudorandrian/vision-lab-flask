"""Classical image operations. Pure functions: BGR uint8 array in, uint8 array out.

No file access and no Flask imports here, so every function is unit-testable on
synthetic arrays with exact, quantitative expectations.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_lab.imaging import Image


def _to_uint8(values: NDArray[Any]) -> Image:
    """Scale a non-negative response map to the full 0..255 range."""
    peak = float(values.max()) if values.size else 0.0
    if peak <= 0.0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(np.rint(values * (255.0 / peak)), 0, 255).astype(np.uint8)


def downscale_to(image: Image, max_side: int) -> Image:
    """Shrink so the longer side is at most max_side. Never enlarges."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    ratio = max_side / longest
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def channel_montage(image: Image, color_space: str) -> Image:
    """Convert to a colour space and lay its channels side by side as grey planes.

    Saving HSV or LAB data as if it were BGR produces false colours that teach
    nothing; separate planes show what each channel actually encodes.
    """
    if color_space == "GRAY":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    codes = {"HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB, "RGB": cv2.COLOR_BGR2RGB}
    converted = cv2.cvtColor(image, codes[color_space])
    return np.hstack([converted[:, :, index] for index in range(3)])


def transform(
    image: Image,
    *,
    rotation_angle: int = 0,
    scale_factor: float = 1.0,
    crop: tuple[int, int, int, int] = (0, 0, 0, 0),
    flip: str = "none",
    max_side: int = 1600,
) -> Image:
    """Rotate (canvas grows, corners are kept), scale, crop, then flip.

    crop is (x, y, width, height); a zero width or height means no crop. The
    rectangle is clamped to the image, so it can never yield an empty array.
    """
    result = image
    if rotation_angle % 360 != 0:
        height, width = result.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)
        matrix[0, 2] += new_width / 2 - width / 2
        matrix[1, 2] += new_height / 2 - height / 2
        result = cv2.warpAffine(result, matrix, (new_width, new_height))
    if scale_factor != 1.0:
        interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
        result = cv2.resize(
            result, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation
        )
        result = downscale_to(result, max_side)
    x, y, crop_width, crop_height = crop
    if crop_width > 0 and crop_height > 0:
        height, width = result.shape[:2]
        x = min(x, width - 1)
        y = min(y, height - 1)
        result = result[y : min(y + crop_height, height), x : min(x + crop_width, width)]
    flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
    if flip in flip_codes:
        result = cv2.flip(result, flip_codes[flip])
    return np.ascontiguousarray(result)


def smooth(image: Image, filter_type: str, kernel_size: int) -> Image:
    """Gaussian or median blur with an odd kernel_size."""
    if filter_type == "median":
        return cv2.medianBlur(image, kernel_size)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image: Image, algorithm: str, threshold1: int, threshold2: int) -> Image:
    """Edge map as a single uint8 plane, full gradient magnitude for every operator."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if algorithm == "canny":
        return cv2.Canny(gray, threshold1, threshold2)
    if algorithm == "log":
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        return _to_uint8(np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)))
    if algorithm == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    elif algorithm == "scharr":
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    else:  # roberts cross
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    return _to_uint8(cv2.magnitude(grad_x, grad_y))


def equalize(image: Image, kind: str, clip_limit: float, tile_grid_size: int) -> Image:
    """Histogram equalisation on the L channel of LAB, so colours are preserved.

    he: global. ahe: adaptive per tile, no clipping. clahe: adaptive with a clip limit.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    if kind == "he":
        lab[:, :, 0] = cv2.equalizeHist(lightness)
    else:
        limit = 0.0 if kind == "ahe" else clip_limit
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(tile_grid_size, tile_grid_size))
        lab[:, :, 0] = clahe.apply(lightness)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance(image: Image, kind: str, value: float) -> Image:
    """sharpen: unsharp mask, value is the amount. denoise: non-local means, strength 10*value.

    brightness and contrast: value is a factor, 1.0 leaves the image unchanged.
    """
    if kind == "sharpen":
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.0 + value, blurred, -value, 0)
    if kind == "denoise":
        strength = max(1.0, 10.0 * value)
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    pixels = image.astype(np.float32)
    if kind == "brightness":
        pixels *= value
    else:  # contrast, around the mean intensity
        mean = float(pixels.mean())
        pixels = (pixels - mean) * value + mean
    return np.clip(pixels, 0, 255).astype(np.uint8)
