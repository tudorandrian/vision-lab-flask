"""Quantitative checks on synthetic images: every expectation is a number, not a smoke test."""

from __future__ import annotations

import numpy as np
import pytest

from vision_lab import ops
from vision_lab.params import COLOR_SPACES, EDGE_ALGORITHMS, ENHANCEMENTS, EQUALIZATIONS


@pytest.fixture
def square() -> np.ndarray:
    """Black 100x100 image with a white 40x40 square: edges are known exactly."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = 255
    return image


@pytest.fixture
def gradient() -> np.ndarray:
    """Low-contrast horizontal ramp, 96..159, wide enough for an 8x8 tile grid."""
    ramp = np.tile(np.linspace(96, 159, 256, dtype=np.uint8), (256, 1))
    return np.dstack([ramp, ramp, ramp])


def test_downscale_keeps_aspect_ratio_and_never_enlarges() -> None:
    wide = np.zeros((500, 1000, 3), dtype=np.uint8)
    assert ops.downscale_to(wide, 200).shape[:2] == (100, 200)
    assert ops.downscale_to(wide, 5000) is wide


@pytest.mark.parametrize("space", COLOR_SPACES)
def test_channel_montage_shape(space: str, square: np.ndarray) -> None:
    montage = ops.channel_montage(square, space)
    assert montage.dtype == np.uint8
    assert montage.shape == ((100, 100) if space == "GRAY" else (100, 300))


def test_rgb_montage_orders_planes_red_green_blue() -> None:
    blue = np.zeros((4, 4, 3), dtype=np.uint8)
    blue[:, :, 0] = 255  # BGR: channel 0 is blue
    montage = ops.channel_montage(blue, "RGB")
    assert (montage[:, :4].max(), montage[:, 4:8].max(), montage[:, 8:].min()) == (0, 0, 255)


def test_rotation_by_90_swaps_sides_and_keeps_every_pixel(square: np.ndarray) -> None:
    tall = np.ascontiguousarray(square[:, :60])
    rotated = ops.transform(tall, rotation_angle=90)
    assert rotated.shape[:2] == (60, 100)
    assert abs(int(rotated.sum()) - int(tall.sum())) / tall.sum() < 0.02


def test_rotation_by_45_grows_the_canvas(square: np.ndarray) -> None:
    assert ops.transform(square, rotation_angle=45).shape[:2] == (141, 141)


def test_full_turn_is_the_identity(square: np.ndarray) -> None:
    assert np.array_equal(ops.transform(square, rotation_angle=360), square)


def test_crop_is_clamped_to_the_image(square: np.ndarray) -> None:
    assert ops.transform(square, crop=(90, 90, 500, 500)).shape[:2] == (10, 10)
    assert ops.transform(square, crop=(5000, 5000, 10, 10)).shape[:2] == (1, 1)
    assert ops.transform(square, crop=(0, 0, 0, 50)).shape[:2] == (100, 100)


def test_upscaling_is_capped_by_max_side(square: np.ndarray) -> None:
    assert max(ops.transform(square, scale_factor=4.0, max_side=250).shape[:2]) == 250


def test_horizontal_flip_mirrors_columns() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, 0] = 255
    assert ops.transform(image, flip="horizontal")[:, 2].min() == 255


def test_median_removes_salt_and_pepper_noise() -> None:
    rng = np.random.default_rng(7)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    noisy = image.copy()
    mask = rng.random((100, 100)) < 0.05
    noisy[mask] = 255
    restored = ops.smooth(noisy, "median", 3)
    assert np.mean(restored == 128) > 0.99


def test_gaussian_reduces_noise_variance() -> None:
    rng = np.random.default_rng(7)
    noisy = rng.normal(128, 20, (100, 100, 3)).clip(0, 255).astype(np.uint8)
    assert ops.smooth(noisy, "gaussian", 7).std() < noisy.std() / 3


@pytest.mark.parametrize("algorithm", EDGE_ALGORITHMS)
def test_edges_lie_on_the_square_outline(algorithm: str, square: np.ndarray) -> None:
    edges = ops.detect_edges(square, algorithm, 100, 200)
    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8
    assert edges.max() == 255
    assert edges[45:55, 45:55].max() == 0, "flat interior must have no response"
    assert edges[0:20, 0:20].max() == 0, "flat background must have no response"
    assert edges[28:32, 40:60].max() >= 128, "top side of the square must respond"
    assert edges[40:60, 28:32].max() >= 128, "left side must respond: both gradient directions"


@pytest.mark.parametrize("kind", EQUALIZATIONS)
def test_equalisation_raises_contrast_and_keeps_shape(kind: str, gradient: np.ndarray) -> None:
    result = ops.equalize(gradient, kind, 2.0, 8)
    assert result.shape == gradient.shape
    assert result.std() > gradient.std()


def test_ahe_is_stronger_than_clahe(gradient: np.ndarray) -> None:
    assert (
        ops.equalize(gradient, "ahe", 2.0, 8).std() > ops.equalize(gradient, "clahe", 2.0, 8).std()
    )


@pytest.mark.parametrize("kind", ["brightness", "contrast"])
def test_factor_one_is_the_identity(kind: str, gradient: np.ndarray) -> None:
    assert np.array_equal(ops.enhance(gradient, kind, 1.0), gradient)


def test_brightness_and_contrast_move_the_right_statistic(gradient: np.ndarray) -> None:
    assert ops.enhance(gradient, "brightness", 1.5).mean() > gradient.mean() * 1.4
    stretched = ops.enhance(gradient, "contrast", 2.0)
    assert stretched.std() > gradient.std() * 1.9
    assert abs(stretched.mean() - gradient.mean()) < 1.0


def test_sharpen_increases_edge_strength(square: np.ndarray) -> None:
    soft = ops.smooth(square, "gaussian", 9)
    assert (
        np.abs(np.diff(ops.enhance(soft, "sharpen", 1.0).astype(int), axis=1)).max()
        > np.abs(np.diff(soft.astype(int), axis=1)).max()
    )


def test_denoise_is_not_a_placeholder() -> None:
    rng = np.random.default_rng(7)
    noisy = rng.normal(128, 15, (64, 64, 3)).clip(0, 255).astype(np.uint8)
    assert ops.enhance(noisy, "denoise", 1.0).std() < noisy.std() / 2


@pytest.mark.parametrize("kind", ENHANCEMENTS)
def test_enhancements_stay_uint8_at_the_extremes(kind: str, square: np.ndarray) -> None:
    for value in (0.0, 4.0):
        assert ops.enhance(square, kind, value).dtype == np.uint8
