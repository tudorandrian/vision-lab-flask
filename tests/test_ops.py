"""Quantitative checks on synthetic images: every expectation is a number, not a smoke test."""

from __future__ import annotations

from typing import Any

import cv2
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


@pytest.mark.parametrize(
    "angle, rot90_k",
    [(90, 1), (180, 2), (270, 3), (-90, -1)],
)
def test_right_angle_rotation_matches_numpy_rot90_exactly(angle: int, rot90_k: int) -> None:
    rng = np.random.default_rng(3)
    image = rng.integers(1, 256, (5, 7, 3), dtype=np.uint8)
    rotated = ops.transform(image, rotation_angle=angle)
    assert np.array_equal(rotated, np.rot90(image, k=rot90_k))


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


def test_downscaling_a_tiny_image_never_rounds_to_zero() -> None:
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    assert ops.transform(tiny, scale_factor=0.1).shape[:2] == (1, 1)


def test_scaling_never_allocates_more_than_max_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """1600 px rotated by 45 degrees is 2263 px; scaled by 4.0 that would be a
    9052 x 9052 x 3 array (246 MB) thrown away by the cap a moment later. The
    cap must be applied to the target size before the resize allocates."""
    seen: list[tuple[int, int]] = []
    real_resize = ops.cv2.resize

    def recording_resize(image: np.ndarray, dsize: tuple[int, int], **kwargs: Any) -> np.ndarray:
        seen.append(dsize)
        return real_resize(image, dsize, **kwargs)

    monkeypatch.setattr(ops.cv2, "resize", recording_resize)
    image = np.zeros((1600, 1600, 3), dtype=np.uint8)
    result = ops.transform(image, rotation_angle=45, scale_factor=4.0, max_side=1600)
    assert max(result.shape[:2]) == 1600
    assert seen, "the scaled image is still produced by one resize"
    assert all(max(dsize) <= 1600 for dsize in seen), seen


def test_scaling_below_the_cap_is_unchanged(square: np.ndarray) -> None:
    assert ops.transform(square, scale_factor=2.0, max_side=1600).shape[:2] == (200, 200)
    assert ops.transform(square, scale_factor=0.5, max_side=1600).shape[:2] == (50, 50)


def test_clamped_to_the_current_size_skips_the_resize_call(
    monkeypatch: pytest.MonkeyPatch, square: np.ndarray
) -> None:
    """When the image is already at max_side, an upscale request clamps to a
    no-op: effective == 1.0, so cv2.resize must not be called at all."""
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        ops.cv2, "resize", lambda image, dsize, **kwargs: calls.append(dsize) or image
    )
    result = ops.transform(square, scale_factor=2.0, max_side=100)
    assert result.shape[:2] == (100, 100)
    assert calls == []


def test_scaling_capped_on_a_non_square_image_rounds_the_shorter_side_once() -> None:
    """The longer side lands exactly on max_side; the shorter side is the
    single rounding of shorter * max_side / longer, aspect preserved."""
    image = np.zeros((100, 300, 3), dtype=np.uint8)  # height=100, width=300
    result = ops.transform(image, scale_factor=3.0, max_side=200)
    height, width = result.shape[:2]
    assert width == 200
    assert height == round(100 * 200 / 300)


def test_horizontal_flip_mirrors_columns() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, 0] = 255
    assert ops.transform(image, flip="horizontal")[:, 2].min() == 255


def test_median_removes_salt_and_pepper_noise() -> None:
    rng = np.random.default_rng(7)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    noisy = image.copy()
    draw = rng.random((100, 100))
    salt_mask = draw < 0.025
    pepper_mask = (draw >= 0.025) & (draw < 0.05)
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0
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
    assert edges[68:72, 40:60].max() >= 128, "bottom side must respond: both gradient directions"
    assert edges[40:60, 68:72].max() >= 128, "right side must respond: both gradient directions"


def test_log_response_is_spread_by_its_gaussian() -> None:
    """A LoG without its blur is a bare 3x3 Laplacian: a single impulse stays a single point.

    With the Gaussian, the same impulse spreads its response over a wider
    neighbourhood, which a plain |Laplacian| response cannot reproduce.
    """
    gray = np.full((64, 64), 128, dtype=np.uint8)
    gray[32, 32] = 255
    image = np.dstack([gray, gray, gray])
    edges = ops.detect_edges(image, "log", 100, 200)
    region = edges[27:38, 27:38]
    assert np.count_nonzero(region > 10) > 20


@pytest.mark.parametrize("kind", EQUALIZATIONS)
def test_equalisation_raises_contrast_and_keeps_shape(kind: str, gradient: np.ndarray) -> None:
    result = ops.equalize(gradient, kind, 2.0, 8)
    assert result.shape == gradient.shape
    assert result.std() > gradient.std()


def test_ahe_is_stronger_than_clahe(gradient: np.ndarray) -> None:
    assert (
        ops.equalize(gradient, "ahe", 2.0, 8).std() > ops.equalize(gradient, "clahe", 2.0, 8).std()
    )


def test_ahe_stretches_locally_far_more_than_global_equalisation() -> None:
    """Two narrow-contrast bands far apart in level: AHE must stretch each tile on its own.

    A defective AHE that falls back to global equalizeHist would show the same
    contrast gain as he on this image, since he is exactly that global operation.
    """
    left = np.linspace(20, 40, 128, dtype=np.uint8)
    right = np.linspace(215, 235, 128, dtype=np.uint8)
    row = np.concatenate([left, right])
    gray = np.tile(row, (256, 1))
    image = np.dstack([gray, gray, gray])

    he_tile_std = ops.equalize(image, "he", 2.0, 8)[:, :32].astype(np.float64).std()
    ahe_tile_std = ops.equalize(image, "ahe", 2.0, 8)[:, :32].astype(np.float64).std()
    assert ahe_tile_std > he_tile_std * 2


@pytest.mark.parametrize("kind", EQUALIZATIONS)
def test_equalisation_preserves_colour(kind: str) -> None:
    """Equalising the L channel only must leave the a/b chrominance channels nearly unchanged."""
    ramp = np.tile(np.linspace(96, 159, 256, dtype=np.uint8), (256, 1))
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, 2] = ramp  # a reddish ramp: only the red (BGR index 2) channel varies
    image[:, :, 1] = 40
    image[:, :, 0] = 40

    lab_before = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    result = ops.equalize(image, kind, 2.0, 8)
    lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)

    assert np.mean(np.abs(lab_after[:, :, 1] - lab_before[:, :, 1])) < 8.0
    assert np.mean(np.abs(lab_after[:, :, 2] - lab_before[:, :, 2])) < 8.0
    assert not np.array_equal(result[:, :, 0], result[:, :, 1])


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


def test_denoise_reduces_flat_noise_but_keeps_the_edge_sharp() -> None:
    """Non-local means, unlike a plain blur, smooths flat regions without softening edges."""
    rng = np.random.default_rng(7)
    base = np.zeros((64, 64), dtype=np.uint8)
    base[:, :32] = 60
    base[:, 32:] = 190
    noise = rng.normal(0, 15, (64, 64))
    noisy_gray = np.clip(base.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    noisy = np.dstack([noisy_gray, noisy_gray, noisy_gray])

    denoised = ops.enhance(noisy, "denoise", 1.0)

    noisy_flat_std = noisy[:, :28].astype(np.float64).std()
    denoised_flat_std = denoised[:, :28].astype(np.float64).std()
    assert denoised_flat_std < noisy_flat_std / 2

    edge_jump = np.mean(np.abs(denoised[:, 32, 0].astype(int) - denoised[:, 31, 0].astype(int)))
    assert edge_jump > 100


@pytest.mark.parametrize("kind", ENHANCEMENTS)
def test_enhancements_stay_uint8_at_the_extremes(kind: str, square: np.ndarray) -> None:
    for value in (0.0, 4.0):
        assert ops.enhance(square, kind, value).dtype == np.uint8
