from __future__ import annotations

import io
import os
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PilImage

from vision_lab.storage import JobStore, UploadError, decode_upload

LIMITS = {"max_pixels": 1_000_000, "max_side": 320}


def encoded(
    fmt: str, size: tuple[int, int] = (64, 48), mode: str = "RGB", **save: object
) -> io.BytesIO:
    buffer = io.BytesIO()
    PilImage.new(mode, size, 128 if mode == "L" else None).save(buffer, format=fmt, **save)
    buffer.seek(0)
    return buffer


def png_header_only(width: int, height: int) -> bytes:
    """A minimal, structurally valid PNG whose IHDR declares width x height.

    There is no IDAT chunk: nothing here can be decoded into pixels. This
    proves that a pixel-count rejection happens from the header alone,
    before any attempt to decode image data.
    """
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + data))
        return struct.pack(">I", len(data)) + kind + data + crc

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + iend


@pytest.mark.parametrize(
    ("fmt", "mode"),
    [("JPEG", "RGB"), ("PNG", "RGBA"), ("PNG", "L"), ("PNG", "P"), ("WEBP", "RGB"), ("BMP", "RGB")],
)
def test_accepted_formats_become_bgr_uint8(fmt: str, mode: str) -> None:
    image = decode_upload(encoded(fmt, mode=mode), **LIMITS)
    assert image.shape == (48, 64, 3)
    assert image.dtype == np.uint8


def test_large_images_are_downscaled() -> None:
    assert max(decode_upload(encoded("JPEG", (800, 400)), **LIMITS).shape[:2]) == 320


def test_exif_orientation_is_applied() -> None:
    # A landscape (64x48) source with a marked block at its top-left corner.
    # EXIF orientation 6 means "rotate 90 degrees clockwise to display
    # correctly", which Pillow's exif_transpose implements by rotating the
    # raw pixel data 270 degrees. For a corner marker that moves the mark
    # from the top-left to the top-right of the now-portrait (48x64) image;
    # this was verified directly against Pillow's own exif_transpose, not
    # against this project's decode_upload.
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    source[0:12, 0:16] = (255, 0, 0)  # red block, top-left, RGB order
    exif = PilImage.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    exif[0x010F] = "SecretCameraMaker"
    buffer = io.BytesIO()
    PilImage.fromarray(source, "RGB").save(buffer, format="JPEG", exif=exif.tobytes())
    buffer.seek(0)

    image = decode_upload(buffer, **LIMITS)

    assert image.shape[:2] == (64, 48), "portrait after applying the orientation tag"
    # BGR order: red shows up as a high value in channel index 2.
    top_right = image[2:10, 38:46, 2]
    top_left = image[2:10, 2:10, 2]
    assert top_right.mean() > 200, "the marker moved to the top-right corner"
    assert top_left.mean() < 50, "nothing was left behind at the top-left corner"


def test_saved_image_carries_no_exif_metadata(tmp_path: Path) -> None:
    exif = PilImage.Exif()
    exif[0x0112] = 6
    exif[0x010F] = "SecretCameraMaker"
    buffer = io.BytesIO()
    PilImage.new("RGB", (64, 48)).save(buffer, format="JPEG", exif=exif.tobytes())
    buffer.seek(0)
    image = decode_upload(buffer, **LIMITS)

    store = JobStore(tmp_path / "jobs", ttl_minutes=60)
    job_id = store.create()
    store.save_image(job_id, "original.jpg", image)

    written = store.file_path(job_id, "original.jpg").read_bytes()
    assert b"SecretCameraMaker" not in written


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not an image",
        b"<svg xmlns='http://www.w3.org/2000/svg'/>",
        b"GIF89a" + b"\x00" * 32,
        b"%PDF-1.7",
    ],
)
def test_non_images_are_rejected(payload: bytes) -> None:
    with pytest.raises(UploadError):
        decode_upload(io.BytesIO(payload), **LIMITS)


def test_gif_is_not_an_accepted_format() -> None:
    with pytest.raises(UploadError, match="JPEG, PNG"):
        decode_upload(encoded("GIF"), **LIMITS)


def test_mpo_jpegs_from_phone_cameras_are_accepted() -> None:
    # Many phone cameras save a JPEG with an embedded MPF segment (a second,
    # small preview frame). Pillow identifies this as format "MPO", not
    # "JPEG"; only the first (primary) frame is used.
    buffer = io.BytesIO()
    primary = PilImage.new("RGB", (64, 48), (255, 0, 0))
    preview = PilImage.new("RGB", (32, 24), (0, 255, 0))
    primary.save(buffer, format="MPO", save_all=True, append_images=[preview])
    buffer.seek(0)
    assert PilImage.open(buffer).format == "MPO"
    buffer.seek(0)

    image = decode_upload(buffer, **LIMITS)

    assert image.shape == (48, 64, 3)
    assert image.dtype == np.uint8


def test_a_real_oversized_image_is_rejected_for_pixel_count() -> None:
    """A genuine, fully decodable 2000x2000 PNG still exceeds max_pixels.

    This does not prove that the rejection happens before decoding (a real
    2000x2000 PNG is cheap to decode either way); see the header-only tests
    below for that guarantee.
    """
    with pytest.raises(UploadError, match="pixels"):
        decode_upload(encoded("PNG", (2000, 2000)), **LIMITS)


def test_header_only_bomb_below_pillows_own_threshold_is_rejected_by_pixel_limit() -> None:
    """10000x10000 has no pixel data at all, and is below Pillow's own 2x
    decompression-bomb threshold (so Pillow only warns, it does not raise).
    It must still be rejected by our max_pixels check, from the header
    alone, proving the check happens before any attempt to decode pixels.
    """
    with pytest.raises(UploadError, match="pixels"):
        decode_upload(io.BytesIO(png_header_only(10_000, 10_000)), **LIMITS)


def test_header_only_bomb_above_pillows_own_threshold_is_rejected() -> None:
    """50000x50000 has no pixel data, and is far above Pillow's own 2x
    decompression-bomb threshold, so Pillow itself refuses to open it. This
    must still surface as an UploadError, not an unhandled PIL exception.
    """
    with pytest.raises(UploadError):
        decode_upload(io.BytesIO(png_header_only(50_000, 50_000)), **LIMITS)


def test_truncated_file_is_rejected() -> None:
    data = encoded("JPEG", (200, 200)).getvalue()
    with pytest.raises(UploadError):
        decode_upload(io.BytesIO(data[: len(data) // 2]), **LIMITS)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs", ttl_minutes=60)


def test_round_trip(store: JobStore) -> None:
    job_id = store.create()
    assert len(job_id) == 32
    store.save_image(job_id, "original.jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    store.save_result(job_id, {"seconds": 1.5})
    assert store.file_path(job_id, "original.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert store.load_result(job_id) == {"seconds": 1.5}


@pytest.mark.parametrize(
    "job_id", ["..", "../..", "a" * 31, "A" * 32, "g" * 32, "", "a" * 32 + "/x"]
)
def test_malformed_job_ids_never_touch_the_disk(store: JobStore, job_id: str) -> None:
    with pytest.raises(KeyError):
        store.load_result(job_id)
    with pytest.raises(KeyError):
        store.file_path(job_id, "original.jpg")


@pytest.mark.parametrize(
    "name",
    ["../result.json", "result.json", "a/b.jpg", "x.exe", ".jpg", "UP.JPG", "a b.jpg", "x.jpg\x00"],
)
def test_only_plain_image_names_are_served(store: JobStore, name: str) -> None:
    with pytest.raises(KeyError):
        store.file_path(store.create(), name)


def test_unknown_job_is_a_key_error(store: JobStore) -> None:
    with pytest.raises(KeyError):
        store.load_result("0" * 32)


def test_purge_removes_only_expired_jobs(store: JobStore, tmp_path: Path) -> None:
    old, fresh = store.create(), store.create()
    two_hours_ago = time.time() - 7200
    os.utime(tmp_path / "jobs" / old, (two_hours_ago, two_hours_ago))
    (tmp_path / "jobs" / "keep-me").mkdir()
    assert store.purge_expired() == 1
    assert not (tmp_path / "jobs" / old).exists()
    assert (tmp_path / "jobs" / fresh).exists()
    assert (tmp_path / "jobs" / "keep-me").exists(), (
        "only directories the store created are removed"
    )


def test_purge_on_a_missing_root_is_a_no_op(tmp_path: Path) -> None:
    assert JobStore(tmp_path / "nowhere", 60).purge_expired() == 0
