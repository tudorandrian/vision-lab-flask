from __future__ import annotations

import io
import os
import time
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
    exif = PilImage.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    exif[0x010F] = "SecretCameraMaker"
    image = decode_upload(encoded("JPEG", (64, 48), exif=exif.tobytes()), **LIMITS)
    assert image.shape[:2] == (64, 48), "portrait after applying the orientation tag"


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


def test_pixel_limit_is_checked_before_decoding() -> None:
    with pytest.raises(UploadError, match="pixels"):
        decode_upload(encoded("PNG", (2000, 2000)), **LIMITS)


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
