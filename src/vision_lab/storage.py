"""Upload decoding and per-job storage.

The client never chooses a path: a job is a random 128-bit id, files inside it
have fixed names, and both are checked against strict patterns before any disk
access. Uploaded pixels are re-encoded, so EXIF data (GPS position, device
serial numbers) never reaches the disk.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np
from PIL import Image as PilImage
from PIL import ImageOps, UnidentifiedImageError

from vision_lab.imaging import Image
from vision_lab.ops import downscale_to

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP"})
_JOB_ID = re.compile(r"[0-9a-f]{32}")
_FILE_NAME = re.compile(r"[a-z0-9_]{1,40}\.(?:jpg|png)")


class UploadError(ValueError):
    """The uploaded file is not an image this application accepts."""


def decode_upload(stream: IO[bytes], *, max_pixels: int, max_side: int) -> Image:
    """Return the upload as a BGR array, or raise UploadError.

    The header is inspected before any pixel is decoded, so an image that
    claims enormous dimensions (a decompression bomb) is rejected cheaply.
    """
    data = stream.read()
    if not data:
        raise UploadError("The file is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PilImage.DecompressionBombWarning)
            picture = PilImage.open(io.BytesIO(data))
            if picture.format not in ALLOWED_FORMATS:
                raise UploadError("Only JPEG, PNG, WebP and BMP images are accepted.")
            if picture.width * picture.height > max_pixels:
                raise UploadError(f"The image has more than {max_pixels:,} pixels.")
            upright = ImageOps.exif_transpose(picture).convert("RGB")
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Warning) as error:
        raise UploadError("The file could not be read as an image.") from error
    bgr = cv2.cvtColor(np.asarray(upright, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return downscale_to(bgr, max_side)


class JobStore:
    """One directory per job under jobs_dir; directories older than the TTL are removed."""

    def __init__(self, jobs_dir: Path, ttl_minutes: int) -> None:
        self._root = jobs_dir
        self._ttl_seconds = ttl_minutes * 60

    def create(self) -> str:
        job_id = uuid.uuid4().hex
        (self._root / job_id).mkdir(parents=True)
        return job_id

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        return self._root / job_id

    def file_path(self, job_id: str, name: str) -> Path:
        if not _FILE_NAME.fullmatch(name):
            raise KeyError(name)
        return self._job_dir(job_id) / name

    def save_image(self, job_id: str, name: str, image: Image) -> str:
        path = self.file_path(job_id, name)
        encoded, buffer = cv2.imencode(path.suffix, image)
        if not encoded:
            raise RuntimeError(f"could not encode {name}")
        # imencode plus write_bytes also works where cv2.imwrite does not:
        # Windows paths that contain non-ASCII characters.
        path.write_bytes(buffer.tobytes())
        return name

    def save_result(self, job_id: str, result: dict[str, Any]) -> None:
        (self._job_dir(job_id) / "result.json").write_text(json.dumps(result), encoding="utf-8")

    def load_result(self, job_id: str) -> dict[str, Any]:
        path = self._job_dir(job_id) / "result.json"
        if not path.is_file():
            raise KeyError(job_id)
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def discard(self, job_id: str) -> None:
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def purge_expired(self, now: float | None = None) -> int:
        """Delete jobs older than the TTL. Returns how many were removed."""
        if not self._root.is_dir():
            return 0
        deadline = (time.time() if now is None else now) - self._ttl_seconds
        removed = 0
        for entry in self._root.iterdir():
            if (
                entry.is_dir()
                and _JOB_ID.fullmatch(entry.name)
                and entry.stat().st_mtime < deadline
            ):
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed
