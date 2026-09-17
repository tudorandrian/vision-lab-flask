"""Upload decoding and per-job storage.

The client never chooses a path: a job is a random UUID4 (122 random bits), files
inside it have fixed names, and both are checked against strict patterns before
any disk access. Uploaded pixels are re-encoded, so EXIF data (GPS position,
device serial numbers) never reaches the disk.
"""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import threading
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
_logger = logging.getLogger("vision_lab")


class UploadError(ValueError):
    """The uploaded file is not an image this application accepts."""


def decode_upload(stream: IO[bytes], *, max_pixels: int, max_side: int) -> Image:
    """Return the upload as a BGR array, or raise UploadError.

    The header is inspected before any pixel is decoded, so an image that
    claims enormous dimensions (a decompression bomb) is rejected cheaply.

    Pillow's own decompression-bomb guard warns above MAX_IMAGE_PIXELS and
    raises above twice that. The warning is suppressed here because our own
    max_pixels check below is stricter and gives a clearer message; the
    error is still caught, for the rare header that is too extreme for
    Pillow to even measure without refusing outright.
    """
    data = stream.read()
    if not data:
        raise UploadError("The file is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PilImage.DecompressionBombWarning)
            picture = PilImage.open(io.BytesIO(data))
            # A phone camera JPEG carrying an MPF segment (a second, small
            # preview frame) is identified by Pillow as format "MPO", not
            # "JPEG". Only the first (primary) frame is used, by default.
            fmt = "JPEG" if picture.format == "MPO" else picture.format
            if fmt not in ALLOWED_FORMATS:
                raise UploadError("Only JPEG, PNG, WebP and BMP images are accepted.")
            if picture.width * picture.height > max_pixels:
                raise UploadError(f"The image has more than {max_pixels:,} pixels.")
            upright = ImageOps.exif_transpose(picture).convert("RGB")
    except UploadError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Warning,
        PilImage.DecompressionBombError,
    ) as error:
        raise UploadError("The file could not be read as an image.") from error
    bgr = cv2.cvtColor(np.asarray(upright, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return downscale_to(bgr, max_side)


class JobStore:
    """One directory per job under jobs_dir; directories older than the TTL are removed."""

    def __init__(self, jobs_dir: Path, ttl_minutes: int) -> None:
        self._root = jobs_dir
        self._ttl_seconds = ttl_minutes * 60
        # Every read purges expired jobs too (see app.py), so request threads
        # race here often. A lock keeps the listing and the removals of one
        # purge atomic with respect to each other; the per-entry try/except
        # below is a second line of defence, not the primary one.
        self._purge_lock = threading.Lock()

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
        """Delete jobs older than the TTL. Returns how many this call actually removed.

        Every read purges expired jobs too (see app.py), so several request
        threads can race here. The lock makes one purge's listing and removals
        atomic with respect to another purge in this process, which is the
        only place this method is ever called from; the per-entry try/except
        below is a second line of defence (for example against something
        outside this process touching the same directory), not the mechanism
        this relies on for correctness.
        """
        if not self._root.is_dir():
            return 0
        deadline = (time.time() if now is None else now) - self._ttl_seconds
        removed = 0
        with self._purge_lock:
            try:
                entries = list(self._root.iterdir())
            except FileNotFoundError:
                return 0
            for entry in entries:
                try:
                    is_expired = (
                        entry.is_dir()
                        and _JOB_ID.fullmatch(entry.name)
                        and entry.stat().st_mtime < deadline
                    )
                except FileNotFoundError:
                    continue  # already gone
                except OSError:
                    _logger.warning("could not check job directory %s", entry.name)
                    continue
                if not is_expired:
                    continue
                try:
                    shutil.rmtree(entry)
                except FileNotFoundError:
                    continue  # already gone
                except OSError:
                    # Another process mid-delete (PermissionError), or a handle held open
                    # by something else (an antivirus or indexer, EBUSY, WinError 145).
                    # One stuck directory must not stop the purge or the request it runs on.
                    _logger.warning("could not remove job directory %s", entry.name)
                    continue
                removed += 1
        return removed
