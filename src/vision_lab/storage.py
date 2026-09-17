"""Upload decoding and per-job storage.

The client never chooses a path: a job is a random UUID4 (122 random bits), files
inside it have fixed names, and both are checked against strict patterns before
any disk access. Uploaded pixels are re-encoded, so EXIF data (GPS position,
device serial numbers) is never stored in the job directory. It may still be
spooled to a temporary file elsewhere on disk while the request body is
received: waitress and werkzeug spool large uploads to a temp file before this
module ever sees them.
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
from dataclasses import dataclass
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


_UNREADABLE = (
    UnidentifiedImageError,
    OSError,
    ValueError,
    Warning,
    PilImage.DecompressionBombError,
)


@dataclass
class PendingUpload:
    """A header-checked upload, not yet decoded to pixel data.

    Produced by validate_upload, the cheap part: it reads the (already
    size-bounded) request body, opens just enough of the file to see its
    format and dimensions, and raises UploadError for anything this
    application will not accept. It does not force Pillow to decode any
    pixel data, so it allocates nothing beyond the encoded bytes already
    read from the stream.

    Consumed by decode_pending, the expensive part: EXIF-orientation and
    colour-space conversion force a full decode, which for a large photo
    can use hundreds of MB. The caller decides when that cost is paid, for
    example only after an inference slot is held (see app.py).
    """

    _picture: PilImage.Image


def validate_upload(stream: IO[bytes], *, max_pixels: int) -> PendingUpload:
    """Check the upload's format and pixel count from its header, or raise UploadError.

    The header is inspected before any pixel is decoded, so an image that
    claims enormous dimensions (a decompression bomb) is rejected cheaply,
    and a request with a bad file never needs to wait for the inference
    slot: the cost of this check is bounded by the upload size limit, not
    by decoding.

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
    except UploadError:
        raise
    except _UNREADABLE as error:
        raise UploadError("The file could not be read as an image.") from error
    return PendingUpload(picture)


def decode_pending(pending: PendingUpload, *, max_side: int) -> Image:
    """Decode a validated upload to a BGR array, or raise UploadError.

    This is the part that can use hundreds of MB for a large photo: EXIF
    orientation and the RGB conversion below both force Pillow to decode
    every pixel. Call this only once any queueing this application does is
    already accounted for (see app.py), not while only a header has been
    checked.
    """
    picture = pending._picture
    try:
        # For a JPEG, ask the decoder for a version already close to the
        # size we will downscale to anyway: libjpeg can decode at 1/2, 1/4
        # or 1/8 scale directly, which lowers the peak memory of decoding a
        # large photo. draft() is a no-op for every other format (the base
        # Image class defines it as such), so this is safe to call
        # unconditionally. For a large JPEG the drafted decode is a close
        # but not pixel-exact approximation of a full decode: measured here
        # with the bundled sample image upscaled to 4000x4000 and
        # re-encoded (quality 90), then both decodes downscaled the same
        # way afterward, the maximum absolute channel difference was 5
        # (mean 0.21). The sample image itself, at its native 512x512
        # (well under this application's default max_side of 1600, so
        # draft() has nothing to do), decodes bit-identical either way.
        picture.draft("RGB", (max_side, max_side))
        upright = ImageOps.exif_transpose(picture).convert("RGB")
    except _UNREADABLE as error:
        raise UploadError("The file could not be read as an image.") from error
    bgr = cv2.cvtColor(np.asarray(upright, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return downscale_to(bgr, max_side)


def decode_upload(stream: IO[bytes], *, max_pixels: int, max_side: int) -> Image:
    """Validate and decode an upload in one call: validate_upload then decode_pending.

    Kept as a thin wrapper with this signature for callers, and tests, that
    have no reason to split the cheap check from the expensive decode.
    """
    return decode_pending(validate_upload(stream, max_pixels=max_pixels), max_side=max_side)


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
                    _logger.warning("could not check job directory %s...", entry.name[:8])
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
                    # Only a short prefix of the job id is logged: the full id is a
                    # capability URL (anyone who has it can view or was meant to view
                    # that job), so it should not sit in full in a log file.
                    _logger.warning("could not remove job directory %s...", entry.name[:8])
                    continue
                removed += 1
        return removed
