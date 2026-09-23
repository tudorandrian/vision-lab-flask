"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}

# The footer of every page links here, as section 13 of the AGPL asks of a
# modified version run over a network. Anyone who modifies this application
# and deploys it must point this at their own source, not the upstream
# repository: set VISION_LAB_SOURCE_URL rather than editing this default.
DEFAULT_SOURCE_URL = "https://github.com/tudorandrian/vision-lab-flask"


class SettingsError(ValueError):
    """An environment variable is missing a sane value: bad type or out of range.

    Raised from Settings.from_env only; the dataclass itself does not
    validate, so tests can still build a Settings directly with whatever
    values they need.
    """


def _int(source: Mapping[str, str], name: str, default: str, minimum: int) -> int:
    raw = source.get(name, default)
    try:
        value = int(raw)
    except ValueError as error:
        raise SettingsError(
            f"{name}={raw!r} is not a whole number; it must be an integer >= {minimum}."
        ) from error
    if value < minimum:
        raise SettingsError(f"{name}={value} must be an integer >= {minimum}.")
    return value


def _float(
    source: Mapping[str, str], name: str, default: str, minimum: float, maximum: float
) -> float:
    raw = source.get(name, default)
    try:
        value = float(raw)
    except ValueError as error:
        raise SettingsError(
            f"{name}={raw!r} is not a number; it must be a number between {minimum} and {maximum}."
        ) from error
    if math.isnan(value):
        raise SettingsError(f"{name}={raw!r} must be a real number, not NaN.")
    if math.isinf(value):
        # threading.Lock.acquire(timeout=inf) raises OverflowError on CPython.
        raise SettingsError(f"{name}={raw!r} must be a finite number, at most {maximum}.")
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name}={value} must be a number between {minimum} and {maximum}.")
    return value


@dataclass(frozen=True)
class Settings:
    """Limits and switches. Every limit exists to keep one request from hurting the host."""

    data_dir: Path
    max_upload_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 25_000_000
    max_side: int = 1600
    job_ttl_minutes: int = 60
    max_concurrent_jobs: int = 1
    queue_seconds: float = 15.0
    queue_depth: int = 4
    enable_emotion: bool = False
    source_url: str = DEFAULT_SOURCE_URL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build Settings from environment variables, or raise SettingsError.

        Every numeric variable is checked for both a parseable type and a
        sane minimum, so a typo or a nonsense value fails fast at start-up
        with the offending variable name and its accepted range, instead of
        starting a server that is silently broken (a 0-minute TTL that
        deletes a job before its page loads, a 0-slot queue that refuses
        every upload, and so on).
        """
        source = os.environ if env is None else env
        max_upload_mb = _int(source, "VISION_LAB_MAX_UPLOAD_MB", "8", minimum=1)
        return cls(
            data_dir=Path(source.get("VISION_LAB_DATA_DIR", "instance")).resolve(),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            max_pixels=_int(source, "VISION_LAB_MAX_PIXELS", "25000000", minimum=1),
            max_side=_int(source, "VISION_LAB_MAX_SIDE", "1600", minimum=64),
            job_ttl_minutes=_int(source, "VISION_LAB_JOB_TTL_MINUTES", "60", minimum=1),
            max_concurrent_jobs=_int(source, "VISION_LAB_MAX_CONCURRENT_JOBS", "1", minimum=1),
            queue_seconds=_float(source, "VISION_LAB_QUEUE_SECONDS", "15", minimum=0, maximum=3600),
            queue_depth=_int(source, "VISION_LAB_QUEUE_DEPTH", "4", minimum=0),
            enable_emotion=source.get("VISION_LAB_ENABLE_EMOTION", "0").strip().lower() in _TRUE,
            source_url=source.get("VISION_LAB_SOURCE_URL", DEFAULT_SOURCE_URL),
        )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def weights_dir(self) -> Path:
        return self.data_dir / "weights"
