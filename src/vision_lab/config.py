"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}


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
    enable_emotion: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        return cls(
            data_dir=Path(source.get("VISION_LAB_DATA_DIR", "instance")).resolve(),
            max_upload_bytes=int(source.get("VISION_LAB_MAX_UPLOAD_MB", "8")) * 1024 * 1024,
            max_pixels=int(source.get("VISION_LAB_MAX_PIXELS", "25000000")),
            max_side=int(source.get("VISION_LAB_MAX_SIDE", "1600")),
            job_ttl_minutes=int(source.get("VISION_LAB_JOB_TTL_MINUTES", "60")),
            max_concurrent_jobs=int(source.get("VISION_LAB_MAX_CONCURRENT_JOBS", "1")),
            queue_seconds=float(source.get("VISION_LAB_QUEUE_SECONDS", "15")),
            enable_emotion=source.get("VISION_LAB_ENABLE_EMOTION", "0").strip().lower() in _TRUE,
        )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def weights_dir(self) -> Path:
        return self.data_dir / "weights"
