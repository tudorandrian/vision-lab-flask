from __future__ import annotations

from pathlib import Path

from vision_lab.config import Settings


def test_defaults_are_conservative() -> None:
    settings = Settings.from_env({})
    assert settings.max_upload_bytes == 8 * 1024 * 1024
    assert settings.max_concurrent_jobs == 1
    assert settings.enable_emotion is False
    assert settings.data_dir.is_absolute()


def test_environment_overrides(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "VISION_LAB_DATA_DIR": str(tmp_path),
            "VISION_LAB_MAX_UPLOAD_MB": "2",
            "VISION_LAB_JOB_TTL_MINUTES": "5",
            "VISION_LAB_ENABLE_EMOTION": "Yes",
        }
    )
    assert settings.jobs_dir == tmp_path.resolve() / "jobs"
    assert settings.weights_dir == tmp_path.resolve() / "weights"
    assert (settings.max_upload_bytes, settings.job_ttl_minutes) == (2 * 1024 * 1024, 5)
    assert settings.enable_emotion is True
