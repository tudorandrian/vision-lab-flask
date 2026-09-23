from __future__ import annotations

from pathlib import Path

import pytest

from vision_lab.config import Settings, SettingsError


def test_defaults_are_conservative() -> None:
    settings = Settings.from_env({})
    assert settings.max_upload_bytes == 8 * 1024 * 1024
    assert settings.max_concurrent_jobs == 1
    assert settings.queue_depth == 4
    assert settings.enable_emotion is False
    assert settings.data_dir.is_absolute()
    assert settings.source_url == "https://github.com/tudorandrian/vision-lab-flask"


def test_source_url_can_be_pointed_at_a_modified_deployment() -> None:
    """Section 13 of the AGPL asks a modified, network-deployed copy to offer its own
    source; this variable is how an operator points the footer link at it instead of
    editing the source default."""
    settings = Settings.from_env({"VISION_LAB_SOURCE_URL": "https://example.invalid/my-fork"})
    assert settings.source_url == "https://example.invalid/my-fork"


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


@pytest.mark.parametrize(
    ("env", "name"),
    [
        ({"VISION_LAB_JOB_TTL_MINUTES": "-5"}, "VISION_LAB_JOB_TTL_MINUTES"),
        ({"VISION_LAB_MAX_CONCURRENT_JOBS": "0"}, "VISION_LAB_MAX_CONCURRENT_JOBS"),
        ({"VISION_LAB_MAX_UPLOAD_MB": "0"}, "VISION_LAB_MAX_UPLOAD_MB"),
        ({"VISION_LAB_MAX_UPLOAD_MB": "1.5"}, "VISION_LAB_MAX_UPLOAD_MB"),
        ({"VISION_LAB_MAX_PIXELS": "0"}, "VISION_LAB_MAX_PIXELS"),
        ({"VISION_LAB_MAX_SIDE": "1"}, "VISION_LAB_MAX_SIDE"),
        ({"VISION_LAB_QUEUE_SECONDS": "nan"}, "VISION_LAB_QUEUE_SECONDS"),
        ({"VISION_LAB_QUEUE_SECONDS": "-1"}, "VISION_LAB_QUEUE_SECONDS"),
        ({"VISION_LAB_QUEUE_SECONDS": "soon"}, "VISION_LAB_QUEUE_SECONDS"),
        ({"VISION_LAB_QUEUE_SECONDS": "inf"}, "VISION_LAB_QUEUE_SECONDS"),
        ({"VISION_LAB_QUEUE_SECONDS": "1e999"}, "VISION_LAB_QUEUE_SECONDS"),
        ({"VISION_LAB_QUEUE_SECONDS": "3601"}, "VISION_LAB_QUEUE_SECONDS"),
        ({"VISION_LAB_QUEUE_DEPTH": "-1"}, "VISION_LAB_QUEUE_DEPTH"),
    ],
)
def test_invalid_settings_name_the_variable_and_are_rejected_at_start_up(
    env: dict[str, str], name: str
) -> None:
    with pytest.raises(SettingsError, match=name):
        Settings.from_env(env)


def test_zero_queue_seconds_is_accepted() -> None:
    """0 means never wait for a busy slot, a legitimate choice, not an error."""
    assert Settings.from_env({"VISION_LAB_QUEUE_SECONDS": "0"}).queue_seconds == 0


def test_a_negative_queue_seconds_message_does_not_mention_nan() -> None:
    """The NaN and the minimum checks are separate failures with separate
    messages; a negative-but-real number should not be told it might be NaN."""
    with pytest.raises(SettingsError) as excinfo:
        Settings.from_env({"VISION_LAB_QUEUE_SECONDS": "-1"})
    assert "NaN" not in str(excinfo.value)


def test_a_nan_queue_seconds_message_does_mention_nan() -> None:
    with pytest.raises(SettingsError, match="NaN"):
        Settings.from_env({"VISION_LAB_QUEUE_SECONDS": "nan"})


def test_an_infinite_queue_timeout_is_rejected_because_acquire_cannot_take_it() -> None:
    """threading.Lock.acquire(timeout=inf) raises OverflowError on CPython; the
    setting must be refused at start-up, not on the first busy upload."""
    with pytest.raises(SettingsError, match="finite"):
        Settings.from_env({"VISION_LAB_QUEUE_SECONDS": "inf"})
