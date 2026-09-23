"""Unit tests for inference.py that need no model weights."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from vision_lab.inference import DeepFaceEmotionAnalyzer

QUIET_VARIABLES = ("DEEPFACE_HOME", "DEEPFACE_LOG_LEVEL", "TF_ENABLE_ONEDNN_OPTS")


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> logging.Logger:
    """Unset the variables the analyzer defaults, and restore them afterwards.

    setenv before delenv makes monkeypatch record the original state, so the
    values the analyzer writes into os.environ are undone after each test.
    """
    for name in QUIET_VARIABLES:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    logger = logging.getLogger("tensorflow")
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    return logger


def test_emotion_analyzer_silences_deepface_and_tensorflow_start_up_noise(
    clean_environment: logging.Logger, tmp_path: Path
) -> None:
    """DeepFace prints a deprecation banner on every import, and TensorFlow
    prints a oneDNN notice and a deprecation warning; none is actionable for a
    user of this application. They must be switched off before the import."""

    DeepFaceEmotionAnalyzer(tmp_path)

    assert os.environ["DEEPFACE_LOG_LEVEL"] == str(logging.ERROR)
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert clean_environment.level == logging.ERROR


def test_emotion_analyzer_keeps_log_levels_the_user_chose(
    clean_environment: logging.Logger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Someone debugging the emotion extra can still get the full output."""
    monkeypatch.setenv("DEEPFACE_LOG_LEVEL", str(logging.INFO))
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "1")
    clean_environment.setLevel(logging.DEBUG)

    DeepFaceEmotionAnalyzer(tmp_path)

    assert os.environ["DEEPFACE_LOG_LEVEL"] == str(logging.INFO)
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "1"
    assert clean_environment.level == logging.DEBUG
