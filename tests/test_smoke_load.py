"""The load smoke script explains its own usage instead of failing with a traceback."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_load.py"
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "astronaut.jpg"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_load", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_prints_usage_and_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        load().main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "url" in out and "image" in out


def test_missing_arguments_print_usage_instead_of_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        load().main([])

    assert exit_info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_missing_image_file_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        load().main(["http://127.0.0.1:8000", str(tmp_path / "missing.jpg")])

    assert exit_info.value.code == 2
    assert "missing.jpg" in capsys.readouterr().err


def test_non_local_host_is_refused_without_any_request(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load()

    def no_upload(*args: object) -> None:
        raise AssertionError("no request may be made to a non-local host")

    monkeypatch.setattr(module, "upload", no_upload)

    assert module.main(["http://example.com", str(SAMPLE)]) == 2
    assert "not local" in capsys.readouterr().out
