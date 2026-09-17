"""The command line entry point: only the configuration-error path is tested here.

Every other branch starts a real waitress server, which is exercised by
Procedure BV / the container CI job instead.
"""

from __future__ import annotations

import sys

import pytest

from vision_lab.__main__ import main


def test_invalid_configuration_exits_readably_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["vision-lab"])
    monkeypatch.setenv("VISION_LAB_MAX_CONCURRENT_JOBS", "0")

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the server must not start with invalid configuration")

    monkeypatch.setattr("vision_lab.__main__.serve", fail_if_called)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    assert "VISION_LAB_MAX_CONCURRENT_JOBS" in error
    assert "Traceback" not in error
