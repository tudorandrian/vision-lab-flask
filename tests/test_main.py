"""The command line entry point: only the configuration-error path is tested here.

Every other branch starts a real waitress server, which is exercised by
Procedure BV / the container CI job instead.
"""

from __future__ import annotations

import signal
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


def test_sigterm_is_turned_into_a_graceful_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """docker stop, compose down and an orchestrator rollout all send SIGTERM.

    Python installs no handler for it when it is PID 1 in a container, so
    docker escalates to SIGKILL (exit 137) after the grace period. The
    handler must be installed before serve() starts, and must raise
    KeyboardInterrupt, which waitress's own run loop already closes on.
    """
    monkeypatch.setattr(sys, "argv", ["vision-lab"])
    registered: dict[int, object] = {}

    def fake_signal(signum: int, handler: object) -> None:
        registered[signum] = handler

    def fake_serve(*_args: object, **_kwargs: object) -> None:
        assert signal.SIGTERM in registered, "the handler must be installed before serve() runs"
        with pytest.raises(KeyboardInterrupt):
            registered[signal.SIGTERM](signal.SIGTERM, None)  # type: ignore[operator]

    monkeypatch.setattr("vision_lab.__main__.signal.signal", fake_signal)
    monkeypatch.setattr("vision_lab.__main__.serve", fake_serve)

    main()
