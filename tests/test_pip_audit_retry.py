"""The CI wrapper around pip-audit retries outages, never findings."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pip_audit_retry.py"

OUTAGE = (
    "Traceback (most recent call last):\n"
    "requests.exceptions.HTTPError: 503 Server Error: Backend is unhealthy\n"
    "pip_audit._service.interface.ServiceError\n"
)
FINDING = "Found 1 known vulnerability in 1 package\n"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pip_audit_retry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def retry(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load()
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def scripted(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, runs: list[tuple[int, str]]
) -> list[list[str]]:
    calls: list[list[str]] = []
    outcomes = iter(runs)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        code, stderr = next(outcomes)
        return subprocess.CompletedProcess(command, code, stdout="", stderr=stderr)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return calls


def test_an_outage_is_retried_until_the_audit_succeeds(
    retry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = scripted(monkeypatch, retry, [(1, OUTAGE), (0, "No known vulnerabilities found\n")])
    assert retry.main(["--skip-editable"]) == 0
    assert len(calls) == 2
    assert calls[0][-1] == "--skip-editable", "arguments reach pip-audit unchanged"


def test_a_finding_fails_at_once_without_a_retry(
    retry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = scripted(monkeypatch, retry, [(1, FINDING)])
    assert retry.main([]) == 1
    assert len(calls) == 1


def test_a_lasting_outage_fails_after_the_last_attempt(
    retry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = scripted(monkeypatch, retry, [(1, OUTAGE)] * retry.ATTEMPTS)
    assert retry.main([]) == 1
    assert len(calls) == retry.ATTEMPTS


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (OUTAGE, True),
        ("Tip: your network may be blocking this service. Try another service\n", True),
        ("requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='pypi.org')\n", True),
        (FINDING, False),
        (OUTAGE + FINDING, False),  # a finding is never hidden behind a retry
        ("ERROR: invalid requirements input: nonsense\n", False),
    ],
)
def test_only_service_failures_count_as_outages(stderr: str, expected: bool) -> None:
    assert load().is_service_failure(stderr) is expected
