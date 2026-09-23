"""Run pip-audit, retrying only when the vulnerability service itself failed.

pip-audit exits 1 both when it finds a vulnerability and when PyPI's API is down (an HTTP 5xx
surfaces as an uncaught ServiceError). A finding must fail CI at once; an outage deserves another
try. Every argument is passed to pip-audit unchanged:

    uv run python scripts/pip_audit_retry.py --skip-editable
"""

from __future__ import annotations

import subprocess
import sys
import time

ATTEMPTS = 3
WAIT_SECONDS = (20, 60)  # before the second and the third attempt

# What pip-audit writes to stderr when the service failed rather than the dependencies, checked
# against pip-audit 2.10 (pip_audit/_cli.py and pip_audit/_service/pypi.py): an HTTP error from
# PyPI, a connect timeout or redirect loop, and a read timeout or dropped connection.
SERVICE_FAILURES = (
    "pip_audit._service.interface.ServiceError",
    "your network may be blocking this service",
    "requests.exceptions.",
)
# The summary pip-audit prints when it found something: never retried.
FINDING = "known vulnerabilit"


def is_service_failure(stderr: str) -> bool:
    """True when a failed run was caused by the vulnerability service, not by a finding."""
    return FINDING not in stderr and any(marker in stderr for marker in SERVICE_FAILURES)


def main(argv: list[str]) -> int:
    for attempt in range(1, ATTEMPTS + 1):
        run = subprocess.run(  # noqa: S603 - runs pip-audit from this environment
            [sys.executable, "-m", "pip_audit", *argv],
            capture_output=True,
            text=True,
            check=False,
        )
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        if run.returncode == 0 or attempt == ATTEMPTS or not is_service_failure(run.stderr):
            return run.returncode
        wait = WAIT_SECONDS[attempt - 1]
        print(
            f"pip-audit: the vulnerability service failed (attempt {attempt} of {ATTEMPTS}); "
            f"retrying in {wait} s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(wait)
    raise AssertionError("unreachable: the last attempt always returns")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
