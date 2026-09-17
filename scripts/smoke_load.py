"""Sequential and concurrent uploads against a running server; prints latency and error counts.

Usage: python scripts/smoke_load.py http://127.0.0.1:8000 samples/astronaut.jpg
Exit code 0 on a clean run. Exit code 1 if any response is a 5xx other than the deliberate 503
busy signal, or if no sequential upload succeeded at all. Exit code 2 if the target host is not
local, in which case no request is made.
Standard library only, so it runs anywhere Python does.
"""

from __future__ import annotations

import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_local(base: str) -> bool:
    parts = urllib.parse.urlsplit(base)
    return parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def upload(base: str, image: bytes) -> tuple[int, float]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="image"; filename="sample.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{base}/jobs",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.perf_counter()
    try:
        with _OPENER.open(request, timeout=300) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    return status, time.perf_counter() - started


def main() -> int:
    base, image = sys.argv[1].rstrip("/"), Path(sys.argv[2]).read_bytes()
    if not is_local(base):
        print("refusing to load-test a host that is not local")
        return 2
    upload(base, image)  # warm-up: loads the models
    sequential = [upload(base, image) for _ in range(10)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda _: upload(base, image), range(12)))

    times = sorted(seconds for status, seconds in sequential if status == 303)
    print(f"sequential: {Counter(status for status, _ in sequential)}")
    print(f"  p50 {statistics.median(times):.2f} s, max {times[-1]:.2f} s")
    print(f"concurrent x4: {Counter(status for status, _ in concurrent)}")
    print("  303 = processed, 503 = refused while busy (by design, with Retry-After)")
    unexpected = [s for s, _ in sequential + concurrent if s >= 500 and s != 503]
    return 1 if unexpected or not times else 0


if __name__ == "__main__":
    raise SystemExit(main())
