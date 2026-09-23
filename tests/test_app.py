"""The web layer end to end, with fake models: status codes, headers, privacy, failure modes."""

from __future__ import annotations

import errno
import io
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient
from PIL import Image as PilImage

from tests.conftest import FakeDetector, FakeRegistry, png_bytes
from vision_lab import app as app_module
from vision_lab.app import create_app
from vision_lab.config import Settings

JOB_URL = re.compile(r"/jobs/([0-9a-f]{32})$")
IMAGE_SRC = re.compile(r'src="(/jobs/[^"]+)"')


def submit(client: FlaskClient, **fields: str) -> Any:
    data = {"image": (io.BytesIO(png_bytes()), "photo.png"), **fields}
    return client.post("/jobs", data=data, content_type="multipart/form-data")


def job_id_of(response: Any) -> str:
    match = JOB_URL.search(response.headers["Location"])
    assert match is not None
    return match.group(1)


def test_pages_render(client: FlaskClient) -> None:
    for path, text in [("/", "Process image"), ("/algorithms", "Canny"), ("/healthz", "ok")]:
        response = client.get(path)
        assert response.status_code == 200
        assert text in response.get_data(as_text=True)


def test_upload_redirects_to_a_result_page_whose_images_all_load(client: FlaskClient) -> None:
    response = submit(client, result_option="all")
    assert response.status_code == 303
    page = client.get(response.headers["Location"])
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "person" in html
    assert "0.91" in html
    assert "50.00 %" in html
    sources = IMAGE_SRC.findall(html)
    # 4 fixed images, then every variant: 2 filters, 5 edges, 3 equalisations,
    # 4 enhancements, 4 colour spaces.
    assert len(sources) == 4 + 2 + 5 + 3 + 4 + 4
    for source in sources:
        image = client.get(source)
        assert image.status_code == 200
        assert image.mimetype == "image/jpeg"
        assert image.headers["Cache-Control"] == "private, max-age=300"
        image.close()  # send_file keeps the file open until the response is closed


def test_single_option_renders_one_variant_per_group(client: FlaskClient) -> None:
    html = client.get(submit(client).headers["Location"]).get_data(as_text=True)
    assert len(IMAGE_SRC.findall(html)) == 4 + 5


def test_viewing_a_result_does_not_run_the_models_again(settings: Settings) -> None:
    calls: list[str] = []

    class Counting(FakeRegistry):
        def detector(self, name: str) -> FakeDetector:
            calls.append(name)
            return super().detector(name)

    client = create_app(settings, Counting(settings.weights_dir)).test_client()
    location = submit(client).headers["Location"]
    for _ in range(3):
        assert client.get(location).status_code == 200
    assert calls == ["fasterrcnn"]


def test_stored_original_has_no_metadata_and_no_client_file_name(
    client: FlaskClient, settings: Settings
) -> None:
    exif = PilImage.Exif()
    exif[0x010F] = "SecretCameraMaker"
    buffer = io.BytesIO()
    PilImage.new("RGB", (64, 48)).save(buffer, format="JPEG", exif=exif.tobytes())
    assert b"SecretCameraMaker" in buffer.getvalue()
    buffer.seek(0)
    data = {"image": (buffer, "../../evil name.jpg")}
    response = client.post("/jobs", data=data, content_type="multipart/form-data")
    original = settings.jobs_dir / job_id_of(response) / "original.jpg"
    assert b"SecretCameraMaker" not in original.read_bytes()
    assert not [path for path in settings.data_dir.rglob("*") if "evil" in path.name]


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("kernel_size", "8", "must be odd"),
        ("scale_factor", "1e9", "between 0.1 and 4.0"),
        ("edge_algorithm", "nope", "must be one of"),
    ],
)
def test_bad_parameters_give_400_and_keep_nothing(
    client: FlaskClient, settings: Settings, field: str, value: str, fragment: str
) -> None:
    response = submit(client, **{field: value})
    assert response.status_code == 400
    assert fragment in response.get_data(as_text=True)
    assert not settings.jobs_dir.exists()


@pytest.mark.parametrize("payload", [b"", b"MZ\x90\x00 not an image", b"<svg/>"])
def test_bad_files_give_400(client: FlaskClient, payload: bytes) -> None:
    data = {"image": (io.BytesIO(payload), "x.jpg")}
    response = client.post("/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_a_file_that_passes_header_validation_but_fails_to_decode_is_400(
    client: FlaskClient,
) -> None:
    """A truncated JPEG has a valid header (format, width, height all parse)
    but fails once the pixel data itself is actually decoded, which only
    happens inside the slot; this must still be a 400, not a 500."""
    buffer = io.BytesIO()
    PilImage.new("RGB", (200, 200)).save(buffer, format="JPEG")
    truncated = buffer.getvalue()[: len(buffer.getvalue()) // 2]
    data = {"image": (io.BytesIO(truncated), "x.jpg")}
    response = client.post("/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "could not be read" in response.get_data(as_text=True)


def test_missing_file_gives_400(client: FlaskClient) -> None:
    assert client.post("/jobs", data={}).status_code == 400


def test_oversized_upload_gives_413(client: FlaskClient) -> None:
    data = {"image": (io.BytesIO(b"\x00" * (300 * 1024)), "big.jpg")}
    response = client.post("/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 413
    assert "larger than" in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "headers",
    [
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "cross-site", "Origin": "http://localhost"},
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
    ],
)
def test_cross_site_uploads_are_refused_with_403(
    client: FlaskClient, settings: Settings, headers: dict[str, str]
) -> None:
    """CSP's form-action only governs pages this app serves; another site can
    still post to a reachable instance. Fetch Metadata (sent by every current
    browser) decides first; the Origin header is the fallback."""
    data = {"image": (io.BytesIO(png_bytes()), "photo.png")}
    response = client.post("/jobs", data=data, content_type="multipart/form-data", headers=headers)
    assert response.status_code == 403
    assert "another site" in response.get_data(as_text=True)
    assert not settings.jobs_dir.is_dir() or not any(settings.jobs_dir.iterdir())


@pytest.mark.parametrize(
    "headers",
    [
        {},  # curl and other API clients send neither header
        {"Sec-Fetch-Site": "same-origin"},
        {"Sec-Fetch-Site": "none"},  # typed into the address bar or a bookmark
        {"Origin": "http://localhost"},  # the test client's host
        {"Sec-Fetch-Site": "same-origin", "Origin": "http://localhost"},
        {"Origin": "http://LOCALHOST"},  # host names are case-insensitive
    ],
)
def test_same_site_and_headerless_uploads_are_accepted(
    client: FlaskClient, headers: dict[str, str]
) -> None:
    data = {"image": (io.BytesIO(png_bytes()), "photo.png")}
    response = client.post("/jobs", data=data, content_type="multipart/form-data", headers=headers)
    assert response.status_code == 303


def test_user_input_is_escaped(client: FlaskClient) -> None:
    response = submit(client, kernel_size="<script>alert(1)</script>")
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize(
    "path",
    [
        "/jobs/" + "0" * 32,
        "/jobs/not-a-job",
        "/jobs/..%2F..%2Fetc",
        "/jobs/" + "0" * 32 + "/files/original.jpg",
        "/nope",
    ],
)
def test_unknown_things_are_404(client: FlaskClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_a_file_deleted_between_the_check_and_send_is_404(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another thread's purge can remove the file after job_file's is_file() check."""
    location = submit(client).headers["Location"]

    def vanished(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("vision_lab.app.send_file", vanished)
    response = client.get(f"{location}/files/original.jpg")
    assert response.status_code == 404


def test_result_json_and_traversal_are_not_served(client: FlaskClient) -> None:
    job = submit(client).headers["Location"]
    for name in ["result.json", "..%2Fresult.json", "%2e%2e/%2e%2e/pyproject.toml"]:
        assert client.get(f"{job}/files/{name}").status_code == 404


def test_there_is_no_listing_of_other_peoples_jobs(client: FlaskClient) -> None:
    submit(client)
    assert client.get("/jobs").status_code == 405
    assert client.get("/results").status_code == 404
    assert client.get("/uploads/").status_code == 404


def test_wrong_method_is_405(client: FlaskClient) -> None:
    response = client.post("/")
    assert response.status_code == 405
    assert "Allow" in response.headers


def test_security_headers_on_every_response(client: FlaskClient) -> None:
    for response in (client.get("/"), client.get("/nope"), submit(client)):
        assert "script-src 'none'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "Set-Cookie" not in client.get("/").headers


def test_result_page_is_never_cached_but_its_images_may_be(client: FlaskClient) -> None:
    """A shared browser cache must not keep a result page past its TTL: once
    the job is purged, the page must be re-fetched, not served stale from a
    cache. The images it links to may still keep a short private cache."""
    location = submit(client).headers["Location"]
    page = client.get(location)
    assert page.headers["Cache-Control"] == "no-store"
    image = client.get(f"{location}/files/original.jpg")
    assert image.headers["Cache-Control"] == "private, max-age=300"
    image.close()


def test_image_cache_lifetime_never_exceeds_the_job_ttl(settings: Settings) -> None:
    """A browser must not keep an image in its private cache after the job it
    belongs to has been deleted; with a 1-minute TTL the cap is 60 s, not 300."""
    short = replace(settings, job_ttl_minutes=1)
    flask_app = create_app(short, FakeRegistry(short.weights_dir))
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    location = submit(client).headers["Location"]
    image = client.get(f"{location}/files/original.jpg")
    assert image.headers["Cache-Control"] == "private, max-age=60"
    image.close()


def test_pages_make_no_third_party_requests(client: FlaskClient) -> None:
    """The two external links in the footer (source and licence text) are places a
    reader can go, not resources the page fetches on its own; nothing else external
    should appear anywhere on the page."""
    for path in ["/", "/algorithms", submit(client).headers["Location"]]:
        html = client.get(path).get_data(as_text=True)
        external = re.findall(r'(?:src|href|action)="(https?://[^"]+)"', html)
        assert external == [
            "https://github.com/tudorandrian/vision-lab-flask",
            "https://www.gnu.org/licenses/agpl-3.0.html",
        ]


def test_footer_source_link_follows_the_configured_source_url(settings: Settings) -> None:
    """An operator who modifies and deploys this application must be able to point
    section 13's source link at their own fork, not the upstream repository."""
    forked = replace(settings, source_url="https://example.invalid/my-fork")
    flask_app = create_app(forked, FakeRegistry(forked.weights_dir))
    html = flask_app.test_client().get("/").get_data(as_text=True)
    assert 'href="https://example.invalid/my-fork"' in html
    assert "https://github.com/tudorandrian/vision-lab-flask" not in html


def test_model_failure_gives_a_500_page_without_details_and_cleans_up(
    settings: Settings,
) -> None:
    flask_app = create_app(settings, FakeRegistry(settings.weights_dir, fail=True))
    response = submit(flask_app.test_client())
    html = response.get_data(as_text=True)
    assert response.status_code == 500
    assert "Traceback" not in html
    assert "model exploded" not in html
    assert not list(settings.jobs_dir.iterdir())


def test_second_concurrent_job_gets_503_with_retry_after(settings: Settings) -> None:
    entered, release = threading.Event(), threading.Event()

    class Slow(FakeRegistry):
        def detector(self, name: str) -> FakeDetector:
            entered.set()
            release.wait(timeout=10)
            return super().detector(name)

    flask_app = create_app(settings, Slow(settings.weights_dir))
    first: list[int] = []
    worker = threading.Thread(
        target=lambda: first.append(submit(flask_app.test_client()).status_code)
    )
    worker.start()
    assert entered.wait(timeout=10)
    busy = submit(flask_app.test_client())
    release.set()
    worker.join(timeout=10)
    assert (busy.status_code, busy.headers["Retry-After"]) == (503, "10")
    busy_html = busy.get_data(as_text=True)
    assert "busy" in busy_html.lower()
    assert "Traceback" not in busy_html
    assert first == [303]
    assert submit(flask_app.test_client()).status_code == 303, "the slot is released afterwards"


def test_decoding_waits_for_the_inference_slot(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decoding a large image can use hundreds of MB; it must happen after the
    inference slot is acquired, or every waitress thread could decode at once.
    """
    settings = replace(settings, max_concurrent_jobs=1, queue_seconds=5.0)
    real_decode = app_module.decode_pending
    active = 0
    overlapped = False
    lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()

    def fake_decode(pending: object, **kwargs: object) -> Any:
        nonlocal active, overlapped
        with lock:
            active += 1
            if active > 1:
                overlapped = True
        entered.set()
        release.wait(timeout=10)
        try:
            return real_decode(pending, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("vision_lab.app.decode_pending", fake_decode)
    flask_app = create_app(settings, FakeRegistry(settings.weights_dir))
    results: list[int] = []
    lock2 = threading.Lock()

    def worker() -> None:
        status = submit(flask_app.test_client()).status_code
        with lock2:
            results.append(status)

    first = threading.Thread(target=worker)
    first.start()
    assert entered.wait(timeout=10), "the first request never reached decode"
    second = threading.Thread(target=worker)
    second.start()
    # Give a buggy (decode-before-slot) implementation a chance to let the
    # second request's decode start while the first is still blocked above;
    # the assertion below is unaffected by scheduling either way.
    time.sleep(0.3)
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not overlapped, "two decodes ran at the same time"
    assert sorted(results) == [303, 303]


def test_pre_slot_checks_stay_fast_while_the_slot_is_held(settings: Settings) -> None:
    """Only a genuinely valid upload should ever wait for a busy slot.

    A bad parameter, a bad file, and an oversized upload must all fail
    immediately, however long the slot is held, and however long the queue
    timeout is; otherwise a hostile or merely unlucky upload turns into a
    slow 503 ("server is busy") instead of an instant, specific 400/413.
    """
    settings = replace(settings, max_concurrent_jobs=1, queue_seconds=1.0)
    entered, release = threading.Event(), threading.Event()

    class Slow(FakeRegistry):
        def detector(self, name: str) -> FakeDetector:
            entered.set()
            release.wait(timeout=10)
            return super().detector(name)

    flask_app = create_app(settings, Slow(settings.weights_dir))
    holder = threading.Thread(target=lambda: submit(flask_app.test_client()))
    holder.start()
    assert entered.wait(timeout=10), "the holding request never reached the slot"

    probe = flask_app.test_client()
    budget = settings.queue_seconds / 2

    start = time.perf_counter()
    response = submit(probe, kernel_size="8")
    assert response.status_code == 400
    assert time.perf_counter() - start < budget, "a bad parameter must not wait for the slot"

    start = time.perf_counter()
    bad_file = {"image": (io.BytesIO(b"not an image"), "x.jpg")}
    response = probe.post("/jobs", data=bad_file, content_type="multipart/form-data")
    assert response.status_code == 400
    assert time.perf_counter() - start < budget, "a bad file must not wait for the slot"

    start = time.perf_counter()
    oversized = {"image": (io.BytesIO(b"\x00" * (300 * 1024)), "big.jpg")}
    response = probe.post("/jobs", data=oversized, content_type="multipart/form-data")
    assert response.status_code == 413
    assert time.perf_counter() - start < budget, "an oversized upload must not wait for the slot"

    start = time.perf_counter()
    response = submit(probe)
    elapsed = time.perf_counter() - start
    assert (response.status_code, response.headers["Retry-After"]) == (503, "10")
    assert elapsed >= settings.queue_seconds, "a valid upload must wait out the full queue timeout"

    release.set()
    holder.join(timeout=10)


def test_emotion_section_only_when_enabled(settings: Settings, app: Flask) -> None:
    off = app.test_client()
    html = off.get(submit(off).headers["Location"]).get_data(as_text=True)
    assert "Facial expression" not in html
    on = create_app(settings, FakeRegistry(settings.weights_dir, emotion=True)).test_client()
    assert "enabled on this server" in on.get("/").get_data(as_text=True)
    html = on.get(submit(on).headers["Location"]).get_data(as_text=True)
    assert "Facial expression" in html
    assert "neutral" in html


def test_expired_jobs_are_purged_on_the_next_upload(
    client: FlaskClient, settings: Settings
) -> None:
    old = submit(client)
    job_dir = settings.jobs_dir / job_id_of(old)
    past = time.time() - 2 * 3600
    for target in (job_dir, job_dir / "result.json"):
        os.utime(target, (past, past))
    submit(client)
    assert client.get(old.headers["Location"]).status_code == 404
    assert not job_dir.exists()


def test_expired_results_are_404_without_a_new_upload(
    client: FlaskClient, settings: Settings
) -> None:
    old = submit(client)
    location = old.headers["Location"]
    job_dir = settings.jobs_dir / job_id_of(old)
    past = time.time() - 2 * 3600
    for target in (job_dir, job_dir / "result.json"):
        os.utime(target, (past, past))
    assert client.get(location).status_code == 404
    assert client.get(f"{location}/files/original.jpg").status_code == 404
    assert not job_dir.exists()


def test_a_stuck_expired_job_directory_does_not_break_other_requests(
    client: FlaskClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One expired directory rmtree cannot remove must not turn every request into a 500.

    A held file handle (an antivirus or indexer on Windows, EBUSY/EROFS on Linux) or a
    PermissionError from another process mid-delete must be swallowed for that one directory;
    a fresh upload, and reading its page, must both still succeed.
    """
    old = submit(client)
    stuck_id = job_id_of(old)
    job_dir = settings.jobs_dir / stuck_id
    past = time.time() - 2 * 3600
    for target in (job_dir, job_dir / "result.json"):
        os.utime(target, (past, past))

    real_rmtree = shutil.rmtree

    def flaky_rmtree(path: object, *args: object, **kwargs: object) -> None:
        if Path(str(path)).name == stuck_id:
            raise OSError(errno.ENOTEMPTY, "directory not empty")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("vision_lab.storage.shutil.rmtree", flaky_rmtree)

    fresh = submit(client)
    assert fresh.status_code == 303
    assert client.get(fresh.headers["Location"]).status_code == 200
    assert client.get("/").status_code == 200


def test_a_result_read_during_a_slow_job_does_not_purge_that_job(settings: Settings) -> None:
    """With a 1-minute TTL and a cold model download, the job directory can be
    older than the TTL before result.json exists; a concurrent GET's purge
    must leave it alone, and the upload must still finish with 303."""
    settings = replace(settings, job_ttl_minutes=1)
    entered, release = threading.Event(), threading.Event()

    class Stalled(FakeRegistry):
        def detector(self, name: str) -> FakeDetector:
            entered.set()
            release.wait(timeout=10)
            return super().detector(name)

    flask_app = create_app(settings, Stalled(settings.weights_dir))
    flask_app.config["TESTING"] = True
    outcome: list[int] = []
    worker = threading.Thread(
        target=lambda: outcome.append(submit(flask_app.test_client()).status_code)
    )
    worker.start()
    try:
        assert entered.wait(timeout=10)
        (job_dir,) = [entry for entry in settings.jobs_dir.iterdir() if entry.is_dir()]
        past = time.time() - 600  # the directory looks ten minutes old, past the 1-minute TTL
        os.utime(job_dir, (past, past))
        assert flask_app.test_client().get(f"/jobs/{job_dir.name}").status_code == 404
        assert job_dir.exists(), "a read while the job is running must not delete it"
    finally:
        release.set()
        worker.join(timeout=10)
    assert outcome == [303]
    assert flask_app.test_client().get(f"/jobs/{job_dir.name}").status_code == 200


def test_the_500_log_redacts_the_job_id_but_keeps_the_route_readable(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """request.path on /jobs/<id> is a capability URL: the full id must not
    reach the log. But redacting it must not blind the log to which route
    failed: /jobs/<id> and /jobs/<id>/files/<name> must not read
    identically, and an unrelated route's path must survive untouched."""
    location = submit(client).headers["Location"]
    match = JOB_URL.search(location)
    assert match is not None
    job_id = match.group(1)

    def boom(self: object, job_id: str) -> None:
        raise RuntimeError("corrupted result.json")

    monkeypatch.setattr("vision_lab.storage.JobStore.load_result", boom)
    with caplog.at_level(logging.ERROR, logger="vision_lab"):
        page_response = client.get(location)
    assert page_response.status_code == 500
    page_log = caplog.text
    caplog.clear()

    def vanished(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("disk error")

    monkeypatch.setattr("vision_lab.app.send_file", vanished)
    with caplog.at_level(logging.ERROR, logger="vision_lab"):
        file_response = client.get(f"{location}/files/original.jpg")
    assert file_response.status_code == 500
    file_log = caplog.text

    assert job_id not in page_log, "the full job id (a capability URL) must not reach the log"
    assert job_id not in file_log
    redacted = f"/jobs/{job_id[:8]}..."
    assert redacted in page_log, "the route must stay identifiable after redaction"
    assert f"{redacted}/files/original.jpg" in file_log
    assert page_log.splitlines()[0] != file_log.splitlines()[0], (
        "the two routes must not log identically"
    )


def test_result_page_shows_the_actual_model_names_and_threshold(client: FlaskClient) -> None:
    html = client.get(submit(client).headers["Location"]).get_data(as_text=True)
    assert "Faster R-CNN, MobileNetV3-Large 320 FPN" in html
    assert "DeepLabV3, MobileNetV3-Large" in html
    assert "0.50" in html
