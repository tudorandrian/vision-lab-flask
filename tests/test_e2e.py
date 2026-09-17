"""A real browser against a live server. Opt in with: pytest -m e2e

Needs the browser once: uv run playwright install chromium
The server uses the fake models, so this checks the user journey, not the networks.
"""

from __future__ import annotations

import re
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from waitress.server import create_server

from tests.conftest import FakeRegistry
from vision_lab.app import create_app
from vision_lab.config import Settings

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e

SAMPLE = Path(__file__).parent.parent / "samples" / "astronaut.jpg"


@pytest.fixture(scope="module")
def live_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    data_dir = tmp_path_factory.mktemp("e2e")
    settings = Settings(data_dir=data_dir)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = create_app(settings, FakeRegistry(settings.weights_dir))
    server = create_server(app, host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.close()


def wait_for_images(page) -> None:  # type: ignore[no-untyped-def]
    """Poll from Python: page.wait_for_function needs eval, which script-src 'none' forbids."""
    for _ in range(100):
        if page.evaluate("[...document.images].every(image => image.complete)"):
            return
        page.wait_for_timeout(100)
    raise AssertionError("images did not finish loading")


def test_upload_journey_has_no_console_errors_and_no_broken_images(page, live_url: str) -> None:  # type: ignore[no-untyped-def]
    problems: list[str] = []
    page.on("console", lambda message: message.type == "error" and problems.append(message.text))
    page.on("requestfailed", lambda request: problems.append(request.url))
    external: list[str] = []
    page.on("request", lambda r: None if r.url.startswith(live_url) else external.append(r.url))

    page.goto(live_url)
    page.set_input_files("#image", str(SAMPLE))
    page.get_by_label("Every variant, for comparison").check()
    page.get_by_role("button", name="Process image").click()

    playwright.expect(page).to_have_url(re.compile(r"/jobs/[0-9a-f]{32}$"))
    playwright.expect(page.get_by_role("heading", level=1)).to_have_text("Results")
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    wait_for_images(page)
    assert page.evaluate("[...document.images].filter(i => i.naturalWidth === 0).length") == 0
    assert page.evaluate("document.images.length") == 22
    assert problems == []
    assert external == [], "the pages must not contact any third party"


def test_validation_error_is_shown_next_to_the_form(page, live_url: str) -> None:  # type: ignore[no-untyped-def]
    page.goto(live_url)
    page.set_input_files("#image", str(SAMPLE))
    page.eval_on_selector("#kernel_size", "input => { input.step = 'any'; input.value = '8'; }")
    page.get_by_role("button", name="Process image").click()
    playwright.expect(page.get_by_role("alert")).to_contain_text("must be odd")


@pytest.mark.parametrize("width", [375, 1280])
def test_no_horizontal_scrolling(page, live_url: str, width: int) -> None:  # type: ignore[no-untyped-def]
    page.set_viewport_size({"width": width, "height": 800})
    for path in ["/", "/algorithms"]:
        page.goto(live_url + path)
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
