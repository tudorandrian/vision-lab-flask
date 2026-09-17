"""Regenerate the README screenshots from a running server.

Usage: python scripts/screenshots.py http://127.0.0.1:8000
Writes docs/images/home.png and docs/images/result.jpg using samples/astronaut.jpg.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"


def main() -> int:
    base = sys.argv[1].rstrip("/")
    IMAGES.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto(base)
        page.screenshot(path=IMAGES / "home.png")
        page.set_input_files("#image", str(ROOT / "samples" / "astronaut.jpg"))
        page.get_by_role("button", name="Process image").click(timeout=300_000)
        page.wait_for_url("**/jobs/*", timeout=300_000)
        page.set_viewport_size({"width": 1200, "height": 1250})
        # Poll from Python: wait_for_function needs eval, which the page's CSP forbids.
        while not page.evaluate("[...document.images].slice(0, 4).every(i => i.complete)"):
            page.wait_for_timeout(100)
        page.screenshot(path=IMAGES / "result.jpg", type="jpeg", quality=85)
        browser.close()
    print("wrote", *sorted(path.name for path in IMAGES.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
