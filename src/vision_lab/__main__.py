"""Command line entry point: serve the app with waitress, a production WSGI server."""

from __future__ import annotations

import argparse
import logging
import sys

from waitress import serve

from vision_lab.app import create_app
from vision_lab.config import Settings, SettingsError


def main() -> None:
    parser = argparse.ArgumentParser(prog="vision-lab", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: loopback)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = Settings.from_env()
    except SettingsError as error:
        # A clear, one-line message naming the offending variable, not a
        # traceback: this runs before any request is served.
        print(f"vision-lab: invalid configuration: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    logging.getLogger("vision_lab").info(
        "serving on http://%s:%d, data in %s", args.host, args.port, settings.data_dir
    )
    # One inference runs at a time (see Settings.max_concurrent_jobs); the extra
    # threads keep pages, images and the busy response fast while it does.
    #
    # waitress defaults max_request_body_size to 1 GB and buffers the whole body
    # (spilling to a temp file) before Flask's MAX_CONTENT_LENGTH can answer 413,
    # so an oversized upload would otherwise be written to disk first. Bounding
    # it here at the upload limit plus a small margin (multipart framing and the
    # other form fields) makes waitress itself refuse it, cheaply.
    serve(
        create_app(settings),
        host=args.host,
        port=args.port,
        threads=8,
        ident="vision-lab",
        max_request_body_size=settings.max_upload_bytes + 1024 * 1024,
    )


if __name__ == "__main__":
    main()
