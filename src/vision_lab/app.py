"""Flask application factory and routes."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any
from urllib.parse import urlsplit

from flask import (
    Flask,
    Response,
    abort,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask.typing import ResponseReturnValue
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException

from vision_lab import __version__, catalog, params
from vision_lab.config import Settings
from vision_lab.inference import ModelRegistry
from vision_lab.pipeline import run_job
from vision_lab.storage import JobStore, UploadError, decode_pending, validate_upload

log = logging.getLogger("vision_lab")

# A job id in a path is a capability URL (anyone who has it can view that
# job); redact it in logs, but keep the rest of the path so the route that
# failed is still identifiable. Truncating the whole path instead (an
# earlier version of this) was worse on both counts: "/jobs/" alone is 6
# characters, so an 8-character truncation left only 2 hex characters of
# the id, and made /jobs/<id> and /jobs/<id>/files/<name> log identically.
_JOB_ID_IN_PATH = re.compile(r"[0-9a-f]{32}")

# Fetch Metadata values a browser sends for a request that this site itself
# initiated ("none" is a navigation typed or bookmarked by the user).
_ALLOWED_FETCH_SITES = {"same-origin", "same-site", "none"}


def _is_cross_site(headers: Headers, host: str) -> bool:
    """True when a browser reports that another site initiated this request.

    CSP's form-action restricts the pages this application serves; it cannot
    stop a page on another site from posting a form or a fetch() to a reachable
    instance and making it burn CPU and disk. Sec-Fetch-Site is authoritative
    when present (every current browser sends it). Without it, a mismatching
    Origin is the fallback; "null" (sandboxed or opaque origins) counts as
    another site. Only the host is compared, not the scheme, so a TLS
    terminating proxy in front does not break the check; the comparison is
    case-insensitive because host names are. A request without either header
    (curl, scripts, tests) is not a browser request and passes. `headers` is
    typed as `Headers` (Flask's `request.headers`, an `EnvironHeaders`) rather
    than a generic mapping so the lookup stays case-insensitive the way HTTP
    headers are meant to be read.
    """
    fetch_site = headers.get("Sec-Fetch-Site")
    if fetch_site:
        return fetch_site not in _ALLOWED_FETCH_SITES
    origin = headers.get("Origin")
    if origin:
        return origin == "null" or urlsplit(origin).netloc.lower() != host.lower()
    return False


_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'none'; img-src 'self' data:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


def create_app(settings: Settings | None = None, models: ModelRegistry | None = None) -> Flask:
    """Build the app. Tests pass their own settings and a registry of fake models."""
    settings = settings or Settings.from_env()
    registry = models or ModelRegistry(settings.weights_dir, settings.enable_emotion)
    store = JobStore(settings.jobs_dir, settings.job_ttl_minutes)
    slots = threading.BoundedSemaphore(settings.max_concurrent_jobs)

    # Images may sit in a private browser cache for a few minutes. The cache
    # lifetime is capped at the job TTL, so a cached copy can outlive the
    # deleted job by at most that cap, min(5 minutes, TTL), not indefinitely.
    image_max_age = min(300, settings.job_ttl_minutes * 60)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.config["MAX_FORM_MEMORY_SIZE"] = 64 * 1024
    app.config["MAX_FORM_PARTS"] = 40

    def form_page(status: int, errors: dict[str, str] | None = None) -> tuple[str, int]:
        page = render_template(
            "index.html",
            catalog=catalog,
            choices=params,
            detectors=registry.available_detectors(),
            emotion=registry.emotion_available(),
            errors=errors or {},
            limits=settings,
        )
        return page, status

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        return {"app_version": __version__, "source_url": settings.source_url}

    @app.after_request
    def harden(response: Response) -> Response:
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    @app.get("/")
    def home() -> tuple[str, int]:
        return form_page(200)

    @app.get("/algorithms")
    def algorithms() -> str:
        return render_template("algorithms.html", catalog=catalog.CATALOG)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/jobs")
    def submit() -> ResponseReturnValue:
        if _is_cross_site(request.headers, request.host):
            abort(403)
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return form_page(400, {"image": "Choose an image to upload."})
        try:
            chosen = params.parse_params(request.form, registry.available_detectors())
        except params.ParamError as error:
            return form_page(400, error.errors)

        # The cheap header check (format, dimensions) runs before the slot, so
        # a bad file returns 400 immediately, however busy the server is; only
        # the expensive part (a full pixel decode, which can use hundreds of
        # MB for a large photo) waits for the slot, below.
        try:
            pending = validate_upload(upload.stream, max_pixels=settings.max_pixels)
        except UploadError as error:
            return form_page(400, {"image": str(error)})

        # Wait briefly for the inference slot, then give up: a bounded queue keeps
        # memory and response times predictable however many uploads arrive at once.
        if not slots.acquire(timeout=settings.queue_seconds):
            page = render_template(
                "error.html",
                status=503,
                message="The server is busy processing another image. Please try again shortly.",
            )
            response = make_response(page, 503)
            response.headers["Retry-After"] = "10"
            return response
        try:
            try:
                image = decode_pending(pending, max_side=settings.max_side)
            except UploadError as error:
                return form_page(400, {"image": str(error)})
            store.purge_expired()
            job_id = store.create()
            try:
                run_job(store, registry, job_id, image, chosen, settings.max_side)
            except Exception:
                store.discard(job_id)
                raise
        finally:
            slots.release()
        return redirect(url_for("show_job", job_id=job_id), code=303)

    @app.get("/jobs/<job_id>")
    def show_job(job_id: str) -> Response:
        # A job past its TTL must read back as gone even if nobody has uploaded
        # since, so this cheap directory scan runs on every read too.
        store.purge_expired()
        try:
            result = store.load_result(job_id)
        except KeyError:
            abort(404)
        page = render_template(
            "result.html", job_id=job_id, result=result, ttl=settings.job_ttl_minutes
        )
        # A result page must not survive in a shared cache past the job's
        # TTL: once purged, a fresh request must always reach this handler
        # and get a 404, not a stale cached copy. Its images may still keep
        # the short private cache set on job_file below.
        response = make_response(page)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/jobs/<job_id>/files/<name>")
    def job_file(job_id: str, name: str) -> Response:
        store.purge_expired()
        try:
            path = store.file_path(job_id, name)
        except KeyError:
            abort(404)
        if not path.is_file():
            abort(404)
        try:
            # Another thread's purge_expired can remove the file between the
            # is_file() check above and send_file actually opening it.
            response = send_file(path, max_age=image_max_age)
        except FileNotFoundError:
            abort(404)
        response.headers["Cache-Control"] = f"private, max-age={image_max_age}"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> Response:
        status = error.code or 500
        messages = {
            404: "That page or result does not exist. Results are deleted after a while.",
            403: "This form cannot be submitted from another site.",
            405: "That action is not available here.",
            413: f"The upload is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.",
        }
        message = messages.get(status, error.description or "The request could not be handled.")
        page = render_template("error.html", status=status, message=message)
        response = make_response(page, status)
        # Werkzeug attaches Allow to its own 405 response; carry it over so a
        # client (or a test) can still see which methods are permitted.
        allow = error.get_response().headers.get("Allow")
        if allow:
            response.headers["Allow"] = allow
        return response

    @app.errorhandler(Exception)
    def unexpected(error: Exception) -> tuple[str, int]:
        redacted_path = _JOB_ID_IN_PATH.sub(lambda m: m.group()[:8] + "...", request.path)
        log.exception("unhandled error while serving %s", redacted_path)
        message = "Something went wrong while processing the image. Nothing was kept."
        return render_template("error.html", status=500, message=message), 500

    return app
