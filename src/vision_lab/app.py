"""Flask application factory and routes."""

from __future__ import annotations

import logging
import threading
from typing import Any

from flask import Flask, Response, abort, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException

from vision_lab import __version__, catalog, params
from vision_lab.config import Settings
from vision_lab.inference import ModelRegistry
from vision_lab.pipeline import run_job
from vision_lab.storage import JobStore, UploadError, decode_upload

log = logging.getLogger("vision_lab")

SOURCE_URL = "https://github.com/tudorandrian/vision-lab-flask"

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
        return {"app_version": __version__, "source_url": SOURCE_URL}

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
    def submit() -> Any:
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return form_page(400, {"image": "Choose an image to upload."})
        try:
            chosen = params.parse_params(request.form, registry.available_detectors())
            image = decode_upload(
                upload.stream, max_pixels=settings.max_pixels, max_side=settings.max_side
            )
        except params.ParamError as error:
            return form_page(400, error.errors)
        except UploadError as error:
            return form_page(400, {"image": str(error)})

        # Wait briefly for the inference slot, then give up: a bounded queue keeps
        # memory and response times predictable however many uploads arrive at once.
        if not slots.acquire(timeout=settings.queue_seconds):
            response = Response("The server is busy with another image. Try again shortly.", 503)
            response.headers["Retry-After"] = "10"
            return response
        try:
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
    def show_job(job_id: str) -> str:
        try:
            result = store.load_result(job_id)
        except KeyError:
            abort(404)
        return render_template(
            "result.html", job_id=job_id, result=result, ttl=settings.job_ttl_minutes
        )

    @app.get("/jobs/<job_id>/files/<name>")
    def job_file(job_id: str, name: str) -> Response:
        try:
            path = store.file_path(job_id, name)
        except KeyError:
            abort(404)
        if not path.is_file():
            abort(404)
        response = send_file(path, max_age=300)
        response.headers["Cache-Control"] = "private, max-age=300"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> tuple[str, int]:
        status = error.code or 500
        messages = {
            404: "That page or result does not exist. Results are deleted after a while.",
            405: "That action is not available here.",
            413: f"The upload is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.",
        }
        message = messages.get(status, error.description or "The request could not be handled.")
        return render_template("error.html", status=status, message=message), status

    @app.errorhandler(Exception)
    def unexpected(error: Exception) -> tuple[str, int]:
        log.exception("unhandled error while serving %s", request.path)
        message = "Something went wrong while processing the image. Nothing was kept."
        return render_template("error.html", status=500, message=message), 500

    return app
