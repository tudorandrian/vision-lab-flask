# Testing

The question every level answers is the same: can a user, a careless client or a hostile one make
the application return something wrong, leak something, or fall over?

## Levels

| Level | Files | Runs in CI | Needs network |
| --- | --- | --- | --- |
| Static analysis | `pyproject.toml` (ruff, mypy strict, bandit), `scripts/check_text.py`, pip-audit, gitleaks | every push | advisories only |
| Unit, quantitative | `tests/test_ops.py`, `test_params.py`, `test_render.py`, `test_storage.py`, `test_config.py` | Linux and Windows, Python 3.12 and 3.13 | no |
| Web layer with fake models | `tests/test_app.py` | same matrix | no |
| Real models | `tests/test_models.py` (`-m models`) | Linux, weights cached | first run |
| Browser | `tests/test_e2e.py` (`-m e2e`), Chromium through Playwright | Linux | no |
| Container | `Dockerfile` built, started, health checked, user id checked | Linux | build only |
| Load | `scripts/smoke_load.py` | manual, before a release | no |

## What the unit tests assert

Expectations are numbers on synthetic images, so a wrong algorithm fails even when it still
returns an image:

- every edge detector responds on all four sides of a white square and nowhere else, which the
  one-direction Sobel of the coursework version would fail;
- a median filter restores more than 99 % of pixels hit by 5 % salt-and-pepper noise;
- AHE yields more contrast than CLAHE on the same ramp, and both more than the input;
- brightness and contrast at factor 1.0 are the identity; denoising halves the noise deviation;
- rotation by 45 degrees grows the canvas to 141 px, by 360 degrees changes nothing;
- crops outside the image are clamped to at least one pixel instead of crashing;
- rotation is also checked exactly against `np.rot90` for the 90 degree steps, AHE is checked for
  locality and colour preservation, and NLM denoising is checked to keep a sharp edge sharp.

## What the web tests assert

- Validation: even kernels, NaN and infinite factors, unknown choices, markup in fields: HTTP 400,
  every error listed, nothing written to disk.
- Uploads: empty files, executables, SVG, PDF, GIF, truncated JPEG, 4-megapixel PNG over the pixel
  limit: HTTP 400. Oversized body: HTTP 413.
- Privacy: the stored original has no EXIF data; the client file name appears nowhere on disk;
  `/jobs`, `/results` and `/uploads/` list nothing; `result.json` is not served.
- Traversal: encoded and plain `..` in identifiers and file names: HTTP 404.
- Robustness: a model that raises gives a generic HTTP 500 page without a traceback and leaves no
  files; a second upload during inference gets HTTP 503 with `Retry-After`, and the slot is free
  again afterwards; expired jobs disappear on the next upload.
- Headers on every response, no cookies, no third-party URLs in any page.
- Viewing a result three times calls the detector once.
- Header-only decompression bombs are rejected cleanly on both sides of Pillow's own threshold,
  MPO camera JPEGs are accepted, and EXIF orientation is applied in the right direction; a purge
  under concurrent requests does not corrupt or double-delete a result.

## Reference measurements

Measured on 2026-09-17 with `scripts/smoke_load.py`: Windows 10, Intel CPU from 2017 with 4 cores
and 8 threads, 16 GB of memory, no GPU, Python 3.13, torch 2.14 CPU, sample image 512 x 512 px,
default options, `yolo` extra installed.

| Measure | Value |
| --- | --- |
| Unit and web suite | 134 tests in about 5 s, 98.71 % line and branch coverage |
| First upload, including weight download | about 25 s |
| Upload with every variant and YOLOv5su, models cold | 10.0 s |
| Upload with defaults, models warm, median of 10 | 0.60 s (maximum 0.73 s) |
| 12 uploads from 4 clients, queue of 15 s | 12 processed, none refused, no 5xx |
| The same with `VISION_LAB_QUEUE_SECONDS=0` | 1 processed, 11 refused with 503 and `Retry-After`, no 5xx, process stays up |
| Resident memory with Faster R-CNN and DeepLabV3 loaded | about 580 MB |
| Resident memory with both YOLO variants loaded as well | about 700 MB |

Repeat the load row after any change to `pipeline.py` or `inference.py` and update the table if a
value moves by more than a quarter.

## Known limits

- The pretrained models are used as published. Their accuracy and bias are not evaluated here; on
  the sample image DeepLabV3 labels part of the helmet as `motorbike`, and the page shows that
  honestly.
- The `emotion` extra is covered by a fake in the web tests only. It needs TensorFlow, which is too
  large for the CI matrix; verify it by hand after changing `DeepFaceEmotionAnalyzer`.
- There is no rate limiting per client. The queue bounds the work the server accepts, not who
  sends it; put a reverse proxy in front before exposing the application.
