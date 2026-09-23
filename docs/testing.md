# Testing

The question every level answers is the same: can a user, a careless client or a hostile one make
the application return something wrong, leak something, or fall over?

## Levels

| Level | Files | Runs in CI | Needs network |
| --- | --- | --- | --- |
| Static analysis | `pyproject.toml` (ruff, mypy strict, bandit), `scripts/check_text.py`, pip-audit, gitleaks | pushes to `main`, pull requests, and weekly | advisories only |
| Unit, quantitative | `tests/test_ops.py`, `test_params.py`, `test_render.py`, `test_storage.py`, `test_config.py` | Linux and Windows, Python 3.12 and 3.13 | no |
| Web layer with fake models | `tests/test_app.py` | same matrix | no |
| Real models | `tests/test_models.py` (`-m models`) | Linux, weights cached | first run |
| Browser | `tests/test_e2e.py` (`-m e2e`), Chromium through Playwright | Linux | first run, to install Chromium |
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
  locality and colour preservation, and NLM denoising is checked to keep a sharp edge sharp;
- an invalid environment variable (non-numeric, zero, negative, or NaN where a positive number is
  required) is rejected at start-up, naming the variable and its accepted range, instead of
  silently starting a broken server.

## What the web tests assert

- Validation: even kernels, NaN and infinite factors, unknown choices, markup in fields: HTTP 400,
  every error listed, nothing written to disk.
- Uploads: rejected by `validate_upload` (header, before the slot) or `decode_pending` (pixel
  decode, inside the slot), which the web layer turns into HTTP 400 (web tests cover empty files,
  executables and SVG; PDF, GIF, truncated JPEG and an oversized PNG over the pixel limit are
  covered through the `decode_upload` wrapper in `test_storage.py`). Oversized body: HTTP 413.
- Privacy: the stored original has no EXIF data; the client file name appears nowhere on disk;
  `/jobs`, `/results` and `/uploads/` list nothing; `result.json` is not served; a failed purge
  logs only the first eight characters of a job id; a job id inside a logged request path (a
  capability URL) is redacted to its first eight characters plus an ellipsis on an unhandled
  error, while the rest of the route stays readable.
- Traversal: encoded and plain `..` in identifiers and file names: HTTP 404.
- Robustness: a model that raises gives a generic HTTP 500 page without a traceback and leaves no
  files; a second upload during inference gets HTTP 503 with `Retry-After`, and the slot is free
  again afterwards; expired jobs disappear on the next upload.
- Queueing: only the upload's header (format, dimensions) is checked before the inference slot is
  acquired; a bad parameter, an unreadable file or an oversized upload all fail immediately with
  their specific 400 or 413, however long the slot is held, instead of waiting out the queue and
  coming back as a slow 503 (`test_pre_slot_checks_stay_fast_while_the_slot_is_held`).
- Cross-site protection: a `POST /jobs` carrying `Sec-Fetch-Site: cross-site` or a foreign
  `Origin` is refused with HTTP 403 and creates nothing; same-origin, `none` and headerless
  requests are accepted.
- Headers on every response, no cookies, no third-party URLs in any page.
- Viewing a result three times calls the detector once.
- Header-only decompression bombs are rejected cleanly on both sides of Pillow's own threshold,
  MPO camera JPEGs are accepted, and EXIF orientation is applied in the right direction; a purge
  under concurrent requests does not corrupt or double-delete a result.

## What the model tests assert

- The person in the sample image is found with confidence above 0.8, in a box covering more than
  25 % of the frame; its segmentation share of the same image is between 25 % and 90 %.
- Models are built once per process: viewing a result again does not run them a second time.
- Ultralytics' automatic package installation is off, and unsafe pickle loading of the YOLO
  checkpoint is disabled, before the library is imported.
- A hostile or careless environment cannot re-enable either guard: setting `YOLO_AUTOINSTALL=true`
  and `ULTRALYTICS_SAFE_LOAD=0` before `vision_lab.inference` is even imported still leaves both
  forced to their safe value, since the module sets them unconditionally rather than defaulting
  them.
- The model registry's weights directory follows `VISION_LAB_DATA_DIR` rather than a path
  hardcoded relative to the working directory.

## Reference measurements

Measured on 2026-09-17: Windows 10, Intel CPU from 2017 with 4 cores and 8 threads, 16 GB of
memory, no GPU, Python 3.13, torch 2.14 CPU, sample image 512 x 512 px, default options unless
stated, `yolo` extra installed, weights already downloaded from a previous run. The latency and
queue rows come from `scripts/smoke_load.py`; the cold-upload and memory rows come from `curl` and
the working set of the server's own python.exe process (found with `netstat -ano` for the port,
read with `(Get-Process -Id <pid>).WorkingSet64`) against a freshly started server. The Docker
image size is a separate measurement, taken from the built image itself rather than this Windows
hardware; see that row for its own date and base image.

| Measure | Value |
| --- | --- |
| Unit and web suite | 159 tests in about 11 s, 99.12 % line and branch coverage |
| Weights to download on first run (not re-measured; they are cached on this machine) | about 140 MB total (yolov5nu 5.3 MB, yolov5su 17.7 MB, DeepLabV3 42.3 MB, Faster R-CNN 74.2 MB) |
| Docker image size (`python:3.13-slim-bookworm` base, CPU wheels, no extras), built 2026-09-17 | 2.08 GB |
| First upload with every variant and YOLOv5su, models cold (weights already on disk) | 11.2 s |
| Upload with defaults, models warm, median of 10 | 0.60 s (maximum 0.73 s) |
| 12 uploads from 4 clients, queue of 15 s | 12 processed, none refused, no 5xx |
| The same with `VISION_LAB_QUEUE_SECONDS=0` | 1 processed, 11 refused with 503 and `Retry-After`, no 5xx, process stays up |
| Resident memory before any model is loaded | about 58 MB |
| Resident memory with Faster R-CNN and DeepLabV3 loaded (one default upload) | about 498 MB |
| Resident memory with YOLOv5nu also loaded | about 551 MB |
| Resident memory with YOLOv5su also loaded | about 655 MB |
| DeepLabV3 process time and peak working set at 512x512, models already loaded | 0.46 s, 504 MB peak |
| DeepLabV3 process time and peak working set at 1600x1600, the configured maximum, models already loaded | 3.91 s, 952 MB peak |

Repeat the load row after any change to `pipeline.py` or `inference.py` and update the table if a
value moves by more than a quarter. The two DeepLabV3 rows measure the cost of running segmentation
at the image's own size rather than the published 520 px transform; see
[docs/architecture.md](architecture.md#decisions) for why.

## Known limits

- The pretrained models are used as published. Their accuracy and bias are not evaluated here; on
  the sample image DeepLabV3 labels part of the helmet as `motorbike`, and the page shows that
  honestly.
- The `emotion` extra is covered by a fake in the web tests only. It needs TensorFlow, which is too
  large for the CI matrix; verify it by hand after changing `DeepFaceEmotionAnalyzer`.
- There is no rate limiting per client. The queue bounds the work the server accepts, not who
  sends it; put a reverse proxy in front before exposing the application.
