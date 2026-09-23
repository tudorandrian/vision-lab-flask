# vision-lab-flask

A small web application that applies classical image processing and pretrained deep-learning
vision models to an uploaded image, and shows every result next to the numbers behind it.
Status: version 1.0.0, beta, maintained as a portfolio project.

[![CI](https://github.com/tudorandrian/vision-lab-flask/actions/workflows/ci.yml/badge.svg)](https://github.com/tudorandrian/vision-lab-flask/actions/workflows/ci.yml)
[![Licence: AGPL-3.0-or-later](https://img.shields.io/badge/licence-AGPL--3.0--or--later-blue.svg)](LICENSE)
![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)

![Result page: the uploaded image, its geometric transform and the detected objects](docs/images/result.jpg)

## What it does

| Area | Operations |
| --- | --- |
| Geometric transform | rotation on a growing canvas, scaling, cropping, flipping |
| Smoothing | Gaussian blur, median blur |
| Edge detection | Canny, Sobel, Scharr, Roberts cross, Laplacian of Gaussian |
| Histogram equalisation | global, adaptive (AHE), contrast-limited adaptive (CLAHE), on the lightness channel |
| Enhancement | unsharp mask, non-local means denoising, brightness, contrast |
| Colour spaces | HSV, CIE L\*a\*b\*, greyscale and RGB, shown as separate channel planes |
| Object detection | Faster R-CNN, MobileNetV3-Large 320 FPN (default, low resolution for CPU speed), YOLOv5u (optional extra) |
| Semantic segmentation | DeepLabV3 over the 21 Pascal VOC classes, with the pixel share of each class |
| Facial expression estimate | DeepFace (optional extra, disabled by default) |

Every operation is described, with its reference, in [docs/algorithms.md](docs/algorithms.md)
and in the application under Algorithms. Both are generated from one table in the code. Each
pretrained model's weights, training data and known limits are in [docs/models.md](docs/models.md).

## Run it

Three independent ways. Pick one; none of them needs Anaconda.

You need [Git](https://git-scm.com/downloads), and for route A, [uv](https://docs.astral.sh/uv/):
install it with `winget install --id=astral-sh.uv -e` (Windows), `brew install uv` (macOS), or
`curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux), then open a new terminal. Route B needs
Docker Desktop (Windows, macOS) or Docker Engine with the Compose plugin (Linux). Route C needs
Python 3.12 or 3.13 already installed.

Supported platforms: Windows 10 or 11 on x64, Linux on x86_64 or aarch64 with glibc 2.28 or newer
(Ubuntu 20.04, Debian 10 and later), and macOS 14 or newer on Apple silicon. Current PyTorch
releases publish no wheels for Intel Macs, macOS 13 or older, or a glibc older than 2.28. Allow
about 1.2 GB of disk space for `.venv` plus about 120 MB for the model weights.

### A. With uv (recommended)

[uv](https://docs.astral.sh/uv/) is a single executable that installs the right Python version
and the locked dependencies into a project-local `.venv`.

```bash
git clone https://github.com/tudorandrian/vision-lab-flask.git
cd vision-lab-flask
uv sync
uv run vision-lab
```

Open <http://127.0.0.1:8000>. The first upload downloads about 120 MB of model weights (Faster
R-CNN and DeepLabV3) into `instance/weights`, about 23 MB more with the `yolo` extra installed;
later runs work offline. Stop the server with Ctrl+C. If port 8000 is already taken, pass
`--host`/`--port`, for example `uv run vision-lab --port 8010`.

### B. With Docker

```bash
docker compose up --build
```

Open <http://127.0.0.1:8000>. Weights and results live in the `vision-lab-data` volume. The
built image measures 2.08 GB. Stop with Ctrl+C, or `docker compose down` from another terminal;
`docker compose down -v` also deletes the volume with the weights and results.

### C. With plain pip

Requires Python 3.12 or 3.13 (`pyproject.toml` sets `requires-python`).

```bash
python3 -m venv .venv
.venv/bin/pip install . --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/vision-lab
```

On Windows the two commands are `.venv\Scripts\pip` and `.venv\Scripts\vision-lab`, and the first
command is `python` instead of `python3`.
This route resolves dependencies afresh instead of using `uv.lock`.

### Optional extras

Each extra is optional and can be combined with the other. `uv sync` installs exactly the extras
you name and removes any it does not find, so running `uv sync --extra emotion` after
`uv sync --extra yolo` removes `yolo` again; to keep both, combine them in one command:
`uv sync --extra yolo --extra emotion`.

| Extra | Adds | Install | Notes |
| --- | --- | --- | --- |
| `yolo` | YOLOv5nu and YOLOv5su detectors | `uv sync --extra yolo` | Ultralytics code and weights are AGPL-3.0. Usage analytics, automatic package installation and unsafe pickle loading are all switched off in code. |
| `emotion` | facial expression estimate | `uv sync --extra emotion`, then set `VISION_LAB_ENABLE_EMOTION=1` | Pulls in TensorFlow, a large download. Read the privacy section first. |

## Configuration

Environment variables, all optional.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VISION_LAB_DATA_DIR` | `instance` | where weights and results are stored |
| `VISION_LAB_MAX_UPLOAD_MB` | `8` | larger uploads get HTTP 413 |
| `VISION_LAB_MAX_PIXELS` | `25000000` | larger images are rejected before decoding |
| `VISION_LAB_MAX_SIDE` | `1600` | images are reduced to this long side before processing |
| `VISION_LAB_JOB_TTL_MINUTES` | `60` | results are deleted this long after their processing finished |
| `VISION_LAB_MAX_CONCURRENT_JOBS` | `1` | inferences running at the same time |
| `VISION_LAB_QUEUE_SECONDS` | `15` | how long an upload waits for a free slot before HTTP 503 (0 to 3600) |
| `VISION_LAB_QUEUE_DEPTH` | `4` | how many uploads may wait for a slot at once; further uploads get HTTP 503 immediately |
| `VISION_LAB_ENABLE_EMOTION` | `0` | enables the `emotion` extra when it is installed |
| `VISION_LAB_SOURCE_URL` | this repository | the link in the footer of every page; point it at your own fork if you modify and deploy the application (see Licences and credits) |

Set a variable for one run with `VISION_LAB_MAX_SIDE=1024 uv run vision-lab` (bash, macOS or
Linux), or `$env:VISION_LAB_MAX_SIDE = "1024"; uv run vision-lab` (PowerShell). For Docker, add it
under `environment:` in `compose.yaml`. `VISION_LAB_DATA_DIR` is relative to the directory you
start the server from.

## Troubleshooting

- `uv: command not found` after installing uv: open a new terminal so the updated PATH is read.
- `ImportError: libGL.so.1` on a minimal Linux or WSL install: `sudo apt install libgl1 libglib2.0-0`.
- The first upload takes a while: the model weights are downloading (about 120 MB); the log shows
  the progress. Later uploads reuse the cached weights and are fast.
- `address already in use`: another program is using port 8000; start with `--port 8010`.
- No internet access when a model is used for the first time: each detector or segmenter needs
  network access once, to download its weights; without it, the upload fails and the server log
  names the model. Copy an existing `instance/weights` directory (or set `VISION_LAB_DATA_DIR` to
  one) to reuse weights downloaded elsewhere.
- A plain-text page naming a byte limit instead of the usual styled error page: the upload was far
  larger than `VISION_LAB_MAX_UPLOAD_MB`, so waitress itself refused the body before Flask ever
  saw the request; see [docs/architecture.md](docs/architecture.md#decisions) for why.

## How it is tested

| Level | What it proves | Command |
| --- | --- | --- |
| Static | style, formatting, types, common security mistakes, no em dash, the algorithms page matches the code, known vulnerable dependencies | `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run bandit -q -r src && uv run python scripts/check_text.py && uv run python scripts/gen_algorithms_doc.py --check && uv run pip-audit --skip-editable` |
| Unit | each operation against synthetic images with exact numeric expectations | `uv run pytest --cov` |
| Web | status codes, headers, validation, path traversal, privacy, failure and overload behaviour, with fake models | `uv run pytest --cov` |
| Models | real weights find the person in the sample image | `uv run pytest -m models` |
| Browser | the upload journey in Chromium, no console errors, no third-party requests, no horizontal scrolling on a phone | `uv run playwright install chromium && uv run pytest -m e2e` |
| Load | latency and error counts under sequential and concurrent uploads | `uv run python scripts/smoke_load.py http://127.0.0.1:8000 samples/astronaut.jpg` |

The fast suite (unit and web) is 159 tests at 99.12 % line and branch coverage; that figure comes
from `uv run pytest --cov`, since plain `uv run pytest` prints no coverage number. Method, measured
numbers and known limits are in [docs/testing.md](docs/testing.md). The design is described in
[docs/architecture.md](docs/architecture.md).

## Privacy and security

- Uploads are decoded, re-encoded without metadata (no GPS position, no device data) and stored
  under a random UUID (122 random bits). The name of the uploaded file is never used.
- There is no gallery and no listing. A result can be opened only by someone who has its address.
  Deletion is lazy and based on the age of the job directory (the end of processing), not a timer:
  results older than the configured time are removed at the next accepted upload or the next
  request for any result or image; nothing deletes them while the server is idle.
- The pages load no third-party resources and run no JavaScript, and they set a strict
  Content-Security-Policy.
- Model weights are downloaded from their publishers on first use and are not part of this repository.
- Facial expression analysis is off by default. Its output is a statistical guess about a visible
  expression and says nothing reliable about what a person feels. Do not use it to make decisions
  about people. Article 5(1)(f) of Regulation (EU) 2024/1689 (the AI Act) prohibits AI systems
  that infer the emotions of people in the workplace and in education institutions, except where
  used for medical or safety reasons.
- The server binds to the loopback interface by default. It has no authentication and is not meant
  to be exposed to the internet as it is.

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Project history

The project started in January 2025 as coursework in image processing at the West University of
Timisoara: one Flask file, written in a few days. That version is kept under the tag
[`v0.1.0-coursework`](https://github.com/tudorandrian/vision-lab-flask/tree/v0.1.0-coursework).
Version 1.0.0 (2026) keeps the scope and rebuilds the engineering: a tested package, validated
input, private storage, correct algorithms, reproducible dependencies and continuous integration.
[CHANGELOG.md](CHANGELOG.md) lists what changed and why.

## Licences and credits

- This project: GNU Affero General Public License, version 3 or later. Copyright (C) 2025-2026
  Tudor Andrian. This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you are
  welcome to redistribute it under the conditions of the licence. See [LICENSE](LICENSE).
- The footer of every page links to this source code. That link does not by itself satisfy
  section 13 of the licence for someone else's deployment: if you modify the application and let
  other people use your version over a network, section 13 requires you to offer them the source
  of your version, not this one. Set `VISION_LAB_SOURCE_URL` to point the footer link at your own
  source before you deploy it.
- Bootstrap 5.3.8 (MIT) is vendored under `src/vision_lab/static/vendor/bootstrap`.
- torchvision (BSD-3-Clause); pretrained weights are subject to the terms of the datasets they
  were trained on (COCO, Pascal VOC, ImageNet). Ultralytics YOLOv5u code and weights: AGPL-3.0,
  trained on COCO. DeepFace (MIT); the emotion model it downloads keeps its own terms and was
  trained on the FER-2013 dataset. Check these before any use beyond study.
- Sample image: astronaut Eileen Collins, NASA (public domain in the United States; see
  [samples/README.md](samples/README.md) for the conditions NASA asks be respected, since Eileen
  Collins is an identifiable person).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up the project, which checks a pull request
must pass, and how contributions are licensed.

## Citation

If you use this project in teaching or research, cite it using [CITATION.cff](CITATION.cff).
