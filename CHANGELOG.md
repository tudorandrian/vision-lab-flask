# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Dependabot no longer proposes Docker base images with Python 3.14 or later, which the project
  does not support yet (`requires-python` is `>=3.12,<3.14`).
- A `Dependabot requirements` workflow re-exports `requirements.txt` from `uv.lock` on Dependabot's
  uv pull requests and pushes the result, so they no longer need a manual commit; a maintainer
  approves the workflow runs on that commit. The commit is marked `[dependabot skip]`, so
  Dependabot keeps rebasing the pull request.
- Dependabot ignores `mpmath` 1.4 and later while `sympy` (required by `torch`) requires
  `mpmath<1.4`; such updates could only change `requirements.txt`, not `uv.lock`.
- `docs/testing.md` records the first `Emotion smoke` run: it passed on 2026-09-23 against 1.1.0.

## [1.1.0] - 2026-09-23

Hardening after an internal source review of 1.0.0: the same scope, with one new setting.

### Added

- `VISION_LAB_QUEUE_DEPTH` (default 4) bounds how many uploads may wait for the inference slot
  at once; an upload arriving beyond that gets HTTP 503 immediately instead of occupying a server
  thread for the wait. A free slot is taken at once and never counts against the depth, so with
  a depth of 0 an upload never waits.
- An `Emotion smoke` workflow (monthly and on demand) runs the real DeepFace no-face path
  outside the required checks.

### Changed

- A job's retention time is measured from the moment its result is written, and a job still
  being processed is never removed by a concurrent read; a directory left behind by a crashed
  process is removed a day after the TTL.
- `POST /jobs` refuses cross-site requests (`Sec-Fetch-Site: cross-site`, or an `Origin` naming
  another host, compared case-insensitively) with HTTP 403. Command-line clients that send
  neither header are unaffected.
- The plain-pip route installs from a committed `requirements.txt` exported from `uv.lock`, with
  the CPU-only PyTorch wheels pinned by name; CI fails if the two files drift apart.

### Fixed

- Scaling clamps the factor before resizing, so a 1600 px image rotated by 45 degrees and scaled
  by 4.0 no longer allocates a 246 MB intermediate that the size cap would discard anyway. The
  longer side is unchanged from 1.0.0; on a non-square image the shorter side can now differ by
  one pixel, because it is rounded once instead of twice.
- Result images are cached by the browser for at most the job's TTL, not a fixed five minutes.
- `VISION_LAB_QUEUE_SECONDS=inf` (or any value above 3600) is refused at start-up instead of
  failing with `OverflowError` on the first busy upload.

## [1.0.0] - 2026-09-17

Same scope as the coursework version, rebuilt as a maintainable application.

### Added

- Installable package `vision_lab` with an application factory and a `vision-lab` command.
- Reproducible environment with uv (`pyproject.toml`, `uv.lock`) and a Docker image; Anaconda is no longer needed.
- Input validation for every form field, with all errors reported at once.
- Settings are validated at start-up: an invalid environment variable is reported by name with
  its accepted range, and the process exits with a clear message instead of starting a server
  that is silently broken.
- Private per-job storage under random identifiers, lazy deletion, metadata removal.
- Strict security headers, generic error pages, a bounded inference queue.
- Test suite at several levels (static analysis, unit, web, models, browser, container, load) and continuous integration on Linux and Windows.
- CI audits the `yolo` and `emotion` extras for known vulnerabilities, in addition to the base dependency set.
- English interface, accessible markup, an Algorithms page and `docs/algorithms.md` generated from one catalogue.
- LICENSE (AGPL-3.0-or-later), SECURITY.md, CITATION.cff, CONTRIBUTING.md, this changelog.
- Served with waitress, with request bodies bounded to the upload limit plus 1 MiB.
- Decompression bombs and camera MPO JPEGs are handled correctly on upload.
- Results are deleted lazily: a job older than a configurable time, measured from when its
  processing finished, is removed at the next accepted upload or the next request for any result
  or image, not by a timer running in the background.

### Changed

- Object detection defaults to torchvision Faster R-CNN, MobileNetV3-Large 320 FPN; YOLOv5u is an optional extra installed from PyPI instead of code fetched from GitHub at run time.
- Segmentation runs at full resolution with all 21 Pascal VOC classes and measures class shares on the class map.
- Models load once per process instead of once per request; viewing a result no longer re-runs them.
- Bootstrap is vendored; the pages contact no third party.

### Fixed

- Sobel, Scharr and Roberts computed one gradient direction only and saved clipped float data.
- "LoG" was a Laplacian without the Gaussian; "AHE" was global equalisation; "denoise" was a sharpening placeholder.
- Segmentation coloured three of 21 classes and computed its metrics on the JPEG overlay instead of the class map.
- Colours were swapped when drawing on images (RGB data passed to BGR functions).
- Any validation error crashed with HTTP 500 (`url_for('index')` on a route named `home`, `flash` without a secret key).
- Even kernel sizes, out-of-range crops and large scale factors crashed OpenCV or exhausted memory.
- The unit tests called functions with signatures that no longer existed and could not pass.
- Rotation used an integer centre half a pixel off true centre, on a fixed-size canvas that cut off the corners of a rotated image; rotation now uses the exact centre on a canvas that grows to fit, checked against `np.rot90` for the 90 degree steps.
- The container ignored SIGTERM (Python as PID 1 has no default handler for it), so a normal
  `docker stop` ran out the grace period and killed the server; it now shuts down gracefully.
- A result page could be served stale from a shared cache past its expiry; it is now sent with
  `Cache-Control: no-store` (its images may still keep a short private cache).

### Security

- Ultralytics' automatic installation of missing packages from PyPI, and its unsafe pickle
  loading of the YOLO checkpoint, are forced off by an unconditional module-scope assignment
  (not a default), before the library is ever imported, so the environment cannot re-enable
  either check: previously a hostile upload with the `yolo` extra installed could make the server
  install an unpinned package, or load an unverified checkpoint, at request time.
- torchvision detector and segmenter weights are pinned to explicit enum members (`COCO_V1`,
  `COCO_WITH_VOC_LABELS_V1`) instead of `.DEFAULT`, so a torchvision upgrade cannot silently
  change detection or segmentation results.
- An upload's header (format, dimensions) is now checked before the inference slot is acquired, so
  an invalid upload is rejected immediately instead of waiting behind a busy server; only the full
  pixel decode, which can use hundreds of MB for a large photo, happens once the slot is held.
- A job id inside a logged request path (a capability URL) is redacted to its first eight
  characters plus an ellipsis; the rest of the path is kept so the route that failed is still
  identifiable.

### Removed

- `yolov5s.pt` (14.8 MB of AGPL-3.0 weights redistributed without a licence); weights are downloaded on first use.
- `environment.yml` and the conda `pip freeze` in `requirements.txt` (Windows build paths, a local user name, not installable elsewhere).
- The public gallery of every uploaded image.
- Flask debug mode as the way to run the application.
- The AI-generated hero image and an unused screenshot; the in-app documentation page that repeated what a README should say.
- Haar-cascade face circles drawn by a separate face detector alongside the object detector; face boxes now come only from the optional emotion extra.

## [0.1.0-coursework] - 2025-01-22

Coursework submission: a single-file Flask application with a Romanian interface. Kept as the
tag `v0.1.0-coursework`.

[Unreleased]: https://github.com/tudorandrian/vision-lab-flask/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/tudorandrian/vision-lab-flask/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/tudorandrian/vision-lab-flask/compare/v0.1.0-coursework...v1.0.0
[0.1.0-coursework]: https://github.com/tudorandrian/vision-lab-flask/tree/v0.1.0-coursework
