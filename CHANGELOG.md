# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-09-17

Same scope as the coursework version, rebuilt as a maintainable application.

### Added

- Installable package `vision_lab` with an application factory and a `vision-lab` command.
- Reproducible environment with uv (`pyproject.toml`, `uv.lock`) and a Docker image; Anaconda is no longer needed.
- Input validation for every form field, with all errors reported at once.
- Private per-job storage under random identifiers, automatic deletion, metadata removal.
- Strict security headers, generic error pages, a bounded inference queue.
- Test suite at five levels (unit, web, models, browser, load) and continuous integration on Linux and Windows.
- English interface, accessible markup, an Algorithms page and `docs/algorithms.md` generated from one catalogue.
- LICENSE (AGPL-3.0-or-later), SECURITY.md, CITATION.cff, this changelog.

### Changed

- Object detection defaults to torchvision Faster R-CNN, MobileNetV3-Large 320 FPN; YOLOv5u is an optional extra installed from PyPI instead of code fetched from GitHub at run time.
- Segmentation runs at full resolution with all 21 Pascal VOC classes and measures class shares on the class map.
- Models load once per process instead of once per request; viewing a result no longer re-runs them.
- Bootstrap is vendored; the pages contact no third party.
- The busy response is now the same HTML error page as other failures, with a `Retry-After` header, instead of a bare status code.
- waitress bounds the request body to the configured upload limit plus a margin, so an oversized upload is refused before it is buffered.
- Expired results are now removed both when a new upload arrives and when an existing job is read, and a lock keeps concurrent purges from corrupting or double-deleting a result.

### Fixed

- Sobel, Scharr and Roberts computed one gradient direction only and saved clipped float data.
- "LoG" was a Laplacian without the Gaussian; "AHE" was global equalisation; "denoise" was a sharpening placeholder.
- Segmentation coloured three of 21 classes and computed its metrics on the JPEG overlay instead of the class map.
- Colours were swapped when drawing on images (RGB data passed to BGR functions).
- Any validation error crashed with HTTP 500 (`url_for('index')` on a route named `home`, `flash` without a secret key).
- Even kernel sizes, out-of-range crops and large scale factors crashed OpenCV or exhausted memory.
- The unit tests called functions with signatures that no longer existed and could not pass.
- Rotation used an off-by-one image centre, which is now exact against `np.rot90` for the 90 degree steps.
- Decompression bombs were not rejected cleanly on both sides of Pillow's own size threshold, and MPO camera JPEGs (multi-picture JPEG, as produced by some phone cameras) were rejected instead of accepted.

### Removed

- `yolov5s.pt` (14.8 MB of AGPL-3.0 weights redistributed without a licence); weights are downloaded on first use.
- `environment.yml` and the conda `pip freeze` in `requirements.txt` (Windows build paths, a local user name, not installable elsewhere).
- The public gallery of every uploaded image.
- Flask debug mode as the way to run the application.
- The AI-generated hero image and an unused screenshot; the in-app documentation page that duplicated the README.
- Haar-cascade face circles (the API was removed in OpenCV 5); face boxes now come from the optional emotion extra.

## [0.1.0-coursework] - 2025-01-22

Coursework submission: a single-file Flask application with a Romanian interface. Kept as the
tag `v0.1.0-coursework`.

[Unreleased]: https://github.com/tudorandrian/vision-lab-flask/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tudorandrian/vision-lab-flask/compare/v0.1.0-coursework...v1.0.0
[0.1.0-coursework]: https://github.com/tudorandrian/vision-lab-flask/tree/v0.1.0-coursework
