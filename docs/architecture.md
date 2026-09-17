# Architecture

## Request flow

```
POST /jobs  ->  body over the limit        waitress, then Flask's MAX_CONTENT_LENGTH        -> 413
            ->  no image field             request.files.get("image") is empty              -> 400
            ->  params.parse_params        every field validated, all errors at once         -> 400
            ->  storage.decode_upload      format, pixel limit, re-encode, downscale         -> 400
            ->  inference slot             bounded wait, then give up                        -> 503 + Retry-After, the HTML error page
            ->  purge + pipeline.run_job   expired jobs removed, then ops and models run      -> 500, job discarded, on failure
            ->  303 See Other              /jobs/<id>

GET /jobs/<id>               purges expired jobs, then reads result.json and renders; models never run on a GET
GET /jobs/<id>/files/<name>  purges expired jobs, checks id and name against strict patterns, then serves one image
```

## Modules

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `config.py` | limits and switches from environment variables | standard library |
| `params.py` | form fields to a frozen `ProcessingParams`, or `ParamError` | standard library |
| `imaging.py` | the shared `Image` type alias | OpenCV typing |
| `ops.py` | classical operations as pure functions on arrays | OpenCV, NumPy |
| `inference.py` | `Detector`, `Segmenter`, `EmotionAnalyzer` protocols, their implementations, `ModelRegistry` | torch, torchvision; ultralytics and deepface when installed |
| `render.py` | boxes, overlays, segmentation metrics | OpenCV, NumPy, `inference.py` for the `Detection` and `FaceEmotion` types |
| `storage.py` | upload decoding, per-job directories, expiry | Pillow, OpenCV |
| `pipeline.py` | runs every stage for one upload, records the result | the modules above |
| `catalog.py` | the one description of every operation | standard library |
| `app.py` | application factory, routes, headers, error pages | Flask |
| `__main__.py` | `vision-lab` command, serves with waitress | waitress |

`ops.py` knows nothing about files or HTTP, `app.py` knows nothing about OpenCV, and only
`inference.py` imports a deep-learning library, inside the class that needs it. That is what lets
the web layer be tested in milliseconds with fake models.

## Decisions

**uv instead of Anaconda.** The original environment was a conda export tied to one Windows
machine. `pyproject.toml` plus `uv.lock` gives the same versions on Windows, Linux and macOS from
one command, and uv installs the Python interpreter itself. CPU-only PyTorch wheels are selected
through an explicit index, which avoids several GB of CUDA libraries this project never uses.

**Faster R-CNN, MobileNetV3-Large 320 FPN by default, YOLO as an extra.** The coursework fetched
YOLOv5 code from GitHub at run time through `torch.hub`, which executes unpinned remote code and
needs a dozen undeclared dependencies. The default detector now ships with torchvision under
BSD-3-Clause; it resizes its input so the short side is 320 px, trading small-object accuracy for
CPU speed. YOLOv5u remains available through the pinned `ultralytics` package for anyone who
installs the `yolo` extra; each YOLO detector serialises its calls with its own lock, because the
underlying YOLO object keeps per-call predictor state and is not safe to call from two threads at
once.

**AGPL-3.0-or-later.** The optional YOLO backend is AGPL-3.0, and this code is written to work
with it, so the whole project takes the same licence instead of arguing about boundaries. The
author holds the copyright of the code and documents written for this project and can relicense
that code for another use; the Ultralytics dependency would then need to be left out or licensed
commercially. The source link in the footer of every page is the way this application offers its
source, as section 13 asks of modified versions: it points at `VISION_LAB_SOURCE_URL` (default
this repository), which whoever modifies and deploys their own version must change to their own
source, since the upstream link cannot do that for them.

**Results are files, not recomputation.** The coursework ran every model again on each page view
and let the query string choose the file to process. A job now runs once, writes `result.json`,
and a GET only reads it. Expired jobs are purged once the inference slot for a new upload has been
acquired (so an upload refused with 503 does not purge) and again on every request for a result or
one of its files. A lock keeps a purge's listing and removals atomic against a concurrent purge in
the same process.

**No JavaScript.** The interface is a form and a result page. Leaving scripts out allows
`script-src 'none'`, removes a class of bugs, and keeps the pages usable everywhere.

**One inference at a time.** A CPU model saturates the cores it is given; running two at once
makes both slower and doubles peak memory. Uploads wait up to `VISION_LAB_QUEUE_SECONDS` for the
slot and then receive 503 with `Retry-After` on the same HTML error page used for other failures,
so load degrades predictably instead of crashing.

**waitress bounds the request body.** waitress defaults its own request body limit to 1 GB and
buffers the whole body before Flask's `MAX_CONTENT_LENGTH` can answer 413. `__main__.py` sets
`max_request_body_size` to the upload limit plus a 1 MiB margin for multipart framing and the
other form fields, so an oversized upload is refused before it is buffered, not after. A request
body larger than that margin is refused by waitress itself, before Flask's own request handling
runs, so the client sees waitress's own plain-text error page instead of this application's styled
one. This is a deliberate trade-off: waitress must be able to reject a wildly oversized body
cheaply, without buffering it first, and the only way to do that is to answer before the WSGI
application, and its error handlers, ever run.

**DeepLabV3 runs at the image's own size, not the published 520 px transform.** The published
transform for `COCO_WITH_VOC_LABELS_V1` resizes the input's short side to 520 px before inference.
`DeepLabSegmenter.segment` instead feeds the image at whatever size it already is, up to
`VISION_LAB_MAX_SIDE` (1600 by default), so the class map comes out at the resolution the rest of
the pipeline is already working with, instead of a coarse map that would need to be upsampled onto
the original image. Measured on the reference machine in docs/testing.md: 0.46 s and 504 MB peak
working set at 512x512, against 3.91 s and 952 MB peak at 1600x1600, the configured maximum. On the
sample image this does not obviously cost accuracy (the person's share of the frame is 44.7 % at
native size against 47.9 % with the official 520 px transform), but the latency and memory cost at
the configured maximum are real and are not evaluated beyond this one image; see
[docs/models.md](models.md) for the model card.

**DeepFace weights under the data directory.** DeepFace reads the `DEEPFACE_HOME` environment
variable at import time and stores its weights under `<home>/.deepface/weights`. `inference.py`
sets it to the configured weights directory before importing DeepFace, unless `DEEPFACE_HOME` is
already set, so every downloaded weight lands under `VISION_LAB_DATA_DIR` like the other models,
instead of the user's home directory.

**OpenCV below 5.** OpenCV 5 removed `cv2.CascadeClassifier`, which DeepFace still uses for its
default face detector. The constraint is recorded in `pyproject.toml` and in the Dependabot
configuration, and can be lifted once the emotion extra is verified on OpenCV 5.
