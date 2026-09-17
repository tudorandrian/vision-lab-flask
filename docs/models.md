# Model card

Facts about each pretrained model used by the application: the exact weights pinned in code,
what they were trained on, what they are reasonable to use for, what they are not, a known
failure mode on the sample image, and the evaluation that was not done. No accuracy, calibration
or fairness evaluation was run for this project; the numbers below are only what the publishers
of each model state.

## Faster R-CNN, MobileNetV3-Large 320 FPN

- Weights: `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1` (torchvision), pinned in
  `inference.py` rather than `.DEFAULT`.
- Training data: COCO 2017 (80 object classes; torchvision's own label list has 91 entries,
  including unused placeholders and background).
- Intended use: general-purpose object detection at low latency on a CPU, for demonstration.
- Not suitable for: small objects (the 320 px short side trades detail for speed), closed sets
  outside the 80 COCO classes, or any safety-critical decision.
- Known failure mode on the sample: none beyond the ordinary limits of the COCO class set.
- Bias and evaluation: not evaluated here; COCO's own known biases in class and scene distribution
  apply unchanged.

## YOLOv5nu and YOLOv5su

- Weights: `yolov5nu.pt` and `yolov5su.pt` (Ultralytics), downloaded from an Ultralytics GitHub
  release on first use. The download is checked only for a minimum file size, not a content hash;
  the checkpoint is loaded with `ULTRALYTICS_SAFE_LOAD=1` (weights-only, no arbitrary pickle
  execution) set before Ultralytics is imported.
- Training data: COCO 2017, the same 80 classes as Faster R-CNN above.
- Intended use: an alternative detector with different size and accuracy trade-offs, opt-in
  through the `yolo` extra because its licence is AGPL-3.0 rather than BSD-3-Clause.
- Not suitable for: the same limits as Faster R-CNN, plus anyone unwilling to accept AGPL-3.0
  terms for a deployment that uses it.
- Known failure mode on the sample: none beyond the ordinary limits of the COCO class set.
- Bias and evaluation: not evaluated here.

## DeepLabV3, MobileNetV3-Large backbone

- Weights: `DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1` (torchvision), pinned
  in `inference.py` rather than `.DEFAULT`.
- Training data: a COCO subset restricted to images containing the 20 Pascal VOC categories, plus
  background (21 classes total).
- Intended use: semantic segmentation over the VOC-2012 category set, for demonstration.
- Not suitable for: fine-grained segmentation outside the 20 VOC categories, or any safety-critical
  decision.
- Known failure mode on the sample: part of the helmet is labelled `motorbike` instead of
  `person`, a real misclassification shown on the result page rather than hidden.
- Runs at the image's own size, up to `VISION_LAB_MAX_SIDE`, instead of the published 520 px
  transform; see [docs/architecture.md](architecture.md#decisions) for the cost this trades and
  why. This choice does not obviously change accuracy: on the sample image the person's share of
  the frame is 44.7 % at native size against 47.9 % measured with the official 520 px transform.
- Bias and evaluation: not evaluated here.

## DeepFace emotion model

- Weights: downloaded by DeepFace itself into the configured weights directory; not pinned to a
  specific release by this project.
- Training data: FER-2013, a dataset of posed and web-scraped grey-scale faces with its own terms
  of use, separate from DeepFace's own MIT-licensed code.
- Intended use: none beyond study and demonstration. Off by default, and gated a second time by
  the `emotion` extra being installed; runs only when a person has already been detected.
- Not suitable for: any decision about a real person. FER-2013 is known in the literature for
  demographic and pose bias, and a facial expression is not a reliable signal of what someone
  feels; see the AI Act note in the README's "Privacy and security" section.
- Known failure mode on the sample: the sample image is not used to demonstrate this model (see
  [samples/README.md](../samples/README.md)).
- Bias and evaluation: not evaluated here.
