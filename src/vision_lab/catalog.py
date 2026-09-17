"""The single description of every operation.

The result cards, the in-app Algorithms page and docs/algorithms.md are all
rendered from this table, so the three can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    key: str
    title: str
    summary: str
    reference: str


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    entries: tuple[Entry, ...]

    def get(self, key: str) -> Entry:
        return next(entry for entry in self.entries if entry.key == key)


_OPENCV = "OpenCV documentation, imgproc module."

CATALOG: tuple[Group, ...] = (
    Group(
        "color_space",
        "Colour spaces",
        (
            Entry(
                "HSV",
                "HSV",
                "Hue, saturation and value planes. Separating hue from brightness makes colour thresholds robust to lighting.",
                "Smith, A. R. (1978). Color gamut transform pairs. SIGGRAPH.",
            ),
            Entry(
                "LAB",
                "CIE L*a*b*",
                "Lightness plus two opponent colour axes, designed so that equal distances are roughly equal perceived differences.",
                "CIE (2004). Colorimetry, 3rd edition. Publication 15:2004.",
            ),
            Entry(
                "GRAY",
                "Greyscale",
                "A single luminance plane, the usual input for edge detection and thresholding.",
                _OPENCV,
            ),
            Entry("RGB", "RGB", "The red, green and blue planes as stored by the camera.", _OPENCV),
        ),
    ),
    Group(
        "filter_type",
        "Smoothing filters",
        (
            Entry(
                "gaussian",
                "Gaussian blur",
                "Convolution with a Gaussian kernel. Suppresses high-frequency noise and blurs edges in proportion to the kernel size.",
                _OPENCV,
            ),
            Entry(
                "median",
                "Median blur",
                "Replaces each pixel with the median of its neighbourhood. Removes salt-and-pepper noise while keeping edges sharp.",
                "Huang, Yang and Tang (1979). A fast two-dimensional median filtering algorithm. IEEE TASSP.",
            ),
        ),
    ),
    Group(
        "edge_algorithm",
        "Edge detection",
        (
            Entry(
                "canny",
                "Canny",
                "Gaussian smoothing, gradient, non-maximum suppression and hysteresis between two thresholds. Produces thin, connected edges.",
                "Canny, J. (1986). A computational approach to edge detection. IEEE TPAMI.",
            ),
            Entry(
                "sobel",
                "Sobel",
                "Gradient magnitude from 3x3 horizontal and vertical derivative kernels.",
                "Sobel and Feldman (1968). A 3x3 isotropic gradient operator for image processing.",
            ),
            Entry(
                "scharr",
                "Scharr",
                "Same idea as Sobel with kernels optimised for rotational symmetry, more accurate on diagonal edges.",
                "Scharr, H. (2000). Optimal operators in digital image processing. PhD thesis, Heidelberg.",
            ),
            Entry(
                "roberts",
                "Roberts cross",
                "Gradient magnitude from two 2x2 diagonal difference kernels. Fast and very sensitive to noise.",
                "Roberts, L. G. (1963). Machine perception of three-dimensional solids. MIT.",
            ),
            Entry(
                "log",
                "Laplacian of Gaussian",
                "Gaussian smoothing followed by the Laplacian; edges appear where the second derivative responds strongly.",
                "Marr and Hildreth (1980). Theory of edge detection. Proc. Royal Society B.",
            ),
        ),
    ),
    Group(
        "equalization_type",
        "Histogram equalisation",
        (
            Entry(
                "he",
                "Global equalisation",
                "Remaps lightness so its histogram is flat over the whole image.",
                _OPENCV,
            ),
            Entry(
                "ahe",
                "Adaptive equalisation (AHE)",
                "Equalises each tile separately, then interpolates. Reveals local detail and amplifies noise in flat regions.",
                "Pizer et al. (1987). Adaptive histogram equalization and its variations. CVGIP.",
            ),
            Entry(
                "clahe",
                "Contrast-limited AHE (CLAHE)",
                "AHE with each tile histogram clipped at a limit, which bounds noise amplification.",
                "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics Gems IV.",
            ),
        ),
    ),
    Group(
        "enhancement_type",
        "Enhancement",
        (
            Entry(
                "sharpen",
                "Unsharp mask",
                "Adds back the difference between the image and a blurred copy; the value is the amount.",
                _OPENCV,
            ),
            Entry(
                "denoise",
                "Non-local means",
                "Averages similar patches from across the image; the value scales the filter strength.",
                "Buades, Coll and Morel (2005). A non-local algorithm for image denoising. CVPR.",
            ),
            Entry(
                "brightness",
                "Brightness",
                "Multiplies every pixel by the value. 1.0 leaves the image unchanged.",
                _OPENCV,
            ),
            Entry(
                "contrast",
                "Contrast",
                "Scales the distance of every pixel from the mean intensity by the value.",
                _OPENCV,
            ),
        ),
    ),
    Group(
        "models",
        "Pretrained models",
        (
            Entry(
                "fasterrcnn",
                "Faster R-CNN, MobileNetV3-Large 320 FPN",
                "Two-stage object detector trained on COCO (80 object classes; torchvision's label "
                "list has 91 entries, including unused placeholders and background). Images are resized so "
                "the short side is 320 px internally, trading small-object accuracy for CPU speed. "
                "torchvision weights, BSD-3-Clause.",
                "Ren et al. (2015). Faster R-CNN. NeurIPS. Howard et al. (2019). Searching for MobileNetV3. ICCV.",
            ),
            Entry(
                "yolov5nu",
                "YOLOv5nu",
                "Single-stage, anchor-free detector, nano size. Optional extra, Ultralytics weights under AGPL-3.0.",
                "Jocher, G. et al. Ultralytics YOLOv5.",
            ),
            Entry(
                "yolov5su",
                "YOLOv5su",
                "Single-stage, anchor-free detector, small size. Optional extra, Ultralytics weights under AGPL-3.0.",
                "Jocher, G. et al. Ultralytics YOLOv5.",
            ),
            Entry(
                "deeplabv3",
                "DeepLabV3, MobileNetV3-Large",
                "Semantic segmentation with atrous spatial pyramid pooling over the 21 Pascal VOC classes.",
                "Chen et al. (2017). Rethinking atrous convolution for semantic image segmentation.",
            ),
            Entry(
                "deepface",
                "DeepFace emotion model",
                "Classifies each detected face into seven expression labels. Optional extra, disabled by default; see the privacy notes.",
                "Serengil and Ozpinar (2021). HyperExtended LightFace. ICEET.",
            ),
        ),
    ),
)


def group(key: str) -> Group:
    return next(item for item in CATALOG if item.key == key)
