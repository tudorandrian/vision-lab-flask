# Algorithms and models

Generated from `src/vision_lab/catalog.py` by `scripts/gen_algorithms_doc.py`.
Edit the catalogue, not this file. The same text appears in the app under Algorithms.

## Colour spaces

| Name | What it does | Reference |
| --- | --- | --- |
| HSV | Hue, saturation and value planes. Separating hue from brightness makes colour thresholds robust to lighting. | Smith, A. R. (1978). Color gamut transform pairs. SIGGRAPH. |
| CIE L*a*b* | Lightness plus two opponent colour axes, designed so that equal distances are roughly equal perceived differences. | CIE (2004). Colorimetry, 3rd edition. Publication 15:2004. |
| Greyscale | A single luminance plane, the usual input for edge detection and thresholding. | OpenCV documentation, imgproc module. |
| RGB | The red, green and blue planes as stored by the camera. | OpenCV documentation, imgproc module. |

## Smoothing filters

| Name | What it does | Reference |
| --- | --- | --- |
| Gaussian blur | Convolution with a Gaussian kernel. Suppresses high-frequency noise and blurs edges in proportion to the kernel size. | OpenCV documentation, imgproc module. |
| Median blur | Replaces each pixel with the median of its neighbourhood. Removes salt-and-pepper noise while keeping edges sharp. | Huang, Yang and Tang (1979). A fast two-dimensional median filtering algorithm. IEEE TASSP. |

## Edge detection

| Name | What it does | Reference |
| --- | --- | --- |
| Canny | Gaussian smoothing, gradient, non-maximum suppression and hysteresis between two thresholds. Produces thin, connected edges. | Canny, J. (1986). A computational approach to edge detection. IEEE TPAMI. |
| Sobel | Gradient magnitude from 3x3 horizontal and vertical derivative kernels. | Sobel and Feldman (1968). A 3x3 isotropic gradient operator for image processing. |
| Scharr | Same idea as Sobel with kernels optimised for rotational symmetry, more accurate on diagonal edges. | Scharr, H. (2000). Optimal operators in digital image processing. PhD thesis, Heidelberg. |
| Roberts cross | Gradient magnitude from two 2x2 diagonal difference kernels. Fast and very sensitive to noise. | Roberts, L. G. (1963). Machine perception of three-dimensional solids. MIT. |
| Laplacian of Gaussian | Gaussian smoothing followed by the Laplacian; edges appear where the second derivative responds strongly. | Marr and Hildreth (1980). Theory of edge detection. Proc. Royal Society B. |

## Histogram equalisation

| Name | What it does | Reference |
| --- | --- | --- |
| Global equalisation | Remaps lightness so its histogram is flat over the whole image. | OpenCV documentation, imgproc module. |
| Adaptive equalisation (AHE) | Equalises each tile separately, then interpolates. Reveals local detail and amplifies noise in flat regions. | Pizer et al. (1987). Adaptive histogram equalization and its variations. CVGIP. |
| Contrast-limited AHE (CLAHE) | AHE with each tile histogram clipped at a limit, which bounds noise amplification. | Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics Gems IV. |

## Enhancement

| Name | What it does | Reference |
| --- | --- | --- |
| Unsharp mask | Adds back the difference between the image and a blurred copy; the value is the amount. | OpenCV documentation, imgproc module. |
| Non-local means | Averages similar patches from across the image; the value scales the filter strength. | Buades, Coll and Morel (2005). A non-local algorithm for image denoising. CVPR. |
| Brightness | Multiplies every pixel by the value. 1.0 leaves the image unchanged. | OpenCV documentation, imgproc module. |
| Contrast | Scales the distance of every pixel from the mean intensity by the value. | OpenCV documentation, imgproc module. |

## Pretrained models

| Name | What it does | Reference |
| --- | --- | --- |
| Faster R-CNN, MobileNetV3-Large 320 FPN | Two-stage object detector trained on COCO (80 object classes; torchvision's label list has 91 entries, including unused placeholders and background). Images are resized so the short side is 320 px internally, trading small-object accuracy for CPU speed. torchvision weights, BSD-3-Clause. | Ren et al. (2015). Faster R-CNN. NeurIPS. Howard et al. (2019). Searching for MobileNetV3. ICCV. |
| YOLOv5nu | Single-stage, anchor-free detector, nano size. Optional extra, Ultralytics weights under AGPL-3.0. | Jocher, G. et al. Ultralytics YOLOv5. |
| YOLOv5su | Single-stage, anchor-free detector, small size. Optional extra, Ultralytics weights under AGPL-3.0. | Jocher, G. et al. Ultralytics YOLOv5. |
| DeepLabV3, MobileNetV3-Large | Semantic segmentation with atrous spatial pyramid pooling over the 21 Pascal VOC classes. | Chen et al. (2017). Rethinking atrous convolution for semantic image segmentation. |
| DeepFace emotion model | Classifies each detected face into seven expression labels. Optional extra, disabled by default; see the privacy notes. | Serengil and Ozpinar (2021). HyperExtended LightFace. ICEET. |
