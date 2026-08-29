"""Image preprocessing helper for inference on arbitrary-resolution digit images."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def _otsu_threshold(gray: np.ndarray) -> float:
    """Calculate the optimal global threshold using Otsu's binarization method."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    current_max, threshold = 0.0, 0.0
    sum_total = np.dot(np.arange(256), hist)
    sum_back, weight_back = 0.0, 0.0

    for i in range(256):
        weight_back += hist[i]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += i * hist[i]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        var_between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = float(i)

    return threshold


def preprocess_image(
    image: Image.Image,
    auto_invert: bool = True,
    method: Literal["otsu", "standard"] = "otsu",
) -> tuple[np.ndarray, Image.Image]:
    """Preprocess an arbitrary-resolution image into an MNIST-compatible 28x28 input.

    Supported preprocessing methods:
    - ``"otsu"``: Recommended for real-world camera photos and ambient lighting with shadows.
      Computes global Otsu thresholding and removes background paper noise.
    - ``"standard"``: Standard thresholding. Recommended for clean digital drawings (e.g. MS Paint).

    Both modes include:
    1. Transparency / alpha channel compositing.
    2. Automatic luminance detection and inversion (ensuring bright strokes on dark canvas).
    3. Connected component filtering to remove tiny accidental pen specks/dots.
    4. Adaptive stroke dilation for high-resolution thin line drawings (preventing stroke washout).
    5. Aspect-ratio preserving resize into a 20x20 bounding box.
    6. Center of mass positioning (NIST/MNIST canonical alignment).

    Args:
        image: PIL Image instance of any size or color mode.
        auto_invert: When True, detects light backgrounds and inverts to dark backgrounds.
            Defaults to True.
        method: Preprocessing strategy to use, either ``"otsu"`` or ``"standard"``.
            Defaults to ``"otsu"``.

    Returns:
        A tuple containing:
            - ``features``: 2D NumPy float32 array of shape ``(1, 784)`` normalized to [0.0, 1.0].
            - ``canvas``: 28x28 PIL Image representing the exact model input.
    """
    # 1. Handle transparency in RGBA / LA / palette images
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.split()[-1] if image.mode == "RGBA" else None
        bg.paste(image, mask=alpha)
        image = bg

    # 2. Convert to grayscale array
    img = image.convert("L")
    arr = np.array(img, dtype=np.float32)

    # 3. Detect background luminance from border pixels
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg_val = float(np.median(border_pixels))

    if auto_invert and bg_val > 127.0:
        arr = 255.0 - arr

    # 4. Thresholding based on selected method
    if method == "otsu":
        thresh = _otsu_threshold(arr)
        arr = np.where(arr > thresh, (arr - thresh) / (arr.max() - thresh + 1e-5) * 255.0, 0.0)
    else:
        arr = np.where(arr > 30.0, arr, 0.0)

    # 5. Connected Component Analysis: Remove stray outlier specks/dots
    binary_mask = arr > 20.0
    labeled_array, num_features = ndimage.label(binary_mask)
    if num_features > 1:
        component_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
        max_size = float(component_sizes.max())
        valid_labels = np.where(component_sizes >= max_size * 0.15)[0] + 1
        filtered_mask = np.isin(labeled_array, valid_labels)
        arr = np.where(filtered_mask, arr, 0.0)

    mask = arr > 20.0
    if not np.any(mask):
        canvas = Image.new("L", (28, 28), color=0)
        return np.zeros((1, 784), dtype=np.float32), canvas

    y_indices, x_indices = np.where(mask)
    ymin, ymax = int(y_indices.min()), int(y_indices.max())
    xmin, xmax = int(x_indices.min()), int(x_indices.max())
    h, w = ymax - ymin + 1, xmax - xmin + 1

    cropped = Image.fromarray(arr[ymin : ymax + 1, xmin : xmax + 1].astype(np.uint8))

    # 6. Adaptive stroke thickening for high-resolution hairline drawings
    max_dim = max(h, w)
    if max_dim > 28:
        filter_size = max(3, int(max_dim / 28 * 1.5))
        if filter_size % 2 == 0:
            filter_size += 1
        cropped = cropped.filter(ImageFilter.MaxFilter(filter_size))

    # 7. Resize to fit within 20x20 bounding box
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 8. Initial placement on 28x28 black canvas
    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - cropped.width) // 2, (28 - cropped.height) // 2)
    canvas.paste(cropped, offset)

    # 9. Contrast normalization (ensuring crisp stroke peak)
    final_arr = np.array(canvas, dtype=np.float32)
    if final_arr.max() > 0:
        final_arr = (final_arr / final_arr.max()) * 255.0
        canvas = Image.fromarray(final_arr.astype(np.uint8))

    # 10. Center of mass alignment
    total_mass = final_arr.sum()
    if total_mass > 0.0:
        y_grid, x_grid = np.indices((28, 28))
        cy = (y_grid * final_arr).sum() / total_mass
        cx = (x_grid * final_arr).sum() / total_mass
        shift_y = int(np.round(14.0 - cy))
        shift_x = int(np.round(14.0 - cx))
        shift_x = max(-3, min(3, shift_x))
        shift_y = max(-3, min(3, shift_y))
        if shift_x != 0 or shift_y != 0:
            shifted = Image.new("L", (28, 28), color=0)
            shifted.paste(canvas, (shift_x, shift_y))
            canvas = shifted

    features = (np.array(canvas, dtype=np.float32) / 255.0).reshape(1, 784)
    return features, canvas
