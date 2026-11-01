"""OpenCV helpers: cropping and OCR-oriented preprocessing."""
from typing import Dict, Tuple

import cv2
import numpy as np

from . import config


def crop_plate(image: np.ndarray, box: Tuple[int, int, int, int], pad: float = 0.04) -> np.ndarray:
    """Crop the detected box with a small relative padding, clamped to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return image[y1:y2, x1:x2].copy()


def plate_variants(crop: np.ndarray,
                   target_height: int = config.OCR_TARGET_HEIGHT,
                   left_trim: float = config.LEFT_TRIM) -> Dict[str, np.ndarray]:
    """Build several binarised versions of the plate; OCR picks the best one."""
    if left_trim > 0:
        crop = crop[:, int(crop.shape[1] * left_trim):]

    scale = target_height / crop.shape[0]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)                       # denoise, keep edges
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)

    variants = {"gray": gray, "otsu": otsu, "adaptive": adaptive}
    out = {}
    for name, img in variants.items():
        if name != "gray" and img.mean() < 127:   # Tesseract prefers dark text on light background
            img = cv2.bitwise_not(img)
        out[name] = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return out
