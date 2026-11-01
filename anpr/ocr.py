"""Tesseract OCR + regex validation."""
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pytesseract

from . import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

_TESS_CONFIG = f"--oem 3 --psm 7 -c tessedit_char_whitelist={config.OCR_WHITELIST}"


@dataclass
class OCRResult:
    text: str
    confidence: float   # mean Tesseract confidence, 0-100
    valid: bool         # matches one of config.PLATE_PATTERNS
    variant: str        # which preprocessing variant produced it


def clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid(text: str) -> bool:
    return any(re.match(p, text) for p in config.PLATE_PATTERNS)


def _read(img: np.ndarray) -> Tuple[str, float]:
    data = pytesseract.image_to_data(img, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
    parts, confs = [], []
    for txt, conf in zip(data["text"], data["conf"]):
        if txt.strip() and float(conf) >= 0:
            parts.append(txt.strip())
            confs.append(float(conf))
    return clean_text("".join(parts)), (sum(confs) / len(confs) if confs else 0.0)


def recognize(variants: Dict[str, np.ndarray]) -> OCRResult:
    """Run OCR on every variant and keep the best (valid first, then highest confidence)."""
    best = OCRResult("", 0.0, False, "none")
    for name, img in variants.items():
        text, conf = _read(img)
        if not text:
            continue
        cand = OCRResult(text, conf, is_valid(text), name)
        if (cand.valid, cand.confidence) > (best.valid, best.confidence):
            best = cand
    return best
