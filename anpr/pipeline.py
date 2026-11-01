"""End-to-end ANPR: YOLOv8 detection -> crop -> preprocessing -> Tesseract -> validation."""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from . import config
from .detector import PlateDetector
from .ocr import recognize
from .preprocess import crop_plate, plate_variants


@dataclass
class PlateResult:
    text: str
    ocr_conf: float
    det_conf: float
    valid: bool
    box: Tuple[int, int, int, int]
    crop: np.ndarray


class ANPRPipeline:
    def __init__(self, model_path=config.MODEL_PATH, **detector_kwargs):
        self.detector = PlateDetector(model_path, **detector_kwargs)

    def process(self, image: np.ndarray) -> List[PlateResult]:
        results = []
        for det in self.detector.detect(image):
            crop = crop_plate(image, det.box)
            if crop.size == 0:
                continue
            ocr = recognize(plate_variants(crop))
            results.append(PlateResult(ocr.text, ocr.confidence, det.conf, ocr.valid, det.box, crop))
        return results

    @staticmethod
    def draw(image: np.ndarray, results: List[PlateResult]) -> np.ndarray:
        out = image.copy()
        for r in results:
            x1, y1, x2, y2 = r.box
            color = (0, 200, 0) if r.valid else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, r.text or "?", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return out
