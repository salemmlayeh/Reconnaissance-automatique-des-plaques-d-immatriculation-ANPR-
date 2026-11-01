"""YOLOv8 license plate detector (PyTorch / Ultralytics)."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    box: Tuple[int, int, int, int]   # x1, y1, x2, y2
    conf: float


class PlateDetector:
    def __init__(self, model_path=config.MODEL_PATH, conf=config.DET_CONF,
                 imgsz=config.IMG_SIZE, device=config.DEVICE):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Return plate detections sorted by confidence (highest first)."""
        result = self.model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            detections.append(Detection((x1, y1, x2, y2), float(b.conf[0])))
        return sorted(detections, key=lambda d: d.conf, reverse=True)
