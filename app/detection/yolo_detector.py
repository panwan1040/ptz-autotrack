from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from ultralytics import YOLO

from app.config import DetectionSection
from app.models.runtime import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]: ...


@dataclass(slots=True)
class YoloDetector:
    config: DetectionSection
    _model: YOLO = field(init=False)

    def __post_init__(self) -> None:
        self._model = YOLO(self.config.model_path)

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        start = time.perf_counter()
        results = self._model.predict(
            source=frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            half=self.config.half,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        detections: list[Detection] = []
        if not results:
            return detections, latency_ms
        result = results[0]
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(names[class_id])
            if class_name not in ["person", "pedestrian", "people"]:
                continue
            class_name = "person"
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            detections.append(
                Detection(bbox_xyxy=(x1, y1, x2, y2), confidence=conf, class_name=class_name)
            )
        return detections, latency_ms
