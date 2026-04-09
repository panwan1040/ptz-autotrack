from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import AppearanceSection
from app.utils.geometry import BBox


@dataclass(slots=True)
class AppearanceExtractor:
    """Computes a lightweight color descriptor for continuity across short occlusions."""

    config: AppearanceSection

    def extract(self, frame: np.ndarray, bbox: BBox) -> list[float] | None:
        if not self.config.enabled:
            return None
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(frame.shape[1] - 1, x1))
        x2 = max(0, min(frame.shape[1], x2))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        y2 = max(0, min(frame.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bins = max(8, self.config.histogram_bins)
        hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype("float32").tolist()

    def blend(
        self,
        baseline: list[float] | None,
        new_signature: list[float] | None,
    ) -> list[float] | None:
        if new_signature is None:
            return baseline
        if baseline is None:
            return new_signature
        alpha = self.config.update_alpha
        base = np.asarray(baseline, dtype=np.float32)
        new = np.asarray(new_signature, dtype=np.float32)
        if base.shape != new.shape:
            return new_signature
        blended = ((1.0 - alpha) * base) + (alpha * new)
        total = float(blended.sum())
        if total > 0:
            blended /= total
        return blended.astype("float32").tolist()

    def similarity(
        self,
        left: list[float] | None,
        right: list[float] | None,
    ) -> float:
        if left is None or right is None:
            return 0.5
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        if a.shape != b.shape or a.size == 0:
            return 0.0
        numerator = float(np.minimum(a, b).sum())
        denominator = max(1e-6, float(np.maximum(a, b).sum()))
        return max(0.0, min(1.0, numerator / denominator))
