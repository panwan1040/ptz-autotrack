from __future__ import annotations

from dataclasses import dataclass

from app.utils.geometry import BBox


@dataclass(slots=True)
class EmaSmoother:
    alpha: float
    _bbox: BBox | None = None

    def update(self, bbox: BBox | None) -> BBox | None:
        if bbox is None:
            return self._bbox
        if self._bbox is None:
            self._bbox = bbox
            return bbox
        self._bbox = tuple(
            self.alpha * current + (1 - self.alpha) * prev
            for current, prev in zip(bbox, self._bbox)
        )
        return self._bbox

    def reset(self) -> None:
        self._bbox = None
