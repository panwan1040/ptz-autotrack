from __future__ import annotations

from dataclasses import dataclass

from app.config import TrackingSection
from app.models.runtime import Detection, TargetState, TrackStatus
from app.tracking.target_selector import TargetSelector


@dataclass(slots=True)
class Tracker:
    config: TrackingSection

    def __post_init__(self) -> None:
        self._selector = TargetSelector(self.config)
        self._next_track_id = 1
        self._state = TargetState(track_id=None, bbox_xyxy=None)

    @property
    def state(self) -> TargetState:
        return self._state

    def update(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> TargetState:
        assigned = self._assign_track_ids(detections)
        selected = self._selector.select(assigned, self._state, frame_width, frame_height, now)
        if selected.status == TrackStatus.TRACKING and selected.track_id is None:
            selected.track_id = self._next_id()
        self._state = selected
        return self._state

    def _assign_track_ids(self, detections: list[Detection]) -> list[Detection]:
        assigned: list[Detection] = []
        for detection in detections:
            if detection.tracker_id is None:
                detection.tracker_id = self._next_id()
            assigned.append(detection)
        return assigned

    def _next_id(self) -> int:
        current = self._next_track_id
        self._next_track_id += 1
        return current
