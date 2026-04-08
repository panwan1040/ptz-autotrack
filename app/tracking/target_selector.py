from __future__ import annotations

from dataclasses import dataclass

from app.config import TrackingSection
from app.models.runtime import Detection, TargetState, TrackStatus
from app.utils.geometry import bbox_area, center_distance_normalized, normalized_bbox_center


@dataclass(slots=True)
class TargetSelector:
    config: TrackingSection

    def select(
        self,
        detections: list[Detection],
        previous: TargetState,
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> TargetState:
        if not detections:
            return self._lost(previous, now)

        if previous.bbox_xyxy is not None and previous.track_id is not None:
            sticky = self._stick_to_previous(detections, previous, frame_width, frame_height)
            if sticky is not None:
                return TargetState(
                    track_id=sticky.tracker_id,
                    bbox_xyxy=sticky.bbox_xyxy,
                    confidence=sticky.confidence,
                    persist_frames=previous.persist_frames + 1,
                    last_seen_ts=now,
                    status=TrackStatus.TRACKING,
                )

        chosen = self._choose_by_strategy(detections, frame_width, frame_height)
        return TargetState(
            track_id=chosen.tracker_id,
            bbox_xyxy=chosen.bbox_xyxy,
            confidence=chosen.confidence,
            persist_frames=1,
            last_seen_ts=now,
            status=TrackStatus.TRACKING,
        )

    def _lost(self, previous: TargetState, now: float) -> TargetState:
        if previous.track_id is None:
            return TargetState(track_id=None, bbox_xyxy=None, last_seen_ts=now, status=TrackStatus.SEARCHING)
        status = (
            TrackStatus.LOST
            if (now - previous.last_seen_ts) <= self.config.lost_timeout_seconds
            else TrackStatus.SEARCHING
        )
        return TargetState(
            track_id=previous.track_id,
            bbox_xyxy=previous.bbox_xyxy,
            confidence=0.0,
            persist_frames=previous.persist_frames,
            last_seen_ts=previous.last_seen_ts,
            status=status,
        )

    def _stick_to_previous(
        self,
        detections: list[Detection],
        previous: TargetState,
        frame_width: int,
        frame_height: int,
    ) -> Detection | None:
        assert previous.bbox_xyxy is not None
        best: tuple[float, Detection] | None = None
        for detection in detections:
            distance = center_distance_normalized(
                detection.bbox_xyxy,
                previous.bbox_xyxy,
                frame_width,
                frame_height,
            )
            if distance > self.config.max_association_distance:
                continue
            score = distance - (detection.confidence * 0.05)
            if best is None or score < best[0]:
                best = (score, detection)
        return best[1] if best else None

    def _choose_by_strategy(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
    ) -> Detection:
        strategy = self.config.strategy
        if strategy == "largest":
            return max(detections, key=lambda det: bbox_area(det.bbox_xyxy))
        if strategy == "highest_confidence":
            return max(detections, key=lambda det: det.confidence)
        if strategy == "most_centered":
            return min(
                detections,
                key=lambda det: self._center_penalty(det, frame_width, frame_height),
            )
        return min(
            detections,
            key=lambda det: self._center_penalty(det, frame_width, frame_height) - det.confidence * 0.02,
        )

    def _center_penalty(self, detection: Detection, frame_width: int, frame_height: int) -> float:
        nx, ny = normalized_bbox_center(detection.bbox_xyxy, frame_width, frame_height)
        return abs(nx - 0.5) + abs(ny - 0.5)
