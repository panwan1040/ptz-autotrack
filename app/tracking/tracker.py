from __future__ import annotations

from dataclasses import dataclass, field

from app.config import TrackingSection
from app.models.runtime import Detection, TargetState
from app.tracking.models import TrackCandidate, TrackRecord
from app.tracking.state_machine import TrackingStateMachine
from app.tracking.target_selector import TargetSelector
from app.utils.geometry import bbox_iou, bbox_size_similarity, center_distance_normalized


@dataclass(slots=True)
class Tracker:
    """Internal multi-frame tracker with stable IDs and short-loss reactivation."""

    config: TrackingSection
    _tracks: dict[int, TrackRecord] = field(init=False, default_factory=dict)
    _next_track_id: int = field(init=False, default=1)
    _selector: TargetSelector = field(init=False)
    _state_machine: TrackingStateMachine = field(init=False)
    _state: TargetState = field(init=False, default_factory=lambda: TargetState(track_id=None, bbox_xyxy=None))

    def __post_init__(self) -> None:
        self._selector = TargetSelector(self.config)
        self._state_machine = TrackingStateMachine(self.config)

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
        for track in self._tracks.values():
            track.visible = False

        self._associate_tracks(detections, frame_width, frame_height, now)
        self._prune_stale_tracks(now)

        candidates = [
            TrackCandidate.from_record(track)
            for track in self._tracks.values()
            if track.visible
        ]
        selection = self._selector.select(candidates, self._state, frame_width, frame_height)
        self._state = self._state_machine.update(selection, self._state, now)
        return self._state

    def _associate_tracks(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> None:
        eligible_tracks = [
            track
            for track in self._tracks.values()
            if (now - track.last_seen_ts) <= self.config.lost_timeout_seconds
        ]
        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        scored_pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track in eligible_tracks:
                score = self._association_score(track, detection, frame_width, frame_height)
                if score > 0:
                    scored_pairs.append((score, track.track_id, detection_index))

        for _score, track_id, detection_index in sorted(scored_pairs, reverse=True):
            if track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            track = self._tracks[track_id]
            detection = detections[detection_index]
            detection.tracker_id = track_id
            track.update(detection, now, self.config.min_persist_frames)
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

        for track in eligible_tracks:
            if track.track_id not in matched_track_ids:
                track.mark_missed()

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            new_track_id = self._next_id()
            detection.tracker_id = new_track_id
            self._tracks[new_track_id] = TrackRecord(
                track_id=new_track_id,
                bbox_xyxy=detection.bbox_xyxy,
                confidence=detection.confidence,
                created_ts=now,
                last_seen_ts=now,
                confirmed=self.config.min_persist_frames <= 1,
            )

    def _association_score(
        self,
        track: TrackRecord,
        detection: Detection,
        frame_width: int,
        frame_height: int,
    ) -> float:
        iou = bbox_iou(track.bbox_xyxy, detection.bbox_xyxy)
        distance_limit = self.config.max_association_distance * (1.0 + min(1.0, 0.35 * track.missed_frames))
        distance = center_distance_normalized(
            track.bbox_xyxy,
            detection.bbox_xyxy,
            frame_width,
            frame_height,
        )
        if distance > distance_limit and iou < 0.05:
            return 0.0

        size_similarity = bbox_size_similarity(track.bbox_xyxy, detection.bbox_xyxy)
        score = (
            (0.40 * iou)
            + (0.35 * max(0.0, 1.0 - (distance / max(distance_limit, 1e-6))))
            + (0.15 * size_similarity)
            + (0.10 * detection.confidence)
        )
        if track.missed_frames:
            score -= min(0.15, 0.04 * track.missed_frames)
        if track.confirmed:
            score += 0.03
        return score if score >= 0.22 else 0.0

    def _prune_stale_tracks(self, now: float) -> None:
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if (now - track.last_seen_ts) > self.config.lost_timeout_seconds
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

    def _next_id(self) -> int:
        current = self._next_track_id
        self._next_track_id += 1
        return current
