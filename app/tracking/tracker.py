from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

from app.config import TrackingSection
from app.models.runtime import Detection, TargetMemory, TargetState
from app.tracking.appearance_extractor import AppearanceExtractor
from app.tracking.models import SelectionResult, TrackCandidate, TrackRecord
from app.tracking.motion_predictor import MotionPredictor
from app.tracking.state_machine import TrackingStateMachine
from app.tracking.target_selector import TargetSelector
from app.utils.geometry import bbox_center, bbox_iou, bbox_size_similarity, center_distance_normalized, height_ratio


T = TypeVar("T")


@dataclass(slots=True)
class Tracker:
    """Internal tracker with sticky target memory, prediction, and short-loss continuity."""

    config: TrackingSection
    _tracks: dict[int, TrackRecord] = field(init=False, default_factory=dict)
    _next_track_id: int = field(init=False, default=1)
    _selector: TargetSelector = field(init=False)
    _state_machine: TrackingStateMachine = field(init=False)
    _appearance_extractor: AppearanceExtractor = field(init=False)
    _motion_predictor: MotionPredictor = field(init=False)
    _state: TargetState = field(init=False, default_factory=lambda: TargetState(track_id=None, bbox_xyxy=None))
    _memory: TargetMemory = field(init=False, default_factory=TargetMemory)

    def __post_init__(self) -> None:
        self._selector = TargetSelector(self.config)
        self._state_machine = TrackingStateMachine(self.config)
        self._appearance_extractor = AppearanceExtractor(self.config.appearance)
        self._motion_predictor = MotionPredictor(self.config.prediction, self.config.recovery)

    @property
    def state(self) -> TargetState:
        return self._state

    @property
    def target_memory(self) -> TargetMemory:
        return self._memory

    def update(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        now: float,
        frame: np.ndarray | None = None,
    ) -> TargetState:
        detection_signatures = self._extract_detection_signatures(detections, frame)
        for track in self._tracks.values():
            track.visible = False

        self._associate_tracks(detections, detection_signatures, frame_width, frame_height, now)
        self._prune_stale_tracks(now)
        self._update_memory_prediction(frame_width, frame_height)

        candidates = [
            TrackCandidate.from_record(track)
            for track in self._tracks.values()
            if track.visible
        ]
        selection = self._selector.select(candidates, self._state, self._memory, frame_width, frame_height)
        self._state = self._state_machine.update(selection, self._state, now)
        self._state.predicted_center = self._memory.predicted_center
        self._state.predicted_window = self._memory.predicted_window
        self._state.appearance_similarity = self._memory.appearance_similarity
        self._state.match_breakdown = {}
        if selection.candidate is not None and selection.candidate.match_breakdown is not None:
            self._state.match_breakdown = selection.candidate.match_breakdown
        self._sync_memory_with_state(selection, frame_height, now)
        return self._state

    def clear_target_memory(self) -> None:
        self._memory = TargetMemory()

    def _extract_detection_signatures(
        self,
        detections: list[Detection],
        frame: np.ndarray | None,
    ) -> dict[int, list[float] | None]:
        if frame is None:
            return {index: None for index, _detection in enumerate(detections)}
        return {
            index: self._appearance_extractor.extract(frame, detection.bbox_xyxy)
            for index, detection in enumerate(detections)
        }

    def _associate_tracks(
        self,
        detections: list[Detection],
        detection_signatures: dict[int, list[float] | None],
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
                score = self._association_score(
                    track,
                    detection,
                    detection_signatures.get(detection_index),
                    frame_width,
                    frame_height,
                )
                if score > 0:
                    scored_pairs.append((score, track.track_id, detection_index))

        for _score, track_id, detection_index in sorted(scored_pairs, reverse=True):
            if track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            track = self._tracks[track_id]
            detection = detections[detection_index]
            detection.tracker_id = track_id
            track.update(
                detection,
                now,
                self.config.min_persist_frames,
                appearance_signature=self._appearance_extractor.blend(
                    track.appearance_signature,
                    detection_signatures.get(detection_index),
                ),
            )
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
                appearance_signature=detection_signatures.get(detection_index),
            )

    def _association_score(
        self,
        track: TrackRecord,
        detection: Detection,
        appearance_signature: list[float] | None,
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
        appearance_similarity = self._appearance_extractor.similarity(track.appearance_signature, appearance_signature)
        score = (
            (0.30 * iou)
            + (0.24 * max(0.0, 1.0 - (distance / max(distance_limit, 1e-6))))
            + (0.12 * size_similarity)
            + (0.12 * detection.confidence)
            + (0.22 * appearance_similarity)
        )
        if track.missed_frames:
            score -= min(0.15, 0.04 * track.missed_frames)
        if track.confirmed:
            score += 0.03
        return score if score >= 0.24 else 0.0

    def _update_memory_prediction(self, frame_width: int, frame_height: int) -> None:
        predicted_center, predicted_window, confidence = self._motion_predictor.predict(
            self._memory,
            frame_width,
            frame_height,
        )
        self._memory.predicted_center = predicted_center
        self._memory.predicted_window = predicted_window
        self._memory.prediction_confidence = confidence

    def _sync_memory_with_state(
        self,
        selection: SelectionResult,
        frame_height: int,
        now: float,
    ) -> None:
        state = self._state
        if state.visible and state.track_id is not None and state.bbox_xyxy is not None:
            if self._memory.track_id != state.track_id:
                self._memory = TargetMemory(track_id=state.track_id, acquisition_ts=now)
            center = bbox_center(state.bbox_xyxy)
            self._memory.track_id = state.track_id
            self._memory.last_confirmed_ts = now
            self._memory.last_confirmed_bbox = state.bbox_xyxy
            self._memory.last_smoothed_bbox = state.bbox_xyxy
            self._memory.last_center = center
            self._memory.last_zoom_ratio = height_ratio(state.bbox_xyxy, frame_height)
            self._memory.consecutive_visible_frames += 1
            self._memory.consecutive_missing_frames = 0
            self._memory.missing_started_ts = None
            self._memory.likely_occluded = False
            self._memory.recent_centers = self._trimmed_list(
                self._memory.recent_centers,
                center,
                self.config.target_memory.max_history_points,
            )
            self._memory.recent_timestamps = self._trimmed_list(
                self._memory.recent_timestamps,
                now,
                self.config.target_memory.max_history_points,
            )
            self._memory.recent_confidences = self._trimmed_list(
                self._memory.recent_confidences,
                state.confidence,
                self.config.target_memory.confidence_window,
            )
            self._memory.confidence_average = (
                sum(self._memory.recent_confidences) / max(1, len(self._memory.recent_confidences))
            )
            if len(self._memory.recent_centers) >= 2 and len(self._memory.recent_timestamps) >= 2:
                prev_center = self._memory.recent_centers[-2]
                prev_ts = self._memory.recent_timestamps[-2]
                dt = max(1e-6, now - prev_ts)
                velocity = ((center[0] - prev_center[0]) / dt, (center[1] - prev_center[1]) / dt)
                self._memory.last_velocity = velocity
                self._memory.last_direction = velocity
            if selection.candidate is not None:
                self._memory.last_match_score = state.candidate_score
                self._memory.appearance_similarity = (selection.candidate.match_breakdown or {}).get("appearance", 0.0)
                self._memory.appearance_signature = self._appearance_extractor.blend(
                    self._memory.appearance_signature,
                    selection.candidate.appearance_signature,
                )
            if self._memory.acquisition_ts <= 0:
                self._memory.acquisition_ts = now
            return

        if self._memory.track_id is None:
            return
        self._memory.consecutive_visible_frames = 0
        self._memory.consecutive_missing_frames += 1
        if self._memory.missing_started_ts is None:
            self._memory.missing_started_ts = now
        self._memory.likely_occluded = (
            self._memory.consecutive_missing_frames >= self.config.recovery.missing_frame_count_occluded
        )

        memory_age = now - self._memory.last_confirmed_ts
        if memory_age >= self.config.target_memory.clear_after_seconds:
            self.clear_target_memory()

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

    def _trimmed_list(self, values: list[T], value: T, max_items: int) -> list[T]:
        updated = [*values, value]
        if len(updated) <= max_items:
            return updated
        return updated[-max_items:]
