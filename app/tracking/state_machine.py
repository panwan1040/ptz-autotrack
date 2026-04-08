from __future__ import annotations

from dataclasses import dataclass

from app.config import TrackingSection
from app.models.runtime import TargetState, TrackStatus
from app.tracking.models import SelectionResult


@dataclass(slots=True)
class TrackingStateMachine:
    """Resolves tracker selection into explicit SEARCHING/TRACKING/LOST states."""

    config: TrackingSection

    def update(
        self,
        selection: SelectionResult,
        previous: TargetState,
        now: float,
    ) -> TargetState:
        candidate = selection.candidate
        if candidate is None:
            return self._on_missing(previous, now, selection.reason)

        if candidate.confirmed:
            return TargetState(
                track_id=candidate.track_id,
                bbox_xyxy=candidate.bbox_xyxy,
                confidence=candidate.confidence,
                persist_frames=candidate.total_visible_frames,
                last_seen_ts=now,
                status=TrackStatus.TRACKING,
                stable=True,
                visible=True,
                selection_reason=selection.reason,
                candidate_score=selection.score,
            )

        if previous.stable and previous.track_id not in (None, candidate.track_id):
            lost = self._on_missing(previous, now, "warming_new_candidate_while_previous_lost")
            return TargetState(
                track_id=lost.track_id,
                bbox_xyxy=lost.bbox_xyxy,
                confidence=lost.confidence,
                persist_frames=lost.persist_frames,
                last_seen_ts=lost.last_seen_ts,
                status=lost.status,
                stable=lost.stable,
                visible=False,
                selection_reason=lost.selection_reason,
                candidate_score=selection.score,
                lost_duration_seconds=lost.lost_duration_seconds,
            )

        return TargetState(
            track_id=candidate.track_id,
            bbox_xyxy=candidate.bbox_xyxy,
            confidence=candidate.confidence,
            persist_frames=candidate.persist_frames,
            last_seen_ts=now,
            status=TrackStatus.SEARCHING,
            stable=False,
            visible=True,
            selection_reason=selection.reason,
            candidate_score=selection.score,
        )

    def _on_missing(self, previous: TargetState, now: float, reason: str) -> TargetState:
        if previous.track_id is None or not previous.stable:
            return TargetState(
                track_id=None,
                bbox_xyxy=None,
                last_seen_ts=now,
                status=TrackStatus.SEARCHING,
                stable=False,
                visible=False,
                selection_reason=reason,
            )

        lost_duration = max(0.0, now - previous.last_seen_ts)
        if lost_duration <= self.config.lost_timeout_seconds:
            return TargetState(
                track_id=previous.track_id,
                bbox_xyxy=previous.bbox_xyxy,
                confidence=0.0,
                persist_frames=previous.persist_frames,
                last_seen_ts=previous.last_seen_ts,
                status=TrackStatus.LOST,
                stable=True,
                visible=False,
                selection_reason=reason,
                candidate_score=previous.candidate_score,
                lost_duration_seconds=lost_duration,
            )

        return TargetState(
            track_id=None,
            bbox_xyxy=None,
            last_seen_ts=now,
            status=TrackStatus.SEARCHING,
            stable=False,
            visible=False,
            selection_reason="lost_timeout_elapsed",
            lost_duration_seconds=lost_duration,
        )
