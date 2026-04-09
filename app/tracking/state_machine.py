from __future__ import annotations

from dataclasses import dataclass

from app.config import TrackingSection
from app.models.runtime import TargetState, TrackStatus
from app.tracking.models import SelectionResult


@dataclass(slots=True)
class TrackingStateMachine:
    """Resolves selection into stable target visibility while preserving short-term loss."""

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

        required_confirm_frames = self._required_confirmation_frames(previous)
        if candidate.confirmed and candidate.persist_frames >= required_confirm_frames:
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
                visible_frames=candidate.total_visible_frames,
                missing_frames=0,
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
                missing_frames=lost.missing_frames,
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
            visible_frames=candidate.persist_frames,
            missing_frames=0,
            match_breakdown=candidate.match_breakdown or {},
        )

    def _required_confirmation_frames(self, previous: TargetState) -> int:
        if previous.status == TrackStatus.LOST and previous.missing_frames >= self.config.recovery.missing_frame_count_occluded:
            return max(1, self.config.recovery.post_wide_recovery_confirm_frames)
        if previous.status == TrackStatus.LOST:
            return max(1, self.config.recovery.post_occlusion_confirm_frames)
        return max(self.config.min_persist_frames, self.config.recovery.initial_confirm_frames)

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
                missing_frames=0,
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
                missing_frames=previous.missing_frames + 1,
                visible_frames=previous.visible_frames,
                predicted_center=previous.predicted_center,
                predicted_window=previous.predicted_window,
                appearance_similarity=previous.appearance_similarity,
                match_breakdown=previous.match_breakdown.copy(),
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
            missing_frames=previous.missing_frames + 1,
        )
