from __future__ import annotations

from dataclasses import dataclass

from app.config import TrackingSection
from app.models.runtime import TargetMemory, TargetState, TrackingPhase


@dataclass(slots=True)
class LifecycleManager:
    """Resolves target visibility and memory into explicit lifecycle phases."""

    config: TrackingSection

    def next_phase(
        self,
        previous_phase: TrackingPhase,
        target: TargetState,
        memory: TargetMemory,
        *,
        handoff_ready: bool,
        handoff_zoom_candidate: bool,
        monitoring_broken: bool,
        return_home_issued: bool,
        now: float,
    ) -> TrackingPhase:
        if previous_phase == TrackingPhase.ERROR:
            return TrackingPhase.ERROR
        if return_home_issued:
            return TrackingPhase.RETURNING_HOME
        if target.visible:
            if not target.stable:
                return TrackingPhase.CANDIDATE_LOCK
            if previous_phase == TrackingPhase.HANDOFF:
                return TrackingPhase.MONITORING
            if previous_phase == TrackingPhase.MONITORING and not monitoring_broken:
                return TrackingPhase.MONITORING
            if handoff_ready:
                return TrackingPhase.HANDOFF if memory.handoff_ts is None else TrackingPhase.MONITORING
            if handoff_zoom_candidate:
                return TrackingPhase.ZOOMING_FOR_HANDOFF
            return TrackingPhase.CENTERING

        if memory.track_id is None:
            return TrackingPhase.SEARCHING

        loss_age = 0.0
        if memory.missing_started_ts is not None:
            loss_age = max(0.0, now - memory.missing_started_ts)

        if (
            loss_age <= self.config.recovery.short_loss_timeout_seconds
            or memory.consecutive_missing_frames <= self.config.recovery.missing_frame_count_short
        ):
            return TrackingPhase.TEMP_LOST
        if (
            loss_age <= self.config.recovery.occlusion_timeout_seconds
            or memory.consecutive_missing_frames <= self.config.recovery.missing_frame_count_occluded
            or memory.likely_occluded
        ):
            return TrackingPhase.OCCLUDED
        if loss_age <= self.config.recovery.recovery_local_timeout_seconds:
            return TrackingPhase.RECOVERY_LOCAL
        if loss_age <= self.config.recovery.recovery_wide_timeout_seconds:
            return TrackingPhase.RECOVERY_WIDE
        return TrackingPhase.LOST
