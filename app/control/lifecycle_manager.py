from __future__ import annotations

from dataclasses import dataclass, field

from app.config import TrackingSection
from app.models.runtime import TargetMemory, TargetState, TrackingPhase


@dataclass(frozen=True, slots=True)
class PhasePolicy:
    pan_tilt_allowed: bool
    zoom_allowed: bool
    handoff_allowed: bool
    recovery_allowed: bool
    switching_allowed: bool


@dataclass(slots=True)
class LifecycleManager:
    """Resolves target visibility and memory into explicit lifecycle phases.

    Phase policy guide:
    - `SEARCHING` / `CANDIDATE_LOCK`: observe only, no PTZ corrections.
    - `CENTERING`: pan/tilt allowed, zoom gated unless alignment is already fine.
    - `ZOOMING_FOR_HANDOFF`: conservative zoom allowed, pan/tilt still allowed.
    - `HANDOFF` / `MONITORING`: external PTZ holds unless monitoring breaks.
    - `TEMP_LOST` / `OCCLUDED`: preserve identity, avoid switching, no aggressive PTZ.
    - `RECOVERY_ZOOM_OUT`: widen FOV in bounded steps before directional recovery.
    - `RECOVERY_LOCAL` / `RECOVERY_WIDE`: recovery PTZ allowed with staged behavior.
    - `LOST` / `RETURNING_HOME`: no target-follow PTZ, only safe reset/home behaviors.
    """

    config: TrackingSection
    _policies: dict[TrackingPhase, PhasePolicy] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._policies = {
            TrackingPhase.IDLE: PhasePolicy(False, False, False, False, False),
            TrackingPhase.SEARCHING: PhasePolicy(False, False, False, False, True),
            TrackingPhase.CANDIDATE_LOCK: PhasePolicy(False, False, False, False, False),
            TrackingPhase.CENTERING: PhasePolicy(True, False, True, False, False),
            TrackingPhase.ZOOMING_FOR_HANDOFF: PhasePolicy(True, True, True, False, False),
            TrackingPhase.HANDOFF: PhasePolicy(False, False, False, False, False),
            TrackingPhase.MONITORING: PhasePolicy(False, False, False, False, False),
            TrackingPhase.TEMP_LOST: PhasePolicy(False, False, False, True, False),
            TrackingPhase.OCCLUDED: PhasePolicy(False, False, False, True, False),
            TrackingPhase.RECOVERY_ZOOM_OUT: PhasePolicy(False, True, False, True, False),
            TrackingPhase.RECOVERY_LOCAL: PhasePolicy(True, True, False, True, False),
            TrackingPhase.RECOVERY_WIDE: PhasePolicy(False, True, False, True, False),
            TrackingPhase.TRACKING: PhasePolicy(True, True, True, False, False),
            TrackingPhase.LOST: PhasePolicy(False, False, False, True, True),
            TrackingPhase.RETURNING_HOME: PhasePolicy(False, False, False, False, False),
            TrackingPhase.ERROR: PhasePolicy(False, False, False, False, False),
        }

    def policy_for(self, phase: TrackingPhase) -> PhasePolicy:
        return self._policies[phase]

    def next_phase(
        self,
        previous_phase: TrackingPhase,
        target: TargetState,
        memory: TargetMemory,
        *,
        handoff_ready: bool,
        handoff_zoom_candidate: bool,
        monitoring_broken: bool,
        visible_candidate_count: int,
        tight_zoom_detected: bool,
        return_home_issued: bool,
        now: float,
    ) -> TrackingPhase:
        if previous_phase == TrackingPhase.ERROR:
            return TrackingPhase.ERROR
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
        if return_home_issued:
            return TrackingPhase.RETURNING_HOME if memory.return_home_pending else TrackingPhase.SEARCHING

        if memory.track_id is None:
            return TrackingPhase.SEARCHING

        loss_age = 0.0
        if memory.missing_started_ts is not None:
            loss_age = max(0.0, now - memory.missing_started_ts)
        no_visible_candidates = visible_candidate_count <= 0
        zoom_out_ready = (
            no_visible_candidates
            and tight_zoom_detected
            and loss_age >= self.config.recovery.recovery_zoom_out_start_timeout_seconds
            and memory.recovery_zoom_steps < self.config.recovery.max_recovery_zoom_steps
        )

        if (
            loss_age <= self.config.recovery.short_loss_timeout_seconds
            or memory.consecutive_missing_frames <= self.config.recovery.missing_frame_count_short
        ):
            return TrackingPhase.TEMP_LOST
        if previous_phase == TrackingPhase.RECOVERY_ZOOM_OUT and memory.recovery_settle_ticks_remaining > 0:
            return TrackingPhase.RECOVERY_ZOOM_OUT
        if zoom_out_ready:
            return TrackingPhase.RECOVERY_ZOOM_OUT
        if (
            previous_phase != TrackingPhase.RECOVERY_ZOOM_OUT
            and previous_phase != TrackingPhase.RECOVERY_LOCAL
            and previous_phase != TrackingPhase.RECOVERY_WIDE
            and (
            loss_age <= self.config.recovery.occlusion_timeout_seconds
            or memory.consecutive_missing_frames <= self.config.recovery.missing_frame_count_occluded
            or memory.likely_occluded
            )
        ):
            return TrackingPhase.OCCLUDED
        if loss_age <= self.config.recovery.recovery_local_timeout_seconds:
            return TrackingPhase.RECOVERY_LOCAL
        if loss_age <= self.config.recovery.recovery_wide_timeout_seconds:
            return TrackingPhase.RECOVERY_WIDE
        return TrackingPhase.LOST
