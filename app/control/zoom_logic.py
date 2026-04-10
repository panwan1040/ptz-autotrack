from __future__ import annotations

from dataclasses import dataclass

from app.config import ControlSection
from app.models.runtime import ControlDecision, ControlMode, PtzDirection, TargetState, TrackStatus
from app.utils.geometry import height_ratio


@dataclass(slots=True)
class ZoomController:
    """Applies guarded zoom decisions so zoom does not fight centering."""

    config: ControlSection

    def decide(
        self,
        target: TargetState,
        frame_height: int,
        pan_tilt_active: bool,
        normalized_error_x: float = 0.0,
        normalized_error_y: float = 0.0,
    ) -> ControlDecision:
        decision = ControlDecision(reason="zoom_idle")
        if not self.config.zoom.enabled or target.status != TrackStatus.TRACKING or target.bbox_xyxy is None:
            return decision
        if target.frame_age_seconds >= self.config.control_stale_frame_block_seconds:
            decision.reason = "zoom_blocked_stale_frame"
            decision.stale_frame_policy_state = "blocked"
            return decision
        if pan_tilt_active and not self.config.allow_zoom_during_pan_tilt:
            decision.reason = "zoom_blocked_pan_tilt_active"
            return decision
        if target.persist_frames < 2:
            decision.reason = "zoom_blocked_target_unstable"
            return decision
        if target.centered_frames < self.config.stable_hold_frames and target.handoff_ready is False:
            decision.reason = "zoom_blocked_align_first"
            return decision

        ratio = height_ratio(target.bbox_xyxy, frame_height)
        decision.target_height_ratio = ratio
        offset = max(abs(normalized_error_x), abs(normalized_error_y))
        far_from_center = offset > max(self.config.dead_zone_x, self.config.dead_zone_y) * 1.6

        if ratio < (self.config.zoom.min_height_ratio - self.config.zoom.hysteresis):
            if far_from_center:
                decision.reason = "zoom_blocked_recenter_first"
                return decision
            decision.zoom_direction = PtzDirection.ZOOM_IN
            decision.zoom_pulse_ms = self._zoom_pulse_ms(target.persist_frames)
            decision.reason = "zoom_in_target_small"
            decision.control_mode = ControlMode.FINE_ALIGN
        elif ratio > (self.config.zoom.max_height_ratio + self.config.zoom.hysteresis):
            decision.zoom_direction = PtzDirection.ZOOM_OUT
            decision.zoom_pulse_ms = self.config.zoom_pulse_ms
            decision.reason = "zoom_out_target_large"
            decision.control_mode = ControlMode.FINE_ALIGN
        return decision

    def _zoom_pulse_ms(self, persist_frames: int) -> int:
        if persist_frames >= 6:
            return self.config.zoom_pulse_ms
        return max(60, int(round(self.config.zoom_pulse_ms * 0.65)))
