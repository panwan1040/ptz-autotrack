from __future__ import annotations

from dataclasses import dataclass

from app.config import ControlSection
from app.models.runtime import ControlDecision, PtzDirection, TargetState, TrackStatus
from app.utils.geometry import height_ratio


@dataclass(slots=True)
class ZoomController:
    config: ControlSection

    def decide(self, target: TargetState, frame_height: int, pan_tilt_active: bool) -> ControlDecision:
        decision = ControlDecision(reason="zoom_idle")
        if not self.config.zoom.enabled or target.status != TrackStatus.TRACKING or target.bbox_xyxy is None:
            return decision
        if pan_tilt_active and not self.config.allow_zoom_during_pan_tilt:
            return decision
        ratio = height_ratio(target.bbox_xyxy, frame_height)
        decision.target_height_ratio = ratio
        if ratio < (self.config.zoom.min_height_ratio - self.config.zoom.hysteresis):
            decision.zoom_direction = PtzDirection.ZOOM_IN
            decision.zoom_pulse_ms = self.config.zoom_pulse_ms
            decision.reason = "zoom_in_target_small"
        elif ratio > (self.config.zoom.max_height_ratio + self.config.zoom.hysteresis):
            decision.zoom_direction = PtzDirection.ZOOM_OUT
            decision.zoom_pulse_ms = self.config.zoom_pulse_ms
            decision.reason = "zoom_out_target_large"
        return decision
