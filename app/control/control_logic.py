from __future__ import annotations

from dataclasses import dataclass

from app.config import ControlSection
from app.models.runtime import ControlDecision, PtzDirection, TargetState, TrackStatus
from app.utils.geometry import height_ratio, inside_dead_zone, normalized_bbox_center


@dataclass(slots=True)
class ControlLogic:
    config: ControlSection

    def decide(self, target: TargetState, frame_width: int, frame_height: int) -> ControlDecision:
        decision = ControlDecision(reason="idle")
        if target.status != TrackStatus.TRACKING or target.bbox_xyxy is None:
            return decision
        nx, ny = normalized_bbox_center(target.bbox_xyxy, frame_width, frame_height)
        decision.normalized_error_x = nx - 0.5
        decision.normalized_error_y = ny - 0.5
        decision.target_height_ratio = height_ratio(target.bbox_xyxy, frame_height)
        if inside_dead_zone(nx, ny, self.config.dead_zone_x, self.config.dead_zone_y):
            decision.reason = "within_dead_zone"
            return decision

        horizontal = None
        vertical = None
        if nx < 0.5 - self.config.dead_zone_x:
            horizontal = PtzDirection.LEFT
        elif nx > 0.5 + self.config.dead_zone_x:
            horizontal = PtzDirection.RIGHT
        if ny < 0.5 - self.config.dead_zone_y:
            vertical = PtzDirection.UP
        elif ny > 0.5 + self.config.dead_zone_y:
            vertical = PtzDirection.DOWN

        if horizontal and vertical:
            decision.move_direction = self._combine(horizontal, vertical)
            decision.move_pulse_ms = self.config.diagonal_pulse_ms
            decision.reason = "diagonal_correction"
            return decision
        if horizontal:
            decision.move_direction = horizontal
            error = abs(nx - 0.5)
            decision.move_pulse_ms = (
                self.config.pan_pulse_ms_large
                if error >= self.config.aggressive_pan_threshold
                else self.config.pan_pulse_ms_small
            )
            decision.reason = "horizontal_correction"
            return decision
        if vertical:
            decision.move_direction = vertical
            error = abs(ny - 0.5)
            decision.move_pulse_ms = (
                self.config.tilt_pulse_ms_large
                if error >= self.config.aggressive_tilt_threshold
                else self.config.tilt_pulse_ms_small
            )
            decision.reason = "vertical_correction"
        return decision

    def _combine(self, horizontal: PtzDirection, vertical: PtzDirection) -> PtzDirection:
        mapping = {
            (PtzDirection.LEFT, PtzDirection.UP): PtzDirection.LEFT_UP,
            (PtzDirection.RIGHT, PtzDirection.UP): PtzDirection.RIGHT_UP,
            (PtzDirection.LEFT, PtzDirection.DOWN): PtzDirection.LEFT_DOWN,
            (PtzDirection.RIGHT, PtzDirection.DOWN): PtzDirection.RIGHT_DOWN,
        }
        return mapping[(horizontal, vertical)]
