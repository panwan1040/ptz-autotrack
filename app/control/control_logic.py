from __future__ import annotations

from dataclasses import dataclass

from app.config import ControlSection
from app.models.runtime import ControlDecision, PtzDirection, TargetState, TrackStatus
from app.utils.geometry import height_ratio, inside_dead_zone, normalized_bbox_center


@dataclass(slots=True)
class ControlLogic:
    """Converts target position into pulse-based pan/tilt decisions."""

    config: ControlSection

    def decide(self, target: TargetState, frame_width: int, frame_height: int) -> ControlDecision:
        decision = ControlDecision(reason="idle")
        if target.status != TrackStatus.TRACKING or target.bbox_xyxy is None:
            return decision

        nx, ny = normalized_bbox_center(target.bbox_xyxy, frame_width, frame_height)
        error_x = nx - 0.5
        error_y = ny - 0.5
        decision.normalized_error_x = error_x
        decision.normalized_error_y = error_y
        decision.target_height_ratio = height_ratio(target.bbox_xyxy, frame_height)
        if inside_dead_zone(nx, ny, self.config.dead_zone_x, self.config.dead_zone_y):
            decision.reason = "within_dead_zone"
            return decision

        horizontal = self._horizontal_direction(error_x)
        vertical = self._vertical_direction(error_y)
        outside_ratio_x = self._outside_dead_zone_ratio(abs(error_x), self.config.dead_zone_x)
        outside_ratio_y = self._outside_dead_zone_ratio(abs(error_y), self.config.dead_zone_y)

        if horizontal and vertical:
            if abs(outside_ratio_x - outside_ratio_y) <= 0.18:
                decision.move_direction = self._combine(horizontal, vertical)
                decision.move_pulse_ms = self._adaptive_pulse(
                    max(outside_ratio_x, outside_ratio_y),
                    min(self.config.pan_pulse_ms_small, self.config.tilt_pulse_ms_small),
                    max(self.config.diagonal_pulse_ms, self.config.pan_pulse_ms_large),
                )
                decision.reason = "diagonal_balanced_correction"
                return decision
            if outside_ratio_x > outside_ratio_y:
                decision.move_direction = horizontal
                decision.move_pulse_ms = self._adaptive_pulse(
                    outside_ratio_x,
                    self.config.pan_pulse_ms_small,
                    self.config.pan_pulse_ms_large,
                )
                decision.reason = "horizontal_priority_correction"
                return decision
            decision.move_direction = vertical
            decision.move_pulse_ms = self._adaptive_pulse(
                outside_ratio_y,
                self.config.tilt_pulse_ms_small,
                self.config.tilt_pulse_ms_large,
            )
            decision.reason = "vertical_priority_correction"
            return decision

        if horizontal:
            decision.move_direction = horizontal
            decision.move_pulse_ms = self._adaptive_pulse(
                outside_ratio_x,
                self.config.pan_pulse_ms_small,
                self.config.pan_pulse_ms_large,
            )
            decision.reason = "horizontal_correction"
            return decision

        if vertical:
            decision.move_direction = vertical
            decision.move_pulse_ms = self._adaptive_pulse(
                outside_ratio_y,
                self.config.tilt_pulse_ms_small,
                self.config.tilt_pulse_ms_large,
            )
            decision.reason = "vertical_correction"
        return decision

    def _outside_dead_zone_ratio(self, absolute_error: float, dead_zone: float) -> float:
        available_range = max(1e-6, 0.5 - dead_zone)
        return min(1.0, max(0.0, (absolute_error - dead_zone) / available_range))

    def _adaptive_pulse(self, magnitude: float, small_pulse: int, large_pulse: int) -> int:
        if large_pulse <= small_pulse:
            return small_pulse
        return int(round(small_pulse + ((large_pulse - small_pulse) * magnitude)))

    def _horizontal_direction(self, error_x: float) -> PtzDirection | None:
        if error_x < -self.config.dead_zone_x:
            return PtzDirection.LEFT
        if error_x > self.config.dead_zone_x:
            return PtzDirection.RIGHT
        return None

    def _vertical_direction(self, error_y: float) -> PtzDirection | None:
        if error_y < -self.config.dead_zone_y:
            return PtzDirection.UP
        if error_y > self.config.dead_zone_y:
            return PtzDirection.DOWN
        return None

    def _combine(self, horizontal: PtzDirection, vertical: PtzDirection) -> PtzDirection:
        mapping = {
            (PtzDirection.LEFT, PtzDirection.UP): PtzDirection.LEFT_UP,
            (PtzDirection.RIGHT, PtzDirection.UP): PtzDirection.RIGHT_UP,
            (PtzDirection.LEFT, PtzDirection.DOWN): PtzDirection.LEFT_DOWN,
            (PtzDirection.RIGHT, PtzDirection.DOWN): PtzDirection.RIGHT_DOWN,
        }
        return mapping[(horizontal, vertical)]
