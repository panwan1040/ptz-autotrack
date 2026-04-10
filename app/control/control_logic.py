from __future__ import annotations

from dataclasses import dataclass, field

from app.config import ControlSection
from app.models.runtime import ControlDecision, ControlMode, PtzDirection, TargetState, TrackStatus
from app.utils.geometry import height_ratio, inside_dead_zone, normalized_bbox_center


@dataclass(slots=True)
class ControlLogic:
    """Converts target position into coarse/fine PTZ alignment intents."""

    config: ControlSection
    _last_mode: ControlMode = field(init=False, default=ControlMode.IDLE)
    _stable_hold_counter: int = field(init=False, default=0)

    def decide(self, target: TargetState, frame_width: int, frame_height: int) -> ControlDecision:
        decision = ControlDecision(reason="idle", control_mode=ControlMode.IDLE)
        if target.status != TrackStatus.TRACKING or target.bbox_xyxy is None:
            self._last_mode = ControlMode.IDLE
            self._stable_hold_counter = 0
            return decision

        if target.frame_age_seconds >= self.config.control_stale_frame_block_seconds:
            decision.reason = "stale_frame_blocked"
            decision.stale_frame_policy_state = "blocked"
            self._last_mode = ControlMode.IDLE
            return decision

        target_ratio = height_ratio(target.bbox_xyxy, frame_height)
        nx, ny = normalized_bbox_center(target.bbox_xyxy, frame_width, frame_height)
        prediction_used = False
        if (
            self.config.control_prediction_enabled
            and target.predicted_center is not None
            and target.prediction_confidence > 0.0
            and target.visible_frames >= self.config.control_prediction_min_history_points
        ):
            predicted_nx = target.predicted_center[0] / max(1.0, frame_width)
            predicted_ny = target.predicted_center[1] / max(1.0, frame_height)
            max_offset = self.config.control_prediction_max_offset_ratio
            lead_scale = max(0.1, self.config.control_prediction_lead_ms / 350.0)
            nx = self._clamp(nx + max(-max_offset, min(max_offset, (predicted_nx - nx) * lead_scale)))
            ny = self._clamp(ny + max(-max_offset, min(max_offset, (predicted_ny - ny) * lead_scale)))
            prediction_used = True
            decision.predicted_target_center = (nx * frame_width, ny * frame_height)
            decision.prediction_confidence = target.prediction_confidence
        else:
            decision.predicted_target_center = target.predicted_center
            decision.prediction_confidence = target.prediction_confidence

        error_x = nx - 0.5
        error_y = ny - 0.5
        decision.normalized_error_x = error_x
        decision.normalized_error_y = error_y
        decision.target_height_ratio = target_ratio
        decision.prediction_used = prediction_used

        if inside_dead_zone(nx, ny, self.config.fine_align_dead_zone_x, self.config.fine_align_dead_zone_y):
            self._stable_hold_counter += 1
            decision.control_mode = ControlMode.HOLD_STABLE
            if self._stable_hold_counter >= self.config.stable_hold_frames:
                decision.reason = "stable_hold"
            else:
                decision.reason = "within_inner_dead_zone"
            self._last_mode = ControlMode.HOLD_STABLE
            return decision

        self._stable_hold_counter = 0
        decision.control_mode = self._select_mode(error_x, error_y)
        self._last_mode = decision.control_mode
        decision.stale_frame_policy_state = (
            "reduced"
            if target.frame_age_seconds >= self.config.control_stale_frame_reduce_aggression_seconds
            else "fresh"
        )
        zoom_scale = self._zoom_compensation_scale(target_ratio)
        stale_scale = 0.6 if decision.stale_frame_policy_state == "reduced" else 1.0
        decision.zoom_compensation_scale = zoom_scale * stale_scale

        base_dead_zone_x = self.config.dead_zone_x if decision.control_mode == ControlMode.COARSE_ALIGN else self.config.fine_align_dead_zone_x
        base_dead_zone_y = self.config.dead_zone_y if decision.control_mode == ControlMode.COARSE_ALIGN else self.config.fine_align_dead_zone_y
        horizontal = self._horizontal_direction(error_x, base_dead_zone_x)
        vertical = self._vertical_direction(error_y, base_dead_zone_y)
        outside_ratio_x = self._outside_dead_zone_ratio(abs(error_x), base_dead_zone_x)
        outside_ratio_y = self._outside_dead_zone_ratio(abs(error_y), base_dead_zone_y)

        if horizontal and vertical:
            if abs(outside_ratio_x - outside_ratio_y) <= 0.16:
                decision.move_direction = self._combine(horizontal, vertical)
                decision.move_pulse_ms = self._scaled_pulse(
                    max(outside_ratio_x, outside_ratio_y),
                    min(self.config.pan_pulse_ms_small, self.config.tilt_pulse_ms_small),
                    max(self.config.diagonal_pulse_ms, self.config.pan_pulse_ms_large),
                    decision.control_mode,
                    decision.zoom_compensation_scale,
                )
                decision.reason = f"{decision.control_mode.value}_diagonal"
                return decision
            if outside_ratio_x >= outside_ratio_y:
                decision.move_direction = horizontal
                decision.move_pulse_ms = self._scaled_pulse(
                    outside_ratio_x,
                    self.config.pan_pulse_ms_small,
                    self.config.pan_pulse_ms_large,
                    decision.control_mode,
                    decision.zoom_compensation_scale,
                )
                decision.reason = f"{decision.control_mode.value}_horizontal_priority"
                return decision
            decision.move_direction = vertical
            decision.move_pulse_ms = self._scaled_pulse(
                outside_ratio_y,
                self.config.tilt_pulse_ms_small,
                self.config.tilt_pulse_ms_large,
                decision.control_mode,
                decision.zoom_compensation_scale,
            )
            decision.reason = f"{decision.control_mode.value}_vertical_priority"
            return decision

        if horizontal:
            decision.move_direction = horizontal
            decision.move_pulse_ms = self._scaled_pulse(
                outside_ratio_x,
                self.config.pan_pulse_ms_small,
                self.config.pan_pulse_ms_large,
                decision.control_mode,
                decision.zoom_compensation_scale,
            )
            decision.reason = f"{decision.control_mode.value}_horizontal"
            return decision

        if vertical:
            decision.move_direction = vertical
            decision.move_pulse_ms = self._scaled_pulse(
                outside_ratio_y,
                self.config.tilt_pulse_ms_small,
                self.config.tilt_pulse_ms_large,
                decision.control_mode,
                decision.zoom_compensation_scale,
            )
            decision.reason = f"{decision.control_mode.value}_vertical"
            return decision

        decision.reason = "within_response_hysteresis"
        return decision

    def _select_mode(self, error_x: float, error_y: float) -> ControlMode:
        abs_x = abs(error_x)
        abs_y = abs(error_y)
        if self._last_mode == ControlMode.COARSE_ALIGN and (abs_x >= self.config.dead_zone_x or abs_y >= self.config.dead_zone_y):
            return ControlMode.COARSE_ALIGN
        if abs_x >= self.config.coarse_align_threshold_x or abs_y >= self.config.coarse_align_threshold_y:
            return ControlMode.COARSE_ALIGN
        return ControlMode.FINE_ALIGN

    def _outside_dead_zone_ratio(self, absolute_error: float, dead_zone: float) -> float:
        available_range = max(1e-6, 0.5 - dead_zone)
        return min(1.0, max(0.0, (absolute_error - dead_zone) / available_range))

    def _scaled_pulse(
        self,
        magnitude: float,
        small_pulse: int,
        large_pulse: int,
        mode: ControlMode,
        scale: float,
    ) -> int:
        if large_pulse <= small_pulse:
            base = small_pulse
        else:
            base = int(round(small_pulse + ((large_pulse - small_pulse) * magnitude)))
        mode_scale = self.config.coarse_pulse_scale if mode == ControlMode.COARSE_ALIGN else self.config.fine_pulse_scale
        return max(40, int(round(base * mode_scale * scale)))

    def _zoom_compensation_scale(self, target_ratio: float) -> float:
        if not self.config.zoom_compensation_enabled:
            return 1.0
        if target_ratio >= self.config.zoom_compensation_high_ratio:
            return self.config.zoom_compensation_high_scale
        if target_ratio >= self.config.zoom_compensation_medium_ratio:
            return self.config.zoom_compensation_medium_scale
        if target_ratio >= self.config.zoom_compensation_low_ratio:
            return min(1.0, (self.config.zoom_compensation_medium_scale + 1.0) / 2.0)
        return 1.0

    def _horizontal_direction(self, error_x: float, threshold: float) -> PtzDirection | None:
        if error_x < -threshold:
            return PtzDirection.LEFT
        if error_x > threshold:
            return PtzDirection.RIGHT
        return None

    def _vertical_direction(self, error_y: float, threshold: float) -> PtzDirection | None:
        if error_y < -threshold:
            return PtzDirection.UP
        if error_y > threshold:
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

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
