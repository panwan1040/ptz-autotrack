from __future__ import annotations

from dataclasses import dataclass

from app.config import PredictionSection, RecoverySection
from app.models.runtime import TargetMemory
from app.utils.geometry import BBox, bbox_center, bbox_height, bbox_width


@dataclass(slots=True)
class MotionPredictor:
    """Predicts short-term target position from recent smoothed center history."""

    prediction: PredictionSection
    recovery: RecoverySection

    def predict(
        self,
        memory: TargetMemory,
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[float, float] | None, BBox | None, float]:
        if (
            not self.prediction.enabled
            or len(memory.recent_centers) < self.prediction.min_history_points
            or len(memory.recent_timestamps) < self.prediction.min_history_points
        ):
            return memory.last_center, self._window_from_memory(memory, frame_width, frame_height), 0.0

        velocities: list[tuple[float, float]] = []
        for index in range(1, min(len(memory.recent_centers), len(memory.recent_timestamps))):
            dt = memory.recent_timestamps[index] - memory.recent_timestamps[index - 1]
            if dt <= 1e-6:
                continue
            prev_x, prev_y = memory.recent_centers[index - 1]
            cur_x, cur_y = memory.recent_centers[index]
            velocities.append(((cur_x - prev_x) / dt, (cur_y - prev_y) / dt))
        if not velocities:
            return memory.last_center, self._window_from_memory(memory, frame_width, frame_height), 0.0

        velocity_x = sum(vx for vx, _ in velocities) / len(velocities)
        velocity_y = sum(vy for _, vy in velocities) / len(velocities)
        lead_time = self.prediction.lead_time_seconds
        displacement_limit_x = frame_width * self.prediction.max_normalized_displacement
        displacement_limit_y = frame_height * self.prediction.max_normalized_displacement
        last_x, last_y = memory.recent_centers[-1]
        predicted_x = min(
            frame_width,
            max(0.0, last_x + max(-displacement_limit_x, min(displacement_limit_x, velocity_x * lead_time))),
        )
        predicted_y = min(
            frame_height,
            max(0.0, last_y + max(-displacement_limit_y, min(displacement_limit_y, velocity_y * lead_time))),
        )
        confidence = min(1.0, 0.35 + (0.2 * len(velocities)))
        return (
            (predicted_x, predicted_y),
            self._window_from_prediction(memory, (predicted_x, predicted_y), frame_width, frame_height),
            confidence,
        )

    def _window_from_prediction(
        self,
        memory: TargetMemory,
        center: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> BBox | None:
        if memory.last_smoothed_bbox is None and memory.last_confirmed_bbox is None:
            return None
        base_bbox = memory.last_smoothed_bbox or memory.last_confirmed_bbox
        assert base_bbox is not None
        width = max(bbox_width(base_bbox), frame_width * self.recovery.local_search_window_ratio)
        height = max(bbox_height(base_bbox), frame_height * self.recovery.local_search_window_ratio)
        half_w = width / 2.0
        half_h = height / 2.0
        cx, cy = center
        return (
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(float(frame_width), cx + half_w),
            min(float(frame_height), cy + half_h),
        )

    def _window_from_memory(
        self,
        memory: TargetMemory,
        frame_width: int,
        frame_height: int,
    ) -> BBox | None:
        bbox = memory.last_smoothed_bbox or memory.last_confirmed_bbox
        if bbox is None:
            return None
        center = memory.last_center or bbox_center(bbox)
        return self._window_from_prediction(memory, center, frame_width, frame_height)
