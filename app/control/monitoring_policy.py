from __future__ import annotations

from dataclasses import dataclass

from app.config import MonitoringSection
from app.models.runtime import TargetMemory, TargetState
from app.utils.geometry import normalized_bbox_center


@dataclass(slots=True)
class MonitoringPolicy:
    """Decides when monitoring mode should hand control back to recovery logic."""

    config: MonitoringSection

    def should_resume_control(
        self,
        target: TargetState,
        memory: TargetMemory,
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> tuple[bool, str]:
        if not target.visible:
            if memory.last_confirmed_ts <= 0:
                return True, "monitoring_no_recent_visibility"
            if now - memory.last_confirmed_ts >= self.config.handoff_break_timeout_seconds:
                return True, "handoff_visibility_timeout"
            return False, "monitoring_waiting_for_camera_tracker"
        if not self.config.break_on_large_error or target.bbox_xyxy is None:
            return False, "monitoring_target_stable"
        nx, ny = normalized_bbox_center(target.bbox_xyxy, frame_width, frame_height)
        center_error = max(abs(nx - 0.5), abs(ny - 0.5))
        if center_error > self.config.max_center_error:
            return True, "handoff_broken_large_center_error"
        return False, "monitoring_target_stable"
