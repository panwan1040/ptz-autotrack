from __future__ import annotations

from dataclasses import dataclass

from app.config import HandoffSection
from app.models.runtime import TargetMemory, TargetState
from app.utils.geometry import height_ratio, inside_dead_zone, normalized_bbox_center


@dataclass(slots=True)
class HandoffManager:
    """Determines when external control can safely hand off to camera tracking."""

    config: HandoffSection

    def evaluate(
        self,
        target: TargetState,
        memory: TargetMemory,
        frame_width: int,
        frame_height: int,
    ) -> tuple[bool, str]:
        if not self.config.enabled:
            memory.centered_frames = 0
            return False, "handoff_disabled"
        if not target.visible or not target.stable or target.bbox_xyxy is None:
            memory.centered_frames = 0
            return False, "target_not_stable"

        nx, ny = normalized_bbox_center(target.bbox_xyxy, frame_width, frame_height)
        if inside_dead_zone(nx, ny, self.config.inner_dead_zone_x, self.config.inner_dead_zone_y):
            memory.centered_frames += 1
        else:
            memory.centered_frames = 0

        target_ratio = height_ratio(target.bbox_xyxy, frame_height)
        if target_ratio < self.config.min_target_height_ratio:
            return False, "target_too_small_for_handoff"
        if target_ratio > self.config.max_target_height_ratio:
            return False, "target_too_large_for_handoff"
        if target.persist_frames < self.config.min_persist_frames:
            return False, "target_not_persistent_enough"
        if memory.centered_frames < self.config.stable_center_frames:
            return False, "target_not_centered_long_enough"
        return True, "handoff_ready"
