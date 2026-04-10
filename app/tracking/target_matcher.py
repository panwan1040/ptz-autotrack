from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from app.config import TrackingSection
from app.models.runtime import TargetMemory
from app.tracking.appearance_extractor import AppearanceExtractor
from app.tracking.models import TrackCandidate
from app.utils.geometry import (
    BBox,
    bbox_area,
    bbox_center,
    bbox_size_similarity,
    center_distance_normalized,
    normalized_bbox_center,
)


@dataclass(slots=True)
class TargetMatcher:
    """Scores candidates against target memory using geometry, motion, and appearance."""

    config: TrackingSection
    appearance_extractor: AppearanceExtractor

    def score(
        self,
        candidate: TrackCandidate,
        memory: TargetMemory | None,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, dict[str, float]]:
        weights = self.config.matching_weights
        candidate_bbox = self._candidate_bbox(candidate)
        nx, ny = normalized_bbox_center(candidate_bbox, frame_width, frame_height)
        centeredness = max(0.0, 1.0 - (abs(nx - 0.5) + abs(ny - 0.5)))
        persistence = min(1.0, candidate.total_visible_frames / max(1, self.config.min_persist_frames))
        predicted_score = 0.5
        last_center_score = 0.5
        size_similarity = 0.5
        motion_consistency = 0.5
        appearance_similarity = 0.5

        if memory is not None and memory.track_id is not None:
            if memory.predicted_center is not None:
                predicted_bbox = self._box_from_center(candidate, memory.predicted_center)
                predicted_score = max(
                    0.0,
                    1.0 - center_distance_normalized(
                        candidate_bbox,
                        predicted_bbox,
                        frame_width,
                        frame_height,
                    ) / max(1e-6, self.config.max_association_distance * 1.5),
                )
            if memory.last_center is not None:
                last_center_score = max(
                    0.0,
                    1.0 - self._center_distance(candidate_bbox, memory.last_center, frame_width, frame_height),
                )
                movement = self._movement_vector(candidate_bbox, memory.last_center)
                motion_consistency = self._direction_similarity(memory.last_direction, movement)
            if memory.last_confirmed_bbox is not None:
                size_similarity = bbox_size_similarity(candidate_bbox, memory.last_confirmed_bbox)
            appearance_similarity = self.appearance_extractor.similarity(
                memory.appearance_signature,
                candidate.appearance_signature,
            )

        breakdown = {
            "confidence": round(candidate.confidence, 4),
            "predicted_center": round(predicted_score, 4),
            "last_center": round(last_center_score, 4),
            "size_similarity": round(size_similarity, 4),
            "motion_consistency": round(motion_consistency, 4),
            "appearance": round(appearance_similarity, 4),
            "centeredness": round(centeredness, 4),
            "persistence": round(persistence, 4),
        }
        total = (
            (weights.confidence * candidate.confidence)
            + (weights.predicted_center * predicted_score)
            + (weights.last_center * last_center_score)
            + (weights.size_similarity * size_similarity)
            + (weights.motion_consistency * motion_consistency)
            + (weights.appearance * appearance_similarity)
            + (weights.centeredness * centeredness)
            + (weights.persistence * persistence)
        )
        if memory is not None and candidate.track_id == memory.track_id:
            total += 0.08
            breakdown["continuity_bonus"] = 0.08
        return total, breakdown

    def _box_from_center(
        self,
        candidate: TrackCandidate,
        center: tuple[float, float],
    ) -> BBox:
        candidate_bbox = self._candidate_bbox(candidate)
        width = max(1.0, bbox_area(candidate_bbox) ** 0.5)
        height = max(width, bbox_area(candidate_bbox) / width)
        cx, cy = center
        return (cx - (width / 2.0), cy - (height / 2.0), cx + (width / 2.0), cy + (height / 2.0))

    def _candidate_bbox(self, candidate: TrackCandidate) -> BBox:
        return candidate.bbox_xyxy

    def _center_distance(
        self,
        bbox: BBox,
        center: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> float:
        bx, by = bbox_center(bbox)
        return sqrt(((bx - center[0]) / max(1.0, frame_width)) ** 2 + ((by - center[1]) / max(1.0, frame_height)) ** 2)

    def _movement_vector(
        self,
        bbox: BBox,
        last_center: tuple[float, float],
    ) -> tuple[float, float]:
        bx, by = bbox_center(bbox)
        return (bx - last_center[0], by - last_center[1])

    def _direction_similarity(
        self,
        baseline: tuple[float, float],
        movement: tuple[float, float],
    ) -> float:
        base_x, base_y = baseline
        move_x, move_y = movement
        base_norm = sqrt((base_x * base_x) + (base_y * base_y))
        move_norm = sqrt((move_x * move_x) + (move_y * move_y))
        if base_norm <= 1e-6 or move_norm <= 1e-6:
            return 0.5
        cosine = ((base_x * move_x) + (base_y * move_y)) / (base_norm * move_norm)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
