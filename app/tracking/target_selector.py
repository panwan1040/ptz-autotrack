from __future__ import annotations

from dataclasses import dataclass

from app.config import TrackingSection
from app.models.runtime import TargetState, TrackStatus
from app.tracking.models import SelectionResult, TrackCandidate
from app.utils.geometry import bbox_area, normalized_bbox_center


@dataclass(slots=True)
class TargetSelector:
    """Scores visible tracks and applies sticky switching policy."""

    config: TrackingSection

    def select(
        self,
        candidates: list[TrackCandidate],
        previous: TargetState,
        frame_width: int,
        frame_height: int,
    ) -> SelectionResult:
        if not candidates:
            return SelectionResult(candidate=None, reason="no_visible_candidates")

        scored = {
            candidate.track_id: self._score_candidate(candidate, previous, frame_width, frame_height)
            for candidate in candidates
        }
        previous_candidate = next((candidate for candidate in candidates if candidate.track_id == previous.track_id), None)

        best_candidate = max(candidates, key=lambda candidate: scored[candidate.track_id])
        best_score = scored[best_candidate.track_id]

        if previous_candidate is None:
            confirmed_best = self._best_confirmed(candidates, scored)
            if confirmed_best is not None:
                return SelectionResult(
                    candidate=confirmed_best,
                    reason="select_confirmed_candidate",
                    score=scored[confirmed_best.track_id],
                )
            return SelectionResult(candidate=best_candidate, reason="candidate_warming_up", score=best_score)

        previous_score = scored[previous_candidate.track_id]
        if previous.status == TrackStatus.TRACKING:
            challenger = self._best_switch_candidate(candidates, scored, previous_candidate, previous_score)
            if challenger is not None:
                return SelectionResult(
                    candidate=challenger,
                    reason="switch_margin_exceeded",
                    score=scored[challenger.track_id],
                    previous_score=previous_score,
                    switched=True,
                )
            return SelectionResult(
                candidate=previous_candidate,
                reason="stick_with_current_target",
                score=previous_score,
                previous_score=previous_score,
            )

        if previous_candidate.confirmed:
            return SelectionResult(
                candidate=previous_candidate,
                reason="reacquired_previous_target",
                score=previous_score,
                previous_score=previous_score,
            )

        confirmed_best = self._best_confirmed(candidates, scored)
        if confirmed_best is not None:
            return SelectionResult(
                candidate=confirmed_best,
                reason="select_confirmed_candidate",
                score=scored[confirmed_best.track_id],
                previous_score=previous_score,
                switched=confirmed_best.track_id != previous_candidate.track_id,
            )

        return SelectionResult(
            candidate=best_candidate,
            reason="candidate_warming_up",
            score=best_score,
            previous_score=previous_score,
            switched=best_candidate.track_id != previous_candidate.track_id,
        )

    def _best_confirmed(
        self,
        candidates: list[TrackCandidate],
        scored: dict[int, float],
    ) -> TrackCandidate | None:
        confirmed = [candidate for candidate in candidates if candidate.confirmed]
        if not confirmed:
            return None
        return max(confirmed, key=lambda candidate: scored[candidate.track_id])

    def _best_switch_candidate(
        self,
        candidates: list[TrackCandidate],
        scored: dict[int, float],
        previous_candidate: TrackCandidate,
        previous_score: float,
    ) -> TrackCandidate | None:
        margin_target = previous_score * (1.0 + self.config.switch_margin_ratio)
        challengers = [
            candidate
            for candidate in candidates
            if candidate.track_id != previous_candidate.track_id
            and candidate.confirmed
            and scored[candidate.track_id] >= margin_target
        ]
        if not challengers:
            return None
        return max(challengers, key=lambda candidate: scored[candidate.track_id])

    def _score_candidate(
        self,
        candidate: TrackCandidate,
        previous: TargetState,
        frame_width: int,
        frame_height: int,
    ) -> float:
        nx, ny = normalized_bbox_center(candidate.bbox_xyxy, frame_width, frame_height)
        center_score = max(0.0, 1.0 - (abs(nx - 0.5) + abs(ny - 0.5)))
        frame_area = max(1.0, float(frame_width * frame_height))
        size_score = min(1.0, bbox_area(candidate.bbox_xyxy) / (frame_area * 0.18))
        stability_score = min(1.0, candidate.total_visible_frames / max(1, self.config.min_persist_frames))
        continuity_bonus = 0.18 if candidate.track_id == previous.track_id else 0.0
        recently_seen_bonus = 0.04 if candidate.missed_frames == 0 else 0.0
        strategy = self.config.strategy
        if strategy == "largest":
            return (
                (0.48 * size_score)
                + (0.20 * center_score)
                + (0.14 * candidate.confidence)
                + (0.18 * stability_score)
                + continuity_bonus
                + recently_seen_bonus
            )
        if strategy == "highest_confidence":
            return (
                (0.48 * candidate.confidence)
                + (0.20 * center_score)
                + (0.14 * size_score)
                + (0.18 * stability_score)
                + continuity_bonus
                + recently_seen_bonus
            )
        if strategy == "most_centered":
            return (
                (0.48 * center_score)
                + (0.20 * candidate.confidence)
                + (0.14 * size_score)
                + (0.18 * stability_score)
                + continuity_bonus
                + recently_seen_bonus
            )
        return (
            (0.38 * candidate.confidence)
            + (0.28 * center_score)
            + (0.14 * size_score)
            + (0.20 * stability_score)
            + continuity_bonus
            + recently_seen_bonus
        )
