from __future__ import annotations

from dataclasses import dataclass, field

from app.config import TrackingSection
from app.models.runtime import TargetMemory, TargetState, TrackStatus
from app.tracking.appearance_extractor import AppearanceExtractor
from app.tracking.models import SelectionResult, TrackCandidate
from app.tracking.target_matcher import TargetMatcher
from app.utils.geometry import bbox_area, normalized_bbox_center


@dataclass(slots=True)
class TargetSelector:
    """Scores visible tracks and applies sticky switching policy."""

    config: TrackingSection
    _appearance_extractor: AppearanceExtractor = field(init=False)
    _matcher: TargetMatcher = field(init=False)

    def __post_init__(self) -> None:
        self._appearance_extractor = AppearanceExtractor(self.config.appearance)
        self._matcher = TargetMatcher(self.config, self._appearance_extractor)

    def select(
        self,
        candidates: list[TrackCandidate],
        previous: TargetState,
        memory: TargetMemory | None,
        frame_width: int,
        frame_height: int,
    ) -> SelectionResult:
        if not candidates:
            return SelectionResult(candidate=None, reason="no_visible_candidates")

        scored: dict[int, float] = {}
        for candidate in candidates:
            score, breakdown = self._matcher.score(candidate, memory, frame_width, frame_height)
            strategy_bonus = self._strategy_bonus(candidate, previous, frame_width, frame_height)
            scored[candidate.track_id] = score + strategy_bonus
            candidate.match_breakdown = breakdown
        previous_candidate = next((candidate for candidate in candidates if candidate.track_id == previous.track_id), None)

        best_candidate = max(candidates, key=lambda candidate: scored[candidate.track_id])
        best_score = scored[best_candidate.track_id]

        if previous_candidate is None and memory is not None and memory.track_id is not None:
            plausible_reacquisition = self._best_reacquisition_candidate(candidates, scored, memory)
            if plausible_reacquisition is None:
                return SelectionResult(candidate=None, reason="holding_memory_during_short_loss")
            return SelectionResult(
                candidate=plausible_reacquisition,
                reason="reacquired_from_memory",
                score=scored[plausible_reacquisition.track_id],
                previous_score=memory.last_match_score,
                switched=plausible_reacquisition.track_id != memory.track_id,
            )

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
            challenger = self._best_switch_candidate(candidates, scored, previous_candidate, previous_score, memory)
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
        memory: TargetMemory | None,
    ) -> TrackCandidate | None:
        margin_target = previous_score * (1.0 + self.config.switch_margin_ratio)
        replacement_margin = self.config.recovery.replacement_score_margin
        challengers = [
            candidate
            for candidate in candidates
            if candidate.track_id != previous_candidate.track_id
            and candidate.confirmed
            and scored[candidate.track_id] >= max(margin_target, previous_score + replacement_margin)
        ]
        if not self.config.recovery.allow_target_replacement and memory is not None and memory.consecutive_missing_frames <= 0:
            return None
        if not challengers:
            return None
        return max(challengers, key=lambda candidate: scored[candidate.track_id])

    def _best_reacquisition_candidate(
        self,
        candidates: list[TrackCandidate],
        scored: dict[int, float],
        memory: TargetMemory,
    ) -> TrackCandidate | None:
        required_similarity = self.config.appearance.min_similarity if self.config.appearance.enabled else 0.0
        recovery_candidates = [
            candidate
            for candidate in candidates
            if candidate.confirmed
            and scored[candidate.track_id] >= max(0.45, memory.last_match_score * 0.75)
            and (
                not self.config.appearance.enabled
                or (candidate.match_breakdown or {}).get("appearance", 0.0) >= required_similarity
            )
        ]
        if not recovery_candidates:
            return None
        return max(recovery_candidates, key=lambda candidate: scored[candidate.track_id])

    def _strategy_bonus(
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
        continuity_bonus = 0.18 if candidate.track_id == previous.track_id else 0.0
        recently_seen_bonus = 0.04 if candidate.missed_frames == 0 else 0.0
        strategy = self.config.strategy
        if strategy == "largest":
            return (0.20 * size_score) + (0.04 * center_score) + continuity_bonus + recently_seen_bonus
        if strategy == "highest_confidence":
            return (0.20 * candidate.confidence) + (0.04 * center_score) + continuity_bonus + recently_seen_bonus
        if strategy == "most_centered":
            return (0.20 * center_score) + (0.04 * candidate.confidence) + continuity_bonus + recently_seen_bonus
        return (0.10 * center_score) + (0.08 * candidate.confidence) + continuity_bonus + recently_seen_bonus
