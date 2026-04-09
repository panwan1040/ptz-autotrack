from app.config import TrackingSection
from app.models.runtime import TargetMemory, TargetState, TrackStatus
from app.tracking.models import TrackCandidate
from app.tracking.target_selector import TargetSelector


def make_candidate(
    track_id: int,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.8,
    hits: int = 4,
    confirmed: bool = True,
) -> TrackCandidate:
    return TrackCandidate(
        track_id=track_id,
        bbox_xyxy=bbox,
        confidence=confidence,
        persist_frames=hits,
        total_visible_frames=hits,
        age_frames=hits,
        missed_frames=0,
        last_seen_ts=1.0,
        confirmed=confirmed,
    )


def test_select_confirmed_candidate() -> None:
    selector = TargetSelector(TrackingSection(strategy="largest"))
    result = selector.select(
        [
            make_candidate(1, (0, 0, 50, 100), confidence=0.7),
            make_candidate(2, (450, 80, 620, 420), confidence=0.6),
        ],
        TargetState(track_id=None, bbox_xyxy=None),
        None,
        960,
        540,
    )
    assert result.candidate is not None
    assert result.candidate.track_id == 2


def test_keep_previous_target_without_switch_margin() -> None:
    selector = TargetSelector(TrackingSection(strategy="stick_nearest", switch_margin_ratio=0.25))
    previous = TargetState(
        track_id=10,
        bbox_xyxy=(380, 90, 560, 430),
        status=TrackStatus.TRACKING,
        stable=True,
    )
    result = selector.select(
        [
            make_candidate(10, (390, 95, 570, 425), confidence=0.84, hits=8),
            make_candidate(11, (430, 90, 640, 430), confidence=0.88, hits=8),
        ],
        previous,
        TargetMemory(track_id=10),
        960,
        540,
    )
    assert result.candidate is not None
    assert result.candidate.track_id == 10
    assert result.reason == "stick_with_current_target"


def test_switch_when_confirmed_candidate_clearly_better_and_replacement_allowed() -> None:
    selector = TargetSelector(
        TrackingSection(
            strategy="stick_nearest",
            switch_margin_ratio=0.10,
            recovery={"allow_target_replacement": True, "replacement_score_margin": 0.10},
        )
    )
    previous = TargetState(
        track_id=10,
        bbox_xyxy=(100, 100, 220, 320),
        status=TrackStatus.TRACKING,
        stable=True,
    )
    result = selector.select(
        [
            make_candidate(10, (110, 110, 220, 300), confidence=0.65, hits=7),
            make_candidate(11, (360, 70, 620, 470), confidence=0.95, hits=9),
        ],
        previous,
        TargetMemory(track_id=10, consecutive_missing_frames=1),
        960,
        540,
    )
    assert result.candidate is not None
    assert result.candidate.track_id == 11
    assert result.switched is True
    assert result.reason == "switch_margin_exceeded"


def test_do_not_switch_to_unconfirmed_candidate() -> None:
    selector = TargetSelector(TrackingSection(strategy="stick_nearest", min_persist_frames=3))
    previous = TargetState(
        track_id=10,
        bbox_xyxy=(360, 90, 560, 430),
        status=TrackStatus.TRACKING,
        stable=True,
    )
    result = selector.select(
        [
            make_candidate(10, (370, 95, 565, 425), confidence=0.8, hits=10),
            make_candidate(11, (400, 80, 690, 470), confidence=0.97, hits=1, confirmed=False),
        ],
        previous,
        TargetMemory(track_id=10),
        960,
        540,
    )
    assert result.candidate is not None
    assert result.candidate.track_id == 10


def test_hold_memory_during_short_loss_without_plausible_reacquisition() -> None:
    selector = TargetSelector(TrackingSection())
    previous = TargetState(
        track_id=10,
        bbox_xyxy=(360, 90, 560, 430),
        status=TrackStatus.LOST,
        stable=True,
        visible=False,
    )
    memory = TargetMemory(
        track_id=10,
        last_match_score=0.9,
        consecutive_missing_frames=2,
        appearance_signature=[1.0, 0.0, 0.0],
    )
    cold_candidate = make_candidate(11, (50, 50, 120, 180), confidence=0.4, hits=3)
    cold_candidate.appearance_signature = [0.0, 1.0, 0.0]

    result = selector.select([cold_candidate], previous, memory, 960, 540)

    assert result.candidate is None
    assert result.reason == "holding_memory_during_short_loss"
