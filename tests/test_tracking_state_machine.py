from app.config import TrackingSection
from app.models.runtime import TargetState, TrackStatus, TrackingPhase, compatibility_status_for_phase
from app.tracking.models import SelectionResult, TrackCandidate
from app.tracking.state_machine import TrackingStateMachine


def make_candidate(track_id: int, confirmed: bool, hits: int = 3) -> TrackCandidate:
    return TrackCandidate(
        track_id=track_id,
        bbox_xyxy=(100, 100, 200, 300),
        confidence=0.9,
        persist_frames=hits,
        total_visible_frames=hits,
        age_frames=hits,
        missed_frames=0,
        last_seen_ts=1.0,
        confirmed=confirmed,
    )


def test_state_machine_requires_confirmation_before_tracking() -> None:
    machine = TrackingStateMachine(TrackingSection(min_persist_frames=3, lost_timeout_seconds=1.0))
    state = machine.update(
        SelectionResult(candidate=make_candidate(1, confirmed=False, hits=1), reason="candidate_warming_up", score=0.5),
        TargetState(track_id=None, bbox_xyxy=None),
        now=1.0,
    )
    assert state.status == TrackStatus.SEARCHING
    assert state.stable is False
    assert state.visible is True


def test_state_machine_enters_lost_then_searching_after_timeout() -> None:
    machine = TrackingStateMachine(TrackingSection(min_persist_frames=1, lost_timeout_seconds=0.5))
    previous = TargetState(
        track_id=7,
        bbox_xyxy=(100, 100, 200, 300),
        last_seen_ts=1.0,
        status=TrackStatus.TRACKING,
        stable=True,
    )
    lost = machine.update(SelectionResult(candidate=None, reason="no_visible_candidates"), previous, now=1.3)
    searching = machine.update(SelectionResult(candidate=None, reason="no_visible_candidates"), lost, now=1.8)

    assert lost.status == TrackStatus.LOST
    assert searching.status == TrackStatus.SEARCHING


def test_state_machine_requires_stricter_confirmation_after_loss() -> None:
    machine = TrackingStateMachine(TrackingSection(min_persist_frames=1, recovery={"post_occlusion_confirm_frames": 2}))
    previous = TargetState(
        track_id=7,
        bbox_xyxy=(100, 100, 200, 300),
        last_seen_ts=1.0,
        status=TrackStatus.LOST,
        stable=True,
        missing_frames=1,
    )
    state = machine.update(
        SelectionResult(candidate=make_candidate(7, confirmed=True, hits=1), reason="reacquired_previous_target", score=0.8),
        previous,
        now=1.3,
    )

    assert state.status == TrackStatus.SEARCHING
    assert state.stable is False


def test_phase_compatibility_mapping() -> None:
    assert compatibility_status_for_phase(TrackingPhase.CANDIDATE_LOCK) == TrackStatus.SEARCHING
    assert compatibility_status_for_phase(TrackingPhase.MONITORING) == TrackStatus.TRACKING
    assert compatibility_status_for_phase(TrackingPhase.RETURNING_HOME) == TrackStatus.LOST
