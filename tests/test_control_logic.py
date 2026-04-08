from app.config import ControlSection
from app.control.control_logic import ControlLogic
from app.models.runtime import PtzDirection, TargetState, TrackStatus


def make_target(bbox):
    return TargetState(track_id=1, bbox_xyxy=bbox, confidence=0.9, persist_frames=5, last_seen_ts=1.0, status=TrackStatus.TRACKING)


def test_dead_zone_no_move() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((430, 220, 530, 420)), 960, 540)
    assert decision.move_direction is None
    assert decision.reason == "within_dead_zone"


def test_move_left() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((10, 220, 110, 420)), 960, 540)
    assert decision.move_direction == PtzDirection.LEFT


def test_diagonal_right_down() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((820, 420, 920, 530)), 960, 540)
    assert decision.move_direction == PtzDirection.RIGHT_DOWN
