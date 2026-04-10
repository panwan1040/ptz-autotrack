from app.config import ControlSection
from app.control.control_logic import ControlLogic
from app.models.runtime import ControlMode, PtzDirection, TargetState, TrackStatus


def make_target(bbox):
    return TargetState(
        track_id=1,
        bbox_xyxy=bbox,
        confidence=0.9,
        persist_frames=5,
        visible_frames=5,
        last_seen_ts=1.0,
        status=TrackStatus.TRACKING,
        stable=True,
        visible=True,
    )


def test_dead_zone_no_move() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((430, 170, 530, 370)), 960, 540)
    assert decision.move_direction is None
    assert decision.reason == "within_inner_dead_zone"
    assert decision.control_mode == ControlMode.HOLD_STABLE


def test_move_left() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((10, 170, 110, 370)), 960, 540)
    assert decision.move_direction == PtzDirection.LEFT
    assert decision.control_mode == ControlMode.COARSE_ALIGN
    assert decision.move_pulse_ms >= 40


def test_diagonal_right_down() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((820, 420, 920, 530)), 960, 540)
    assert decision.move_direction == PtzDirection.RIGHT_DOWN
    assert decision.control_mode == ControlMode.COARSE_ALIGN


def test_prioritize_stronger_axis_when_errors_are_unbalanced() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((850, 220, 950, 320)), 960, 540)
    assert decision.move_direction == PtzDirection.RIGHT
    assert decision.reason == "coarse_align_horizontal"


def test_fine_align_mode_uses_shorter_pulses() -> None:
    logic = ControlLogic(ControlSection())
    decision = logic.decide(make_target((640, 170, 740, 370)), 960, 540)
    assert decision.move_direction == PtzDirection.RIGHT
    assert decision.control_mode == ControlMode.FINE_ALIGN
    assert decision.move_pulse_ms < ControlSection().pan_pulse_ms_large


def test_prediction_can_shift_control_direction_toward_future_center() -> None:
    logic = ControlLogic(ControlSection())
    target = make_target((380, 170, 480, 370))
    target.predicted_center = (700.0, 270.0)
    target.prediction_confidence = 0.8
    decision = logic.decide(target, 960, 540)
    assert decision.prediction_used is True
    assert decision.move_direction == PtzDirection.RIGHT


def test_zoom_compensation_reduces_pulse_when_target_is_large() -> None:
    logic = ControlLogic(ControlSection())
    wide = logic.decide(make_target((50, 170, 150, 370)), 960, 540)
    tight = logic.decide(make_target((50, 40, 250, 500)), 960, 540)
    assert tight.zoom_compensation_scale < wide.zoom_compensation_scale


def test_stale_frame_blocks_movement() -> None:
    logic = ControlLogic(ControlSection())
    target = make_target((10, 220, 110, 420))
    target.frame_age_seconds = 0.8
    decision = logic.decide(target, 960, 540)
    assert decision.move_direction is None
    assert decision.reason == "stale_frame_blocked"
