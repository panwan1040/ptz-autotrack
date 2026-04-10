from app.config import ControlSection
from app.control.zoom_logic import ZoomController
from app.models.runtime import PtzDirection, TargetState, TrackStatus


def make_target(bbox):
    return TargetState(
        track_id=1,
        bbox_xyxy=bbox,
        confidence=0.9,
        persist_frames=5,
        centered_frames=5,
        last_seen_ts=1.0,
        status=TrackStatus.TRACKING,
        stable=True,
        visible=True,
    )


def test_zoom_in_when_target_small() -> None:
    controller = ZoomController(ControlSection())
    decision = controller.decide(make_target((100, 100, 200, 150)), 540, pan_tilt_active=False)
    assert decision.zoom_direction == PtzDirection.ZOOM_IN


def test_zoom_out_when_target_large() -> None:
    controller = ZoomController(ControlSection())
    decision = controller.decide(make_target((100, 0, 200, 400)), 540, pan_tilt_active=False)
    assert decision.zoom_direction == PtzDirection.ZOOM_OUT


def test_zoom_is_blocked_when_recenter_first_is_needed() -> None:
    controller = ZoomController(ControlSection())
    decision = controller.decide(
        make_target((0, 100, 100, 150)),
        540,
        pan_tilt_active=False,
        normalized_error_x=-0.42,
    )
    assert decision.zoom_direction is None
    assert decision.reason == "zoom_blocked_recenter_first"


def test_zoom_is_blocked_until_alignment_is_stable() -> None:
    controller = ZoomController(ControlSection())
    target = make_target((100, 100, 200, 150))
    target.centered_frames = 0
    decision = controller.decide(target, 540, pan_tilt_active=False)
    assert decision.zoom_direction is None
    assert decision.reason == "zoom_blocked_align_first"
