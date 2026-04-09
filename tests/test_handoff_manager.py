from app.config import TrackingSection
from app.control.handoff_manager import HandoffManager
from app.control.monitoring_policy import MonitoringPolicy
from app.models.runtime import TargetMemory, TargetState


def test_handoff_ready_requires_center_size_and_persistence() -> None:
    config = TrackingSection()
    manager = HandoffManager(config.handoff)
    memory = TargetMemory(centered_frames=config.handoff.stable_center_frames - 1)
    target = TargetState(
        track_id=1,
        bbox_xyxy=(420, 140, 540, 360),
        visible=True,
        stable=True,
        persist_frames=config.handoff.min_persist_frames,
    )

    ready, reason = manager.evaluate(target, memory, 960, 540)

    assert ready is True
    assert reason == "handoff_ready"


def test_monitoring_breaks_when_target_drift_is_large() -> None:
    policy = MonitoringPolicy(TrackingSection().monitoring)
    memory = TargetMemory(track_id=1, last_confirmed_ts=1.0, handoff_ts=1.0)
    target = TargetState(track_id=1, bbox_xyxy=(700, 60, 840, 280), visible=True, stable=True)

    broken, reason = policy.should_resume_control(target, memory, 960, 540, now=1.5)

    assert broken is True
    assert reason == "handoff_broken_large_center_error"
