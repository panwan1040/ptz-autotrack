from app.config import TrackingSection
from app.control.lifecycle_manager import LifecycleManager
from app.models.runtime import TargetMemory, TargetState, TrackingPhase


def test_lifecycle_progresses_from_candidate_to_temp_lost_to_recovery() -> None:
    config = TrackingSection(
        recovery={
            "short_loss_timeout_seconds": 1.0,
            "occlusion_timeout_seconds": 2.0,
            "recovery_zoom_out_start_timeout_seconds": 1.3,
            "recovery_local_timeout_seconds": 3.0,
            "recovery_wide_timeout_seconds": 5.0,
            "tight_zoom_height_ratio_threshold": 0.40,
        }
    )
    manager = LifecycleManager(config)
    memory = TargetMemory(track_id=7, missing_started_ts=10.0, last_zoom_ratio=0.5, tight_zoom_detected=True)
    unstable = TargetState(track_id=7, bbox_xyxy=(100, 100, 200, 300), stable=False, visible=True)
    missing = TargetState(track_id=7, bbox_xyxy=None, stable=True, visible=False)

    assert manager.next_phase(TrackingPhase.SEARCHING, unstable, memory, handoff_ready=False, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=1, tight_zoom_detected=False, return_home_issued=False, now=10.1) == TrackingPhase.CANDIDATE_LOCK
    memory.consecutive_missing_frames = 1
    assert manager.next_phase(TrackingPhase.CENTERING, missing, memory, handoff_ready=False, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=0, tight_zoom_detected=True, return_home_issued=False, now=10.5) == TrackingPhase.TEMP_LOST
    memory.consecutive_missing_frames = 4
    assert manager.next_phase(TrackingPhase.TEMP_LOST, missing, memory, handoff_ready=False, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=0, tight_zoom_detected=True, return_home_issued=False, now=11.0) == TrackingPhase.OCCLUDED
    memory.consecutive_missing_frames = 6
    assert manager.next_phase(TrackingPhase.OCCLUDED, missing, memory, handoff_ready=False, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=0, tight_zoom_detected=True, return_home_issued=False, now=11.5) == TrackingPhase.RECOVERY_ZOOM_OUT
    memory.recovery_zoom_steps = 3
    assert manager.next_phase(TrackingPhase.RECOVERY_ZOOM_OUT, missing, memory, handoff_ready=False, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=0, tight_zoom_detected=True, return_home_issued=False, now=12.5) == TrackingPhase.RECOVERY_LOCAL
    memory.consecutive_missing_frames = 8
    assert manager.next_phase(TrackingPhase.RECOVERY_LOCAL, missing, memory, handoff_ready=False, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=0, tight_zoom_detected=False, return_home_issued=False, now=14.5) == TrackingPhase.RECOVERY_WIDE
    memory.consecutive_missing_frames = 10
    assert manager.next_phase(TrackingPhase.RECOVERY_WIDE, missing, memory, handoff_ready=False, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=0, tight_zoom_detected=False, return_home_issued=False, now=16.0) == TrackingPhase.LOST


def test_lifecycle_enters_handoff_and_monitoring() -> None:
    manager = LifecycleManager(TrackingSection())
    memory = TargetMemory(track_id=3)
    visible = TargetState(track_id=3, bbox_xyxy=(400, 80, 520, 320), stable=True, visible=True, handoff_ready=True)

    assert manager.next_phase(TrackingPhase.CENTERING, visible, memory, handoff_ready=True, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=1, tight_zoom_detected=False, return_home_issued=False, now=1.0) == TrackingPhase.HANDOFF
    memory.handoff_ts = 1.0
    assert manager.next_phase(TrackingPhase.HANDOFF, visible, memory, handoff_ready=True, handoff_zoom_candidate=False, monitoring_broken=False, visible_candidate_count=1, tight_zoom_detected=False, return_home_issued=False, now=1.2) == TrackingPhase.MONITORING


def test_lifecycle_uses_zooming_for_handoff_when_centered_but_still_small() -> None:
    manager = LifecycleManager(TrackingSection())
    memory = TargetMemory(track_id=3)
    visible = TargetState(track_id=3, bbox_xyxy=(430, 150, 520, 270), stable=True, visible=True)

    phase = manager.next_phase(
        TrackingPhase.CENTERING,
        visible,
        memory,
        handoff_ready=False,
        handoff_zoom_candidate=True,
        monitoring_broken=False,
        visible_candidate_count=1,
        tight_zoom_detected=False,
        return_home_issued=False,
        now=1.0,
    )

    assert phase == TrackingPhase.ZOOMING_FOR_HANDOFF


def test_phase_policy_disables_pan_tilt_in_monitoring() -> None:
    manager = LifecycleManager(TrackingSection())
    policy = manager.policy_for(TrackingPhase.MONITORING)
    assert policy.pan_tilt_allowed is False
    assert policy.zoom_allowed is False


def test_lifecycle_avoids_passive_occlusion_wait_when_zoom_is_tight_and_no_candidates_exist() -> None:
    manager = LifecycleManager(TrackingSection(recovery={"recovery_zoom_out_start_timeout_seconds": 1.2}))
    memory = TargetMemory(track_id=4, missing_started_ts=10.0, last_zoom_ratio=0.5)
    memory.consecutive_missing_frames = 4
    missing = TargetState(track_id=4, bbox_xyxy=None, stable=True, visible=False)

    phase = manager.next_phase(
        TrackingPhase.OCCLUDED,
        missing,
        memory,
        handoff_ready=False,
        handoff_zoom_candidate=False,
        monitoring_broken=False,
        visible_candidate_count=0,
        tight_zoom_detected=True,
        return_home_issued=False,
        now=11.4,
    )

    assert phase == TrackingPhase.RECOVERY_ZOOM_OUT
