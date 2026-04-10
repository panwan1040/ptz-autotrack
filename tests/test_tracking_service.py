from dataclasses import dataclass, field

import numpy as np
from pydantic import SecretStr

from app.config import (
    ApiConfig,
    AppConfig,
    AppSection,
    CameraSection,
    ControlSection,
    PtzSection,
    SnapshotSection,
    TrackingSection,
    VideoSection,
)
from app.control.ptz_client import PtzCommandResult
from app.models.runtime import ControlDecision, TargetMemory, TargetState, TrackStatus, TrackingPhase
from app.services.tracking_service import TrackingService


@dataclass
class DummyReaderStats:
    reconnect_count: int = 0


@dataclass
class DummyReader:
    stats: DummyReaderStats = field(default_factory=DummyReaderStats)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class DummyDetector:
    def detect(self, frame):  # pragma: no cover - unused in these focused unit tests
        return [], 0.0


class DummyTracker:
    def __init__(self) -> None:
        self.target_memory = TargetMemory()

    def update(self, detections, frame_width, frame_height, now):  # pragma: no cover - unused in these tests
        raise NotImplementedError


class DummyPtz:
    def __init__(self) -> None:
        self.starts: list[str] = []
        self.home_moves = 0
        self.stop_calls = 0
        self.active_direction = None

    def pulse(self, direction, pulse_ms: int) -> PtzCommandResult:
        self.starts.append(direction.value)
        self.active_direction = direction
        return PtzCommandResult(True, "pulse", direction.value, pulse_ms, True, detail="ok", issued=True, accepted=True)

    def start(self, direction) -> PtzCommandResult:
        self.starts.append(direction.value)
        self.active_direction = direction
        return PtzCommandResult(True, "start", direction.value, 0, False, detail="ok", issued=True, accepted=True)

    def move_home(self) -> PtzCommandResult:
        self.home_moves += 1
        return PtzCommandResult(True, "preset", "home", 0, True, detail="ok")

    def stop(self, direction=None) -> PtzCommandResult:
        self.stop_calls += 1
        direction_name = direction.value if direction is not None else "Stop"
        if direction is not None and self.active_direction == direction:
            self.active_direction = None
        return PtzCommandResult(True, "stop", direction_name, 0, False, detail="ok", issued=True, accepted=True)

    def emergency_stop(self) -> None:
        pass

    def startup_preset(self) -> PtzCommandResult:
        return PtzCommandResult(True, "preset", "startup", 0, True, detail="ok")


def make_service(control: ControlSection) -> tuple[TrackingService, DummyPtz]:
    ptz = DummyPtz()
    config = AppConfig(
        app=AppSection(api=ApiConfig(enabled=False)),
        camera=CameraSection(host="1.2.3.4", username="admin", password=SecretStr("secret"), home_preset_name="Home"),
        video=VideoSection(),
        tracking=TrackingSection(),
        control=control,
        ptz=PtzSection(),
        snapshots=SnapshotSection(on_target_acquired=False, on_target_lost=False),
    )
    service = TrackingService(config, DummyReader(), DummyDetector(), DummyTracker(), ptz)
    return service, ptz


def test_recovery_wide_zoom_out_behavior() -> None:
    service, ptz = make_service(ControlSection(lost_behavior="zoom_out", lost_zoom_out_enabled=True))
    service._tracker.target_memory.track_id = 3
    service._tracker.target_memory.last_confirmed_ts = 1.0
    service._tracker.target_memory.missing_started_ts = 1.0
    service._tracker.target_memory.last_zoom_ratio = 0.5
    service._tracker.target_memory.consecutive_missing_frames = 8
    previous = TargetState(
        track_id=3,
        bbox_xyxy=(100, 100, 220, 320),
        last_seen_ts=1.0,
        status=TrackStatus.TRACKING,
        stable=True,
    )
    current = TargetState(
        track_id=3,
        bbox_xyxy=None,
        last_seen_ts=1.0,
        status=TrackStatus.LOST,
        stable=True,
        visible=False,
    )

    service._handle_tracking_state(
        np.zeros((10, 10, 3), dtype=np.uint8),
        previous,
        current,
        ControlDecision(reason="idle"),
        6.0,
        10,
        10,
    )

    assert ptz.stop_calls == 0
    assert ptz.starts == ["ZoomWide"]
    assert service._tracking_phase == TrackingPhase.RECOVERY_WIDE


def test_return_home_after_prolonged_loss() -> None:
    service, ptz = make_service(ControlSection(lost_behavior="return_home", return_home_timeout_seconds=2.0))
    service._loss_started_at = 1.0
    service._tracker.target_memory.track_id = 4
    service._tracker.target_memory.last_confirmed_ts = 1.0
    service._tracker.target_memory.missing_started_ts = 1.0
    service._tracker.target_memory.consecutive_missing_frames = 10
    previous = TargetState(
        track_id=4,
        bbox_xyxy=(150, 100, 260, 340),
        last_seen_ts=1.0,
        status=TrackStatus.LOST,
        stable=True,
    )
    current = TargetState(track_id=None, bbox_xyxy=None, status=TrackStatus.SEARCHING)

    service._handle_tracking_state(
        np.zeros((10, 10, 3), dtype=np.uint8),
        previous,
        current,
        ControlDecision(reason="idle"),
        8.5,
        10,
        10,
    )

    assert ptz.home_moves == 1
    assert service._tracking_phase == TrackingPhase.RETURNING_HOME


def test_candidate_lock_phase_for_visible_unstable_target() -> None:
    service, _ptz = make_service(ControlSection())
    current = TargetState(
        track_id=8,
        bbox_xyxy=(100, 100, 200, 300),
        status=TrackStatus.SEARCHING,
        stable=False,
        visible=True,
    )

    service._handle_tracking_state(
        np.zeros((10, 10, 3), dtype=np.uint8),
        TargetState(track_id=None, bbox_xyxy=None),
        current,
        ControlDecision(reason="idle"),
        1.0,
        960,
        540,
    )

    assert service._tracking_phase == TrackingPhase.CANDIDATE_LOCK
    assert current.status == TrackStatus.SEARCHING


def test_handoff_phase_when_target_is_centered_and_large_enough() -> None:
    service, ptz = make_service(ControlSection())
    service._tracker.target_memory.centered_frames = service._config.tracking.handoff.stable_center_frames - 1
    service._tracker.target_memory.recent_confidences = [0.8, 0.82, 0.84]
    service._tracker.target_memory.confidence_average = 0.82
    current = TargetState(
        track_id=8,
        bbox_xyxy=(420, 140, 540, 360),
        status=TrackStatus.TRACKING,
        stable=True,
        visible=True,
        persist_frames=service._config.tracking.handoff.min_persist_frames,
    )

    service._handle_tracking_state(
        np.zeros((540, 960, 3), dtype=np.uint8),
        TargetState(track_id=8, bbox_xyxy=(418, 138, 538, 358), status=TrackStatus.TRACKING, stable=True, visible=True),
        current,
        service._control_logic.decide(current, 960, 540),
        1.0,
        960,
        540,
    )

    assert service._tracking_phase == TrackingPhase.HANDOFF
    assert current.handoff_ready is True
    assert current.centered_frames >= service._config.tracking.handoff.stable_center_frames


def test_stale_frame_blocks_local_recovery_motion() -> None:
    service, ptz = make_service(ControlSection())
    service._tracking_phase = TrackingPhase.RECOVERY_LOCAL
    service._tracker.target_memory.track_id = 5
    service._tracker.target_memory.predicted_window = (300, 120, 420, 320)
    current = TargetState(
        track_id=5,
        bbox_xyxy=None,
        status=TrackStatus.LOST,
        stable=True,
        visible=False,
        frame_age_seconds=service._config.tracking.stale_frame.aggressive_recovery_max_age_seconds + 0.1,
        predicted_window=(300, 120, 420, 320),
    )

    service._apply_phase_behavior(np.zeros((540, 960, 3), dtype=np.uint8), current, ControlDecision(reason="idle"), 2.0, 960, 540)

    assert ptz.starts == []
    assert service._last_skip_reason == "stale_frame_blocks_local_recovery"
