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
from app.models.runtime import TargetState, TrackStatus, TrackingPhase
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
    def update(self, detections, frame_width, frame_height, now):  # pragma: no cover - unused in these tests
        raise NotImplementedError


class DummyPtz:
    def __init__(self) -> None:
        self.pulses: list[tuple[str, int]] = []
        self.home_moves = 0
        self.stop_calls = 0

    def pulse(self, direction, pulse_ms: int) -> PtzCommandResult:
        self.pulses.append((direction.value, pulse_ms))
        return PtzCommandResult(True, "pulse", direction.value, pulse_ms, True, detail="ok")

    def move_home(self) -> PtzCommandResult:
        self.home_moves += 1
        return PtzCommandResult(True, "preset", "home", 0, True, detail="ok")

    def stop(self, direction=None) -> PtzCommandResult:
        self.stop_calls += 1
        direction_name = direction.value if direction is not None else "Stop"
        return PtzCommandResult(True, "stop", direction_name, 0, True, detail="ok")

    def emergency_stop(self) -> None:
        pass

    def startup_preset(self) -> PtzCommandResult:
        return PtzCommandResult(True, "preset", "startup", 0, True, detail="ok")


def make_service(control: ControlSection) -> tuple[TrackingService, DummyPtz]:
    ptz = DummyPtz()
    config = AppConfig(
        app=AppSection(api=ApiConfig(enabled=False)),
        camera=CameraSection(host="1.2.3.4", username="admin", password=SecretStr("secret")),
        video=VideoSection(),
        tracking=TrackingSection(),
        control=control,
        ptz=PtzSection(),
        snapshots=SnapshotSection(on_target_acquired=False, on_target_lost=False),
    )
    service = TrackingService(config, DummyReader(), DummyDetector(), DummyTracker(), ptz)
    return service, ptz


def test_lost_zoom_out_behavior() -> None:
    service, ptz = make_service(ControlSection(lost_behavior="zoom_out", lost_zoom_out_enabled=True))
    previous = TargetState(
        track_id=3,
        bbox_xyxy=(100, 100, 220, 320),
        last_seen_ts=1.0,
        status=TrackStatus.TRACKING,
        stable=True,
    )
    current = TargetState(
        track_id=3,
        bbox_xyxy=(100, 100, 220, 320),
        last_seen_ts=1.0,
        status=TrackStatus.LOST,
        stable=True,
    )

    service._handle_tracking_state(np.zeros((10, 10, 3), dtype=np.uint8), previous, current, 1.2)

    assert ptz.stop_calls == 1
    assert ptz.pulses == [("ZoomWide", service._config.control.zoom_pulse_ms)]


def test_return_home_after_prolonged_loss() -> None:
    service, ptz = make_service(ControlSection(lost_behavior="return_home", return_home_timeout_seconds=2.0))
    service._loss_started_at = 1.0
    previous = TargetState(
        track_id=4,
        bbox_xyxy=(150, 100, 260, 340),
        last_seen_ts=1.0,
        status=TrackStatus.LOST,
        stable=True,
    )
    current = TargetState(track_id=None, bbox_xyxy=None, status=TrackStatus.SEARCHING)

    service._handle_tracking_state(np.zeros((10, 10, 3), dtype=np.uint8), previous, current, 3.5)

    assert ptz.home_moves == 1
    assert service._tracking_phase == TrackingPhase.RETURNING_HOME


def test_acquiring_phase_for_visible_unstable_target() -> None:
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
        1.0,
    )

    assert service._tracking_phase == TrackingPhase.ACQUIRING
    assert current.status == TrackStatus.SEARCHING
