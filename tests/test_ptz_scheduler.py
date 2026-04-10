from dataclasses import dataclass

from app.control.control_intent import PtzIntent, PtzIntentKind
from app.control.ptz_client import PtzCommandResult
from app.control.ptz_scheduler import PtzScheduler
from app.models.runtime import ControlMode, PtzDirection


@dataclass
class FakePtzClient:
    start_calls: list[str]
    stop_calls: list[str]
    fail_stop: bool = False

    def start(self, direction: PtzDirection) -> PtzCommandResult:
        self.start_calls.append(direction.value)
        return PtzCommandResult(True, "start", direction.value, 0, False, issued=True, accepted=True, detail="ok")

    def stop(self, direction: PtzDirection | None = None) -> PtzCommandResult:
        direction_name = direction.value if direction is not None else "Stop"
        self.stop_calls.append(direction_name)
        if self.fail_stop:
            return PtzCommandResult(False, "stop", direction_name, 0, False, issued=True, accepted=False, detail="failed")
        return PtzCommandResult(True, "stop", direction_name, 0, False, issued=True, accepted=True, detail="ok")


def make_intent(direction: PtzDirection, pulse_ms: int = 120) -> PtzIntent:
    return PtzIntent(
        kind=PtzIntentKind.MOVE if direction not in {PtzDirection.ZOOM_IN, PtzDirection.ZOOM_OUT} else PtzIntentKind.ZOOM,
        direction=direction,
        pulse_ms=pulse_ms,
        reason="test",
        control_mode=ControlMode.COARSE_ALIGN,
    )


def test_scheduler_starts_and_stops_on_due_tick() -> None:
    scheduler = PtzScheduler(FakePtzClient([], []))

    start_result = scheduler.submit(make_intent(PtzDirection.LEFT, 100), 1.0)
    stop_results = scheduler.tick(1.11)

    assert start_result.succeeded is True
    assert scheduler.state.pulse_active is False
    assert len(stop_results) == 1
    assert stop_results[0].direction == "Left"


def test_scheduler_interrupts_active_pulse_before_new_direction() -> None:
    client = FakePtzClient([], [])
    scheduler = PtzScheduler(client)

    scheduler.submit(make_intent(PtzDirection.LEFT, 200), 1.0)
    result = scheduler.submit(make_intent(PtzDirection.RIGHT, 150), 1.05)

    assert result.interrupted is True
    assert client.stop_calls == ["Left"]
    assert client.start_calls == ["Left", "Right"]


def test_scheduler_reports_partial_failure_when_stop_fails() -> None:
    client = FakePtzClient([], [], fail_stop=True)
    scheduler = PtzScheduler(client)

    scheduler.submit(make_intent(PtzDirection.LEFT, 100), 1.0)
    result = scheduler.force_stop(1.2, reason="test_stop")

    assert result.partial_failure is True
    assert scheduler.state.pulse_active is True


def test_scheduler_extends_same_direction_without_new_http_start() -> None:
    client = FakePtzClient([], [])
    scheduler = PtzScheduler(client)

    scheduler.submit(make_intent(PtzDirection.LEFT, 100), 1.0)
    result = scheduler.submit(make_intent(PtzDirection.LEFT, 160), 1.05)

    assert result.accepted is True
    assert result.issued is False
    assert client.start_calls == ["Left"]
