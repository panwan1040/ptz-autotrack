from __future__ import annotations

from dataclasses import dataclass, field

from app.control.control_intent import PtzIntent, PtzIntentKind
from app.control.ptz_client import DahuaPtzClient
from app.control.ptz_runtime_state import PtzRuntimeState, PtzScheduleResult
from app.logging_config import get_logger
from app.models.runtime import PtzDirection

logger = get_logger(__name__)


@dataclass(slots=True)
class PtzScheduler:
    """Non-blocking pulse scheduler built on explicit start/stop ownership."""

    client: DahuaPtzClient
    _state: PtzRuntimeState = field(init=False, default_factory=PtzRuntimeState)

    @property
    def state(self) -> PtzRuntimeState:
        return self._state

    def seconds_until_due_stop(self, now: float) -> float | None:
        if not self._state.pulse_active or self._state.pulse_due_stop_ts is None:
            return None
        return max(0.0, self._state.pulse_due_stop_ts - now)

    def tick(self, now: float) -> list[PtzScheduleResult]:
        results: list[PtzScheduleResult] = []
        if not self._state.pulse_active or self._state.pulse_due_stop_ts is None:
            return results
        if now < self._state.pulse_due_stop_ts:
            return results
        active_direction = self._active_direction()
        if active_direction is None:
            self._clear_active()
            return results
        stop_result = self.client.stop(active_direction)
        result = PtzScheduleResult(
            kind="stop",
            direction=active_direction.value,
            attempted=True,
            issued=stop_result.issued,
            accepted=stop_result.success,
            succeeded=stop_result.success and stop_result.issued,
            partial_failure=not stop_result.success,
            detail=stop_result.detail or "scheduled_stop",
        )
        if stop_result.success:
            self._clear_active()
        else:
            self._state.stop_pending = True
            self._state.pulse_due_stop_ts = now + 0.05
        self._state.last_command_outcome = self._result_dict(result)
        results.append(result)
        return results

    def submit(self, intent: PtzIntent, now: float) -> PtzScheduleResult:
        if intent.kind == PtzIntentKind.NOOP or intent.direction is None or intent.pulse_ms <= 0:
            result = PtzScheduleResult(
                kind=intent.kind.value,
                direction=intent.direction.value if intent.direction else None,
                skipped=True,
                detail="noop_intent",
            )
            self._state.last_command_outcome = self._result_dict(result)
            return result

        if intent.kind == PtzIntentKind.STOP:
            return self.force_stop(now, reason=intent.reason)

        current_direction = self._active_direction()
        if current_direction is not None and current_direction == intent.direction and self._state.pulse_active:
            self._state.pulse_due_stop_ts = max(self._state.pulse_due_stop_ts or now, now + (intent.pulse_ms / 1000.0))
            result = PtzScheduleResult(
                kind=intent.kind.value,
                direction=intent.direction.value,
                attempted=True,
                accepted=True,
                succeeded=False,
                detail="extended_active_pulse",
            )
            self._state.last_command_outcome = self._result_dict(result)
            return result

        interrupted = False
        if current_direction is not None and current_direction != intent.direction:
            if not intent.allow_interrupt:
                result = PtzScheduleResult(
                    kind=intent.kind.value,
                    direction=intent.direction.value,
                    attempted=True,
                    skipped=True,
                    stop_deferred=True,
                    detail="active_pulse_not_interruptible",
                )
                self._state.last_command_outcome = self._result_dict(result)
                return result
            stop_result = self.client.stop(current_direction)
            interrupted = stop_result.success
            if not stop_result.success:
                result = PtzScheduleResult(
                    kind=intent.kind.value,
                    direction=intent.direction.value,
                    attempted=True,
                    partial_failure=True,
                    interrupted=True,
                    detail="failed_to_interrupt_previous_pulse",
                )
                self._state.last_command_outcome = self._result_dict(result)
                return result
            self._clear_active()

        start_result = self.client.start(intent.direction)
        result = PtzScheduleResult(
            kind=intent.kind.value,
            direction=intent.direction.value,
            attempted=True,
            issued=start_result.issued,
            accepted=start_result.success,
            succeeded=start_result.success and start_result.issued,
            partial_failure=interrupted and not start_result.success,
            interrupted=interrupted,
            detail=start_result.detail or "scheduled_start",
            cooldown_eligible=start_result.success and start_result.issued,
        )
        if start_result.success:
            self._state.active_direction = intent.direction.value
            self._state.active_kind = intent.kind.value
            self._state.pulse_active = True
            self._state.pulse_started_ts = now
            self._state.pulse_due_stop_ts = now + (intent.pulse_ms / 1000.0)
            self._state.stop_pending = True
        self._state.last_command_outcome = self._result_dict(result)
        return result

    def force_stop(self, now: float, reason: str = "force_stop") -> PtzScheduleResult:
        active_direction = self._active_direction()
        if active_direction is None:
            result = PtzScheduleResult(kind="stop", direction=None, skipped=True, detail="no_active_pulse")
            self._state.last_command_outcome = self._result_dict(result)
            return result
        stop_result = self.client.stop(active_direction)
        result = PtzScheduleResult(
            kind="stop",
            direction=active_direction.value,
            attempted=True,
            issued=stop_result.issued,
            accepted=stop_result.success,
            succeeded=stop_result.success and stop_result.issued,
            partial_failure=not stop_result.success,
            detail=stop_result.detail or reason,
        )
        if stop_result.success:
            self._clear_active()
        else:
            self._state.stop_pending = True
            self._state.pulse_due_stop_ts = now + 0.05
        self._state.last_command_outcome = self._result_dict(result)
        return result

    def shutdown(self, now: float) -> PtzScheduleResult:
        return self.force_stop(now, reason="shutdown_stop")

    def _active_direction(self) -> PtzDirection | None:
        if self._state.active_direction is None:
            return None
        try:
            return PtzDirection(self._state.active_direction)
        except ValueError:
            logger.warning("ptz_scheduler_unknown_active_direction", direction=self._state.active_direction)
            return None

    def _clear_active(self) -> None:
        self._state.active_direction = None
        self._state.active_kind = None
        self._state.pulse_active = False
        self._state.pulse_started_ts = None
        self._state.pulse_due_stop_ts = None
        self._state.stop_pending = False

    def _result_dict(self, result: PtzScheduleResult) -> dict[str, object]:
        return {
            "kind": result.kind,
            "direction": result.direction,
            "attempted": result.attempted,
            "issued": result.issued,
            "accepted": result.accepted,
            "succeeded": result.succeeded,
            "partial_failure": result.partial_failure,
            "skipped": result.skipped,
            "interrupted": result.interrupted,
            "stop_deferred": result.stop_deferred,
            "detail": result.detail,
        }
