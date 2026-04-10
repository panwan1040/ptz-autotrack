from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PtzRuntimeState:
    active_direction: str | None = None
    active_kind: str | None = None
    pulse_active: bool = False
    pulse_started_ts: float | None = None
    pulse_due_stop_ts: float | None = None
    stop_pending: bool = False
    last_command_outcome: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PtzScheduleResult:
    kind: str
    direction: str | None
    attempted: bool = False
    issued: bool = False
    accepted: bool = False
    succeeded: bool = False
    partial_failure: bool = False
    skipped: bool = False
    interrupted: bool = False
    stop_deferred: bool = False
    detail: str | None = None
    cooldown_eligible: bool = False
