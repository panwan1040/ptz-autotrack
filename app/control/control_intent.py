from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models.runtime import ControlMode, PtzDirection


class PtzIntentKind(str, Enum):
    NOOP = "noop"
    MOVE = "move"
    ZOOM = "zoom"
    STOP = "stop"


@dataclass(slots=True)
class PtzIntent:
    kind: PtzIntentKind = PtzIntentKind.NOOP
    direction: PtzDirection | None = None
    pulse_ms: int = 0
    reason: str = "idle"
    control_mode: ControlMode = ControlMode.IDLE
    allow_interrupt: bool = True
    priority: int = 0
    predicted: bool = False
