from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class CooldownTimer:
    cooldown_seconds: float
    last_ts: float = 0.0

    def ready(self, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        return (current - self.last_ts) >= self.cooldown_seconds

    def mark(self, now: float | None = None) -> None:
        self.last_ts = now if now is not None else time.monotonic()


@dataclass
class RateLimiter:
    max_rate_hz: float
    _last_ts: float = 0.0

    def allow(self, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        min_interval = 1.0 / self.max_rate_hz if self.max_rate_hz > 0 else 0.0
        if current - self._last_ts >= min_interval:
            self._last_ts = current
            return True
        return False
