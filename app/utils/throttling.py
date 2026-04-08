from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Debouncer:
    min_interval_seconds: float
    _last_seen: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, key: str, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        with self._lock:
            last = self._last_seen.get(key, 0.0)
            if current - last < self.min_interval_seconds:
                return False
            self._last_seen[key] = current
            return True


@dataclass
class RollingCounter:
    window_seconds: float
    _events: deque[float] = field(default_factory=deque)

    def add(self, ts: float | None = None) -> None:
        current = ts if ts is not None else time.monotonic()
        self._events.append(current)
        self._trim(current)

    def count(self, ts: float | None = None) -> int:
        current = ts if ts is not None else time.monotonic()
        self._trim(current)
        return len(self._events)

    def _trim(self, current: float) -> None:
        while self._events and current - self._events[0] > self.window_seconds:
            self._events.popleft()
