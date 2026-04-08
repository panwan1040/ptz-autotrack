from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from app.config import CameraSection, PtzSection
from app.logging_config import get_logger
from app.models.runtime import PtzDirection
from app.utils.throttling import Debouncer

logger = get_logger(__name__)


class HttpSession(Protocol):
    def get(self, url: str, *, params: dict[str, object], timeout: float, verify: bool): ...


@dataclass(slots=True)
class PtzCommandResult:
    success: bool
    direction: str
    pulse_ms: int
    dry_run: bool
    error: str | None = None


class DahuaPtzClient:
    """Thin reusable Dahua PTZ CGI client with dry-run, retries, and pulse motion."""

    def __init__(
        self,
        camera: CameraSection,
        config: PtzSection,
        session: HttpSession | None = None,
        detect_only: bool = False,
    ) -> None:
        self._camera = camera
        self._config = config
        self._detect_only = detect_only
        self._session = session or requests.Session()
        self._debouncer = Debouncer(config.debounce_seconds)
        self._active_direction: str | None = None
        self._configure_auth()

    def _configure_auth(self) -> None:
        if not isinstance(self._session, requests.Session):
            return
        username = self._camera.username
        password = self._camera.password.get_secret_value()
        if self._config.auth_mode == "basic":
            self._session.auth = HTTPBasicAuth(username, password)
        elif self._config.auth_mode == "digest":
            self._session.auth = HTTPDigestAuth(username, password)
        else:
            self._session.auth = HTTPDigestAuth(username, password)

    @property
    def is_dry_run(self) -> bool:
        return self._config.dry_run or self._detect_only

    def pulse(self, direction: PtzDirection, pulse_ms: int) -> PtzCommandResult:
        if direction == PtzDirection.STOP:
            return self.stop()
        if pulse_ms <= 0:
            return PtzCommandResult(True, direction.value, pulse_ms, self.is_dry_run)
        start_result = self.start(direction)
        if not start_result.success:
            return start_result
        time.sleep(pulse_ms / 1000.0)
        stop_result = self.stop(direction)
        return stop_result if not stop_result.success else start_result

    def start(self, direction: PtzDirection) -> PtzCommandResult:
        if not self._debouncer.allow(f"start:{direction.value}"):
            return PtzCommandResult(True, direction.value, 0, self.is_dry_run)
        return self._request("start", direction)

    def stop(self, direction: PtzDirection | None = None) -> PtzCommandResult:
        target = direction or (PtzDirection(self._active_direction) if self._active_direction else PtzDirection.LEFT)
        return self._request("stop", target)

    def move_home(self) -> PtzCommandResult:
        if not self._camera.home_preset_name:
            return PtzCommandResult(False, "GotoPreset", 0, self.is_dry_run, "home preset not set")
        return self._preset("moveAbsolutely", self._camera.home_preset_name)

    def startup_preset(self) -> PtzCommandResult:
        if not self._camera.startup_preset_name:
            return PtzCommandResult(True, "StartupPreset", 0, self.is_dry_run)
        return self._preset("moveAbsolutely", self._camera.startup_preset_name)

    def emergency_stop(self) -> None:
        try:
            for direction in [
                PtzDirection.LEFT,
                PtzDirection.RIGHT,
                PtzDirection.UP,
                PtzDirection.DOWN,
                PtzDirection.LEFT_UP,
                PtzDirection.RIGHT_UP,
                PtzDirection.LEFT_DOWN,
                PtzDirection.RIGHT_DOWN,
                PtzDirection.ZOOM_IN,
                PtzDirection.ZOOM_OUT,
            ]:
                self._request("stop", direction)
        except Exception as exc:
            logger.error("ptz_emergency_stop_failed", error=str(exc))

    def _preset(self, action: str, name: str) -> PtzCommandResult:
        if self.is_dry_run:
            logger.info("ptz_preset_dry_run", action=action, preset=name)
            return PtzCommandResult(True, action, 0, True)
        url = f"{self._camera.http_base_url}/cgi-bin/ptz.cgi"
        params = {
            "action": action,
            "channel": self._camera.channel,
            "code": name,
            "arg1": 0,
            "arg2": self._config.speed,
            "arg3": 0,
        }
        return self._perform_request(url, params, action)

    def _request(self, action: str, direction: PtzDirection) -> PtzCommandResult:
        if self.is_dry_run:
            logger.info("ptz_dry_run", action=action, direction=direction.value)
            if action == "start":
                self._active_direction = direction.value
            elif action == "stop":
                self._active_direction = None
            return PtzCommandResult(True, direction.value, 0, True)
        url = f"{self._camera.http_base_url}/cgi-bin/ptz.cgi"
        params = {
            "action": action,
            "channel": self._camera.channel,
            "code": direction.value,
            "arg1": 0,
            "arg2": self._config.speed,
            "arg3": 0,
        }
        result = self._perform_request(url, params, direction.value)
        if result.success:
            self._active_direction = direction.value if action == "start" else None
        return result

    def _perform_request(
        self,
        url: str,
        params: dict[str, object],
        direction: str,
    ) -> PtzCommandResult:
        attempts = max(1, self._config.command_retry_count + 1)
        last_error: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._config.request_timeout_seconds,
                    verify=self._config.verify_ssl,
                )
                response.raise_for_status()
                logger.info("ptz_command_sent", direction=direction, params=params)
                return PtzCommandResult(True, direction, 0, False)
            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning(
                    "ptz_command_failed",
                    direction=direction,
                    attempt=attempt,
                    error=last_error,
                )
                if attempt < attempts:
                    time.sleep(self._config.command_retry_backoff_seconds)
        return PtzCommandResult(False, direction, 0, False, last_error)
