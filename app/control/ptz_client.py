from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import requests
from requests.auth import AuthBase, HTTPBasicAuth, HTTPDigestAuth

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
    action: str
    direction: str
    pulse_ms: int
    dry_run: bool
    issued: bool = True
    auth_mode: str | None = None
    http_status: int | None = None
    error: str | None = None
    detail: str | None = None


class DahuaPtzClient:
    """Thin reusable Dahua PTZ CGI client with pulse safety and auth fallback."""

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
        self._active_direction: PtzDirection | None = None
        self._last_command_at = 0.0
        self._auth_mode_in_use = self._configure_auth()

    def _configure_auth(self) -> str | None:
        if not isinstance(self._session, requests.Session):
            return None
        auth = self._build_auth(self._config.auth_mode)
        self._session.auth = auth
        if isinstance(auth, HTTPBasicAuth):
            return "basic"
        if isinstance(auth, HTTPDigestAuth):
            return "digest"
        return None

    def _build_auth(self, mode: str) -> AuthBase | None:
        username = self._camera.username
        password = self._camera.password.get_secret_value()
        if mode == "basic":
            return HTTPBasicAuth(username, password)
        return HTTPDigestAuth(username, password)

    @property
    def is_dry_run(self) -> bool:
        return self._config.dry_run or self._detect_only

    def pulse(self, direction: PtzDirection, pulse_ms: int) -> PtzCommandResult:
        if direction == PtzDirection.STOP:
            return self.stop()
        if pulse_ms <= 0:
            return PtzCommandResult(
                success=True,
                action="pulse",
                direction=direction.value,
                pulse_ms=pulse_ms,
                dry_run=self.is_dry_run,
                issued=False,
                detail="pulse_duration_not_positive",
            )

        start_result = self.start(direction)
        if not start_result.success:
            return PtzCommandResult(
                success=False,
                action="pulse",
                direction=direction.value,
                pulse_ms=pulse_ms,
                dry_run=self.is_dry_run,
                issued=start_result.issued,
                auth_mode=start_result.auth_mode,
                http_status=start_result.http_status,
                error=start_result.error,
                detail=start_result.detail,
            )

        stop_result: PtzCommandResult | None = None
        try:
            time.sleep(pulse_ms / 1000.0)
        finally:
            stop_result = self.stop(direction if self._active_direction == direction else None)

        if stop_result is not None and not stop_result.success:
            return PtzCommandResult(
                success=False,
                action="pulse",
                direction=direction.value,
                pulse_ms=pulse_ms,
                dry_run=self.is_dry_run,
                issued=True,
                auth_mode=stop_result.auth_mode,
                http_status=stop_result.http_status,
                error=stop_result.error,
                detail="stop_failed_after_pulse",
            )

        return PtzCommandResult(
            success=True,
            action="pulse",
            direction=direction.value,
            pulse_ms=pulse_ms,
            dry_run=self.is_dry_run,
            issued=start_result.issued or bool(stop_result and stop_result.issued),
            auth_mode=start_result.auth_mode,
            detail="pulse_completed",
        )

    def start(self, direction: PtzDirection) -> PtzCommandResult:
        if self._active_direction == direction:
            return PtzCommandResult(
                success=True,
                action="start",
                direction=direction.value,
                pulse_ms=0,
                dry_run=self.is_dry_run,
                issued=False,
                auth_mode=self._auth_mode_in_use,
                detail="duplicate_start_suppressed",
            )
        if self._active_direction is not None and self._active_direction != direction:
            stop_result = self.stop(self._active_direction)
            if not stop_result.success:
                return PtzCommandResult(
                    success=False,
                    action="start",
                    direction=direction.value,
                    pulse_ms=0,
                    dry_run=self.is_dry_run,
                    issued=False,
                    auth_mode=stop_result.auth_mode,
                    http_status=stop_result.http_status,
                    error=stop_result.error,
                    detail="failed_to_stop_previous_direction",
                )
        if not self._debouncer.allow(f"start:{direction.value}"):
            return PtzCommandResult(
                success=True,
                action="start",
                direction=direction.value,
                pulse_ms=0,
                dry_run=self.is_dry_run,
                issued=False,
                auth_mode=self._auth_mode_in_use,
                detail="debounced_start",
            )
        return self._request("start", direction)

    def stop(self, direction: PtzDirection | None = None) -> PtzCommandResult:
        target = direction or self._active_direction
        if target is None:
            return PtzCommandResult(
                success=True,
                action="stop",
                direction=PtzDirection.STOP.value,
                pulse_ms=0,
                dry_run=self.is_dry_run,
                issued=False,
                auth_mode=self._auth_mode_in_use,
                detail="no_active_direction",
            )
        return self._request("stop", target)

    def move_home(self) -> PtzCommandResult:
        return self._goto_preset(self._camera.home_preset_name, purpose="home")

    def startup_preset(self) -> PtzCommandResult:
        if not self._camera.startup_preset_name:
            return PtzCommandResult(
                success=True,
                action="preset",
                direction="StartupPreset",
                pulse_ms=0,
                dry_run=self.is_dry_run,
                issued=False,
                auth_mode=self._auth_mode_in_use,
                detail="startup_preset_not_configured",
            )
        return self._goto_preset(self._camera.startup_preset_name, purpose="startup")

    def emergency_stop(self) -> None:
        try:
            if self._active_direction is not None:
                self.stop(self._active_direction)
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

    def _goto_preset(self, preset_name: str | None, purpose: str) -> PtzCommandResult:
        if not preset_name:
            return PtzCommandResult(
                success=False,
                action="preset",
                direction="GotoPreset",
                pulse_ms=0,
                dry_run=self.is_dry_run,
                issued=False,
                auth_mode=self._auth_mode_in_use,
                detail=f"{purpose}_preset_not_configured",
                error="preset not set",
            )
        if self.is_dry_run:
            logger.info("ptz_preset_dry_run", purpose=purpose, preset=preset_name)
            return PtzCommandResult(
                success=True,
                action="preset",
                direction="GotoPreset",
                pulse_ms=0,
                dry_run=True,
                auth_mode=self._auth_mode_in_use,
                detail=f"{purpose}_preset_dry_run",
            )
        url = f"{self._camera.http_base_url}/cgi-bin/ptz.cgi"
        params = {
            "action": "moveAbsolutely",
            "channel": self._camera.channel,
            "code": preset_name,
            "arg1": 0,
            "arg2": self._config.speed,
            "arg3": 0,
        }
        return self._perform_request(url, params, action="preset", direction=purpose)

    def _request(self, action: str, direction: PtzDirection) -> PtzCommandResult:
        if self.is_dry_run:
            logger.info("ptz_dry_run", action=action, direction=direction.value)
            self._update_active_direction(action, direction)
            return PtzCommandResult(
                success=True,
                action=action,
                direction=direction.value,
                pulse_ms=0,
                dry_run=True,
                auth_mode=self._auth_mode_in_use,
                detail="dry_run",
            )
        url = f"{self._camera.http_base_url}/cgi-bin/ptz.cgi"
        params = {
            "action": action,
            "channel": self._camera.channel,
            "code": direction.value,
            "arg1": 0,
            "arg2": self._config.speed,
            "arg3": 0,
        }
        result = self._perform_request(url, params, action=action, direction=direction.value)
        if result.success:
            self._update_active_direction(action, direction)
        return result

    def _update_active_direction(self, action: str, direction: PtzDirection) -> None:
        self._last_command_at = time.monotonic()
        if action == "start":
            self._active_direction = direction
        elif action == "stop" and (self._active_direction is None or self._active_direction == direction):
            self._active_direction = None

    def _perform_request(
        self,
        url: str,
        params: dict[str, object],
        *,
        action: str,
        direction: str,
    ) -> PtzCommandResult:
        attempts = max(1, self._config.command_retry_count + 1)
        last_error: str | None = None
        last_status: int | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._config.request_timeout_seconds,
                    verify=self._config.verify_ssl,
                )
                last_status = getattr(response, "status_code", None)
                response.raise_for_status()
                logger.info(
                    "ptz_command_sent",
                    action=action,
                    direction=direction,
                    params=params,
                    auth_mode=self._auth_mode_in_use,
                )
                return PtzCommandResult(
                    success=True,
                    action=action,
                    direction=direction,
                    pulse_ms=0,
                    dry_run=False,
                    auth_mode=self._auth_mode_in_use,
                    http_status=last_status,
                    detail="request_sent",
                )
            except requests.HTTPError as exc:
                last_status = getattr(exc.response, "status_code", last_status)
                if self._maybe_fallback_to_basic(last_status):
                    continue
                last_error = str(exc)
            except requests.RequestException as exc:
                last_error = str(exc)

            logger.warning(
                "ptz_command_failed",
                action=action,
                direction=direction,
                attempt=attempt,
                error=last_error,
                status_code=last_status,
                auth_mode=self._auth_mode_in_use,
            )
            if attempt < attempts:
                time.sleep(self._config.command_retry_backoff_seconds)
        return PtzCommandResult(
            success=False,
            action=action,
            direction=direction,
            pulse_ms=0,
            dry_run=False,
            issued=False,
            auth_mode=self._auth_mode_in_use,
            http_status=last_status,
            error=last_error,
            detail="request_failed",
        )

    def _maybe_fallback_to_basic(self, status_code: int | None) -> bool:
        if (
            status_code != 401
            or self._config.auth_mode != "digest_or_basic"
            or not isinstance(self._session, requests.Session)
            or self._auth_mode_in_use == "basic"
        ):
            return False
        self._session.auth = self._build_auth("basic")
        self._auth_mode_in_use = "basic"
        logger.warning("ptz_auth_fallback", from_mode="digest", to_mode="basic")
        return True
