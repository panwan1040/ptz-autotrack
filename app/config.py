from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


TargetStrategy = Literal["largest", "most_centered", "highest_confidence", "stick_nearest"]
LostBehavior = Literal["hold", "zoom_out", "return_home"]
AuthMode = Literal["digest_or_basic", "basic", "digest"]


class ApiConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    metrics_enabled: bool = True


class AppSection(BaseModel):
    name: str = "ptz-autotrack"
    environment: str = "development"
    log_level: str = "INFO"
    debug_window: bool = False
    overlay: bool = False
    detect_only: bool = False
    dry_run_ptz: bool = True
    save_action_screenshots: bool = False
    screenshot_dir: Path = Path("./artifacts/screenshots")
    snapshot_dir: Path = Path("./artifacts/snapshots")
    api: ApiConfig = Field(default_factory=ApiConfig)


class CameraSection(BaseModel):
    host: str
    username: str
    password: SecretStr
    rtsp_port: int = 554
    http_port: int = 80
    channel: int = 1
    rtsp_path: str = "/cam/realmonitor?channel=1&subtype=1"
    timeout_seconds: float = 3.0
    connect_timeout_seconds: float = 10.0
    home_preset_name: str | None = None
    startup_preset_name: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rtsp_url(self) -> str:
        password = self.password.get_secret_value()
        return (
            f"rtsp://{self.username}:{password}@{self.host}:{self.rtsp_port}{self.rtsp_path}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def http_base_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"


class VideoSection(BaseModel):
    resize_width: int = 960
    resize_height: int = 540
    queue_size: int = 2
    reconnect_backoff_seconds: float = 2.0
    read_retry_delay_seconds: float = 0.2
    target_fps: float = 10.0


class DetectionSection(BaseModel):
    model_path: str = "yolov8n.pt"
    confidence: float = 0.45
    iou: float = 0.5
    imgsz: int = 640
    device: str = "cpu"
    half: bool = False
    classes: list[str] = Field(default_factory=lambda: ["person"])


class TrackingSection(BaseModel):
    strategy: TargetStrategy = "stick_nearest"
    min_persist_frames: int = 3
    lost_timeout_seconds: float = 2.0
    switch_margin_ratio: float = 0.25
    ema_alpha: float = 0.35
    max_association_distance: float = 0.18


class ZoomSection(BaseModel):
    enabled: bool = True
    target_height_ratio: float = 0.38
    min_height_ratio: float = 0.24
    max_height_ratio: float = 0.52
    hysteresis: float = 0.04


class ControlSection(BaseModel):
    enabled: bool = True
    tick_hz: float = 5.0
    dead_zone_x: float = 0.10
    dead_zone_y: float = 0.12
    movement_cooldown_seconds: float = 0.25
    zoom_cooldown_seconds: float = 1.2
    aggressive_pan_threshold: float = 0.30
    aggressive_tilt_threshold: float = 0.30
    pan_pulse_ms_small: int = 120
    pan_pulse_ms_large: int = 220
    tilt_pulse_ms_small: int = 120
    tilt_pulse_ms_large: int = 220
    diagonal_pulse_ms: int = 160
    zoom_pulse_ms: int = 180
    startup_stable_frames: int = 5
    max_command_rate_hz: float = 4.0
    allow_zoom_during_pan_tilt: bool = False
    lost_behavior: LostBehavior = "hold"
    lost_zoom_out_enabled: bool = True
    lost_zoom_out_cooldown_seconds: float = 5.0
    return_home_timeout_seconds: float = 30.0
    zoom: ZoomSection = Field(default_factory=ZoomSection)


class PtzSection(BaseModel):
    auth_mode: AuthMode = "digest_or_basic"
    verify_ssl: bool = False
    request_timeout_seconds: float = 2.0
    command_retry_count: int = 2
    command_retry_backoff_seconds: float = 0.2
    speed: int = 1
    debounce_seconds: float = 0.05
    dry_run: bool = True


class SnapshotSection(BaseModel):
    on_target_acquired: bool = True
    on_target_lost: bool = True
    periodic_debug_frame_seconds: int = 0
    max_files: int = 200


class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    camera: CameraSection
    video: VideoSection = Field(default_factory=VideoSection)
    detection: DetectionSection = Field(default_factory=DetectionSection)
    tracking: TrackingSection = Field(default_factory=TrackingSection)
    control: ControlSection = Field(default_factory=ControlSection)
    ptz: PtzSection = Field(default_factory=PtzSection)
    snapshots: SnapshotSection = Field(default_factory=SnapshotSection)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "AppConfig":
        if self.control.dead_zone_x <= 0 or self.control.dead_zone_x >= 0.5:
            raise ValueError("control.dead_zone_x must be between 0 and 0.5")
        if self.control.dead_zone_y <= 0 or self.control.dead_zone_y >= 0.5:
            raise ValueError("control.dead_zone_y must be between 0 and 0.5")
        if self.control.zoom.min_height_ratio >= self.control.zoom.max_height_ratio:
            raise ValueError("zoom min_height_ratio must be less than max_height_ratio")
        return self

    def sanitized_dump(self) -> dict[str, object]:
        data = self.model_dump(mode="json")
        camera_data = data.get("camera", {})
        if isinstance(camera_data, dict):
            camera_data["password"] = "***redacted***"
            if isinstance(camera_data.get("rtsp_url"), str):
                camera_data["rtsp_url"] = mask_secret_in_url(camera_data["rtsp_url"])
        return data


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_path: Path = Path("configs/config.example.yaml")
    env: str | None = None
    log_level: str | None = None
    enable_debug_window: bool | None = None
    enable_overlay: bool | None = None
    enable_api: bool | None = None
    api_host: str | None = None
    api_port: int | None = None
    metrics_enabled: bool | None = None
    detect_only: bool | None = None
    dry_run_ptz: bool | None = None
    save_action_screenshots: bool | None = None
    screenshot_dir: Path | None = None
    snapshot_dir: Path | None = None
    model_path: str | None = None
    device: str | None = None
    half: bool | None = None
    confidence: float | None = None
    iou: float | None = None
    imgsz: int | None = None
    camera_host: str | None = None
    camera_username: str | None = None
    camera_password: str | None = None
    camera_rtsp_port: int | None = None
    camera_http_port: int | None = None
    camera_rtsp_path: str | None = None
    camera_channel: int | None = None
    camera_timeout_seconds: float | None = None
    camera_connect_timeout_seconds: float | None = None


def mask_secret_in_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, remainder = url.split("://", 1)
    credentials, host = remainder.split("@", 1)
    if ":" not in credentials:
        return f"{scheme}://***@{host}"
    username, _password = credentials.split(":", 1)
    return f"{scheme}://{username}:***@{host}"


def deep_update(base: dict[str, object], updates: dict[str, object]) -> dict[str, object]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value
    return base


def load_yaml_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def load_config() -> AppConfig:
    settings = EnvSettings()
    file_data = load_yaml_config(settings.config_path)

    override_data: dict[str, object] = {
        "app": {
            key: value
            for key, value in {
                "environment": settings.env,
                "log_level": settings.log_level,
                "debug_window": settings.enable_debug_window,
                "overlay": settings.enable_overlay,
                "detect_only": settings.detect_only,
                "dry_run_ptz": settings.dry_run_ptz,
                "save_action_screenshots": settings.save_action_screenshots,
                "screenshot_dir": str(settings.screenshot_dir) if settings.screenshot_dir else None,
                "snapshot_dir": str(settings.snapshot_dir) if settings.snapshot_dir else None,
            }.items()
            if value is not None
        },
        "camera": {
            key: value
            for key, value in {
                "host": settings.camera_host,
                "username": settings.camera_username,
                "password": settings.camera_password,
                "rtsp_port": settings.camera_rtsp_port,
                "http_port": settings.camera_http_port,
                "rtsp_path": settings.camera_rtsp_path,
                "channel": settings.camera_channel,
                "timeout_seconds": settings.camera_timeout_seconds,
                "connect_timeout_seconds": settings.camera_connect_timeout_seconds,
            }.items()
            if value is not None
        },
        "detection": {
            key: value
            for key, value in {
                "model_path": settings.model_path,
                "device": settings.device,
                "half": settings.half,
                "confidence": settings.confidence,
                "iou": settings.iou,
                "imgsz": settings.imgsz,
            }.items()
            if value is not None
        },
    }

    api_overrides = {
        key: value
        for key, value in {
            "enabled": settings.enable_api,
            "host": settings.api_host,
            "port": settings.api_port,
            "metrics_enabled": settings.metrics_enabled,
        }.items()
        if value is not None
    }
    if api_overrides:
        override_data.setdefault("app", {})
        app_data = override_data["app"]
        if isinstance(app_data, dict):
            app_data["api"] = api_overrides

    merged = deep_update(file_data, override_data)
    return AppConfig.model_validate(merged)
