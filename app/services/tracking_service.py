from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn

from app.api.server import StateStore, create_app
from app.camera.rtsp_reader import RtspReader
from app.config import AppConfig
from app.control.control_logic import ControlLogic
from app.control.ptz_client import DahuaPtzClient
from app.control.smoothing import EmaSmoother
from app.control.zoom_logic import ZoomController
from app.detection.yolo_detector import Detector
from app.logging_config import get_logger
from app.models.runtime import Detection, PtzDirection, TrackStatus, TrackingSnapshot
from app.services.metrics import MetricsRegistry
from app.services.overlay import draw_overlay
from app.services.snapshot_manager import SnapshotManager
from app.tracking.tracker import Tracker
from app.utils.timers import CooldownTimer, RateLimiter

logger = get_logger(__name__)


class TrackingService:
    """Coordinates RTSP ingestion, YOLO detection, tracking, PTZ decisions, and API exposure."""

    def __init__(
        self,
        config: AppConfig,
        reader: RtspReader,
        detector: Detector,
        tracker: Tracker,
        ptz_client: DahuaPtzClient,
    ) -> None:
        self._config = config
        self._reader = reader
        self._detector = detector
        self._tracker = tracker
        self._ptz = ptz_client
        self._control_logic = ControlLogic(config.control)
        self._zoom_logic = ZoomController(config.control)
        self._smoother = EmaSmoother(config.tracking.ema_alpha)
        self._stop_event = threading.Event()
        self._state_store = StateStore()
        self._metrics = MetricsRegistry()
        self._move_cooldown = CooldownTimer(config.control.movement_cooldown_seconds)
        self._zoom_cooldown = CooldownTimer(config.control.zoom_cooldown_seconds)
        self._lost_zoom_cooldown = CooldownTimer(config.control.lost_zoom_out_cooldown_seconds)
        self._rate_limiter = RateLimiter(config.control.max_command_rate_hz)
        self._startup_frames = 0
        self._api_server: uvicorn.Server | None = None
        self._api_thread: threading.Thread | None = None
        self._snapshot_manager = SnapshotManager(config.app.snapshot_dir, config.snapshots.max_files)
        self._screenshot_manager = SnapshotManager(config.app.screenshot_dir, config.snapshots.max_files)
        self._last_snapshot_ts = 0.0

    @property
    def state_store(self) -> StateStore:
        return self._state_store

    def install_signal_handlers(self) -> None:
        def _handler(_signum: int, _frame: object | None) -> None:
            self.stop()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def start(self) -> None:
        self.install_signal_handlers()
        self._start_api()
        self._reader.start()
        startup = self._ptz.startup_preset()
        logger.info("startup_preset_result", success=startup.success, dry_run=startup.dry_run)
        self.run_loop()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        logger.info("tracking_service_stopping")
        self._reader.stop()
        self._ptz.emergency_stop()
        if self._api_server is not None:
            self._api_server.should_exit = True
        if self._api_thread is not None:
            self._api_thread.join(timeout=3.0)

    def run_loop(self) -> None:
        logger.info("tracking_service_started")
        sleep_seconds = 1.0 / max(1.0, self._config.control.tick_hz)
        try:
            while not self._stop_event.is_set():
                packet = self._reader.read(timeout=1.0)
                if packet is None:
                    continue
                self._metrics.frames_received.inc()
                frame = packet.frame
                detections, latency_ms = self._detector.detect(frame)
                self._metrics.frames_processed.inc()
                self._metrics.inference_latency_ms.observe(latency_ms)
                target_state = self._tracker.update(
                    detections,
                    frame.shape[1],
                    frame.shape[0],
                    packet.timestamp,
                )
                smoothed_bbox = self._smoother.update(target_state.bbox_xyxy)
                if smoothed_bbox is not None:
                    target_state.bbox_xyxy = smoothed_bbox
                decision = self._control_logic.decide(target_state, frame.shape[1], frame.shape[0])
                zoom_decision = self._zoom_logic.decide(
                    target_state,
                    frame.shape[0],
                    pan_tilt_active=decision.move_direction is not None,
                )
                if zoom_decision.zoom_direction is not None:
                    decision.zoom_direction = zoom_decision.zoom_direction
                    decision.zoom_pulse_ms = zoom_decision.zoom_pulse_ms
                    decision.reason = (
                        f"{decision.reason}+{zoom_decision.reason}"
                        if decision.reason != "idle"
                        else zoom_decision.reason
                    )
                self._handle_state_events(frame, target_state.status, packet.timestamp)
                self._execute_decision(frame, decision, packet.timestamp)
                self._publish_snapshot(
                    frame,
                    packet.frame_index,
                    packet.timestamp,
                    detections,
                    target_state,
                    decision,
                    latency_ms,
                    packet.source_fps,
                )
                time.sleep(sleep_seconds)
        except Exception as exc:
            logger.exception("tracking_service_exception", error=str(exc))
            self._ptz.emergency_stop()
            raise
        finally:
            self.stop()

    def _handle_state_events(self, frame: np.ndarray, status: TrackStatus, now: float) -> None:
        if status == TrackStatus.TRACKING:
            self._startup_frames += 1
        elif status == TrackStatus.LOST:
            self._metrics.target_lost.inc()
            if self._config.snapshots.on_target_lost:
                self._snapshot_manager.save(frame, "target-lost", now)
            if self._config.control.lost_zoom_out_enabled and self._lost_zoom_cooldown.ready(now):
                self._ptz.pulse(PtzDirection.ZOOM_OUT, self._config.control.zoom_pulse_ms)
                self._lost_zoom_cooldown.mark(now)
        elif status == TrackStatus.SEARCHING:
            self._startup_frames = 0
        if self._reader.stats.reconnect_count:
            self._metrics.reconnects.inc(self._reader.stats.reconnect_count)
            self._reader.stats.reconnect_count = 0

    def _execute_decision(self, frame: np.ndarray, decision, now: float) -> None:
        if self._config.app.detect_only or not self._config.control.enabled:
            return
        if self._startup_frames < self._config.control.startup_stable_frames:
            return
        if not self._rate_limiter.allow(now):
            return
        if decision.move_direction and self._move_cooldown.ready(now):
            result = self._ptz.pulse(decision.move_direction, decision.move_pulse_ms)
            self._metrics.ptz_commands.inc()
            self._move_cooldown.mark(now)
            self._save_action_screenshot(frame, decision.move_direction.value, now)
            logger.info("ptz_move_action", direction=decision.move_direction.value, success=result.success)
        if decision.zoom_direction and self._zoom_cooldown.ready(now):
            result = self._ptz.pulse(decision.zoom_direction, decision.zoom_pulse_ms)
            self._metrics.ptz_commands.inc()
            self._zoom_cooldown.mark(now)
            self._save_action_screenshot(frame, decision.zoom_direction.value, now)
            logger.info("ptz_zoom_action", direction=decision.zoom_direction.value, success=result.success)

    def _publish_snapshot(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        detections: list[Detection],
        target_state,
        decision,
        latency_ms: float,
        source_fps: float,
    ) -> None:
        status_map = {TrackStatus.SEARCHING: 0, TrackStatus.TRACKING: 1, TrackStatus.LOST: 2}
        self._metrics.tracking_status.set(status_map[target_state.status])
        snapshot = TrackingSnapshot(
            frame_index=frame_index,
            timestamp=timestamp,
            detections=detections,
            target=target_state,
            decision=decision,
            inference_latency_ms=latency_ms,
            extras={"fps": source_fps},
        )
        self._state_store.set_snapshot(snapshot)
        if self._config.app.overlay:
            overlay = draw_overlay(frame, snapshot, self._config.control)
            if self._config.app.debug_window:
                cv2.imshow("ptz-autotrack", overlay)
                cv2.waitKey(1)
        if (
            self._config.snapshots.periodic_debug_frame_seconds > 0
            and timestamp - self._last_snapshot_ts >= self._config.snapshots.periodic_debug_frame_seconds
        ):
            self._snapshot_manager.save(frame, "periodic", timestamp)
            self._last_snapshot_ts = timestamp

    def _save_action_screenshot(self, frame: np.ndarray, action: str, now: float) -> None:
        if self._config.app.save_action_screenshots:
            self._screenshot_manager.save(frame, f"action-{action}", now)

    def _start_api(self) -> None:
        if not self._config.app.api.enabled:
            return
        app = create_app(
            self._config,
            self._metrics,
            self._state_store,
            ptz_test_callback=lambda direction: self._ptz.pulse(direction, self._config.control.pan_pulse_ms_small),
        )
        uvicorn_config = uvicorn.Config(
            app,
            host=self._config.app.api.host,
            port=self._config.app.api.port,
            log_level="warning",
        )
        self._api_server = uvicorn.Server(uvicorn_config)
        self._api_thread = threading.Thread(target=self._api_server.run, daemon=True, name="api-server")
        self._api_thread.start()
