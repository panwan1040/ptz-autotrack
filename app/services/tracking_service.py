from __future__ import annotations

import signal
import threading
import time

import cv2
import numpy as np

from app.api.server import StateStore
from app.camera.rtsp_reader import RtspReader
from app.config import AppConfig
from app.control.control_logic import ControlLogic
from app.control.ptz_client import DahuaPtzClient
from app.control.smoothing import EmaSmoother
from app.control.zoom_logic import ZoomController
from app.detection.yolo_detector import Detector
from app.logging_config import get_logger
from app.models.runtime import (
    ControlDecision,
    Detection,
    PtzDirection,
    TargetState,
    TrackStatus,
    TrackingPhase,
    TrackingSnapshot,
    compatibility_status_for_phase,
)
from app.services.metrics import MetricsRegistry
from app.services.overlay import draw_overlay
from app.services.snapshot_manager import SnapshotManager
from app.tracking.tracker import Tracker
from app.utils.throttling import Debouncer
from app.utils.timers import CooldownTimer, LoopRegulator, RateLimiter

logger = get_logger(__name__)


class TrackingService:
    """Coordinates RTSP ingestion, detection, tracking, PTZ control, and API exposure."""

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
        self._loop_regulator = LoopRegulator(config.control.tick_hz)
        self._skip_log_debouncer = Debouncer(1.0)
        self._startup_frames = 0
        self._worker_thread: threading.Thread | None = None
        self._snapshot_manager = SnapshotManager(config.app.snapshot_dir, config.snapshots.max_files)
        self._screenshot_manager = SnapshotManager(config.app.screenshot_dir, config.snapshots.max_files)
        self._last_snapshot_ts = 0.0
        self._previous_target = TargetState(track_id=None, bbox_xyxy=None)
        self._loss_started_at: float | None = None
        self._return_home_issued = False
        self._tracking_phase = TrackingPhase.IDLE
        self._last_ptz_action: str | None = None
        self._last_skip_reason: str | None = None
        self._runtime_started = False

    @property
    def state_store(self) -> StateStore:
        return self._state_store

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def config(self) -> AppConfig:
        return self._config

    def install_signal_handlers(self) -> None:
        def _handler(_signum: int, _frame: object | None) -> None:
            self.request_stop()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._startup_runtime()
        self._worker_thread = threading.Thread(target=self._run_worker, name="tracking-service", daemon=True)
        self._worker_thread.start()

    def run_foreground(self) -> None:
        self.install_signal_handlers()
        self._stop_event.clear()
        self._startup_runtime()
        try:
            self.run_loop()
        finally:
            self._shutdown_runtime()

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            if threading.current_thread() is not self._worker_thread:
                self._worker_thread.join(timeout=3.0)
        self._shutdown_runtime()

    def run_loop(self) -> None:
        logger.info("tracking_service_started")
        self._tracking_phase = TrackingPhase.SEARCHING
        read_timeout = min(1.0, max(0.05, self._loop_regulator.target_period_seconds or 0.05))
        try:
            while not self._stop_event.is_set():
                loop_started_at = time.monotonic()
                packet = self._reader.read(timeout=read_timeout)
                if packet is None:
                    self._sleep_for_tick(loop_started_at)
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
                if smoothed_bbox is not None and target_state.visible:
                    target_state.bbox_xyxy = smoothed_bbox

                decision = self._control_logic.decide(target_state, frame.shape[1], frame.shape[0])
                zoom_decision = self._zoom_logic.decide(
                    target_state,
                    frame.shape[0],
                    pan_tilt_active=decision.move_direction is not None,
                    normalized_error_x=decision.normalized_error_x,
                    normalized_error_y=decision.normalized_error_y,
                )
                if zoom_decision.zoom_direction is not None:
                    decision.zoom_direction = zoom_decision.zoom_direction
                    decision.zoom_pulse_ms = zoom_decision.zoom_pulse_ms
                    decision.reason = (
                        f"{decision.reason}+{zoom_decision.reason}"
                        if decision.reason != "idle"
                        else zoom_decision.reason
                    )
                elif decision.reason == "idle":
                    decision.reason = zoom_decision.reason

                self._handle_tracking_state(frame, self._previous_target, target_state, packet.timestamp)
                self._execute_decision(frame, target_state, decision, packet.timestamp)
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
                self._previous_target = self._copy_target_state(target_state)
                self._sleep_for_tick(loop_started_at)
        except Exception as exc:
            self._tracking_phase = TrackingPhase.ERROR
            logger.exception("tracking_service_exception", error=str(exc))
            self._ptz.emergency_stop()
            raise

    def _handle_tracking_state(
        self,
        frame: np.ndarray,
        previous: TargetState,
        current: TargetState,
        now: float,
    ) -> None:
        phase_before = self._tracking_phase
        self._tracking_phase = self._determine_phase(current)
        current.status = compatibility_status_for_phase(self._tracking_phase)
        self._log_state_transition(previous, current, now, phase_before)

        if current.status == TrackStatus.TRACKING:
            self._startup_frames += 1
            self._loss_started_at = None
            self._return_home_issued = False
            if previous.status != TrackStatus.TRACKING and self._config.snapshots.on_target_acquired:
                self._snapshot_manager.save(frame, "target-acquired", now)
        else:
            self._startup_frames = 0

        if previous.status == TrackStatus.TRACKING and current.status != TrackStatus.TRACKING:
            self._metrics.target_lost.inc()
            if self._config.snapshots.on_target_lost:
                self._snapshot_manager.save(frame, "target-lost", now)
            self._loss_started_at = previous.last_seen_ts
            self._return_home_issued = False
            self._ptz.stop()

        if current.status in {TrackStatus.LOST, TrackStatus.SEARCHING} and self._loss_started_at is not None:
            self._apply_lost_behavior(now)

        if self._reader.stats.reconnect_count:
            self._metrics.reconnects.inc(self._reader.stats.reconnect_count)
            self._reader.stats.reconnect_count = 0

    def _apply_lost_behavior(self, now: float) -> None:
        behavior = self._config.control.lost_behavior
        loss_age = now - self._loss_started_at if self._loss_started_at is not None else 0.0

        if behavior in {"zoom_out", "return_home"} and self._config.control.lost_zoom_out_enabled:
            if self._lost_zoom_cooldown.ready(now):
                result = self._ptz.pulse(PtzDirection.ZOOM_OUT, self._config.control.zoom_pulse_ms)
                self._lost_zoom_cooldown.mark(now)
                self._last_ptz_action = PtzDirection.ZOOM_OUT.value
                logger.info(
                    "tracking_loss_zoom_out",
                    success=result.success,
                    detail=result.detail,
                    loss_age=loss_age,
                )

        if behavior == "return_home" and not self._return_home_issued:
            if loss_age >= self._config.control.return_home_timeout_seconds:
                result = self._ptz.move_home()
                self._return_home_issued = result.success
                if result.success:
                    self._tracking_phase = TrackingPhase.RETURNING_HOME
                    self._last_ptz_action = "return_home"
                logger.info(
                    "tracking_return_home",
                    success=result.success,
                    detail=result.detail,
                    loss_age=loss_age,
                )
            elif not self._config.camera.home_preset_name:
                if self._skip_log_debouncer.allow("tracking_return_home_disabled", now):
                    logger.info("tracking_return_home_disabled", reason="home_preset_not_configured")

    def _execute_decision(
        self,
        frame: np.ndarray,
        target_state: TargetState,
        decision: ControlDecision,
        now: float,
    ) -> None:
        skip_reason = self._decision_skip_reason(target_state, decision, now)
        if skip_reason is not None:
            self._last_skip_reason = skip_reason
            if self._skip_log_debouncer.allow(skip_reason, now):
                logger.info("ptz_action_skipped", reason=skip_reason, status=target_state.status.value)
            return

        if decision.move_direction is not None:
            if not self._move_cooldown.ready(now):
                self._log_local_skip("movement_cooldown", target_state, now)
            else:
                result = self._ptz.pulse(decision.move_direction, decision.move_pulse_ms)
                self._metrics.ptz_commands.inc()
                self._move_cooldown.mark(now)
                self._last_ptz_action = decision.move_direction.value
                self._last_skip_reason = None
                self._save_action_screenshot(frame, decision.move_direction.value, now)
                logger.info(
                    "ptz_move_action",
                    direction=decision.move_direction.value,
                    pulse_ms=decision.move_pulse_ms,
                    success=result.success,
                    detail=result.detail,
                    reason=decision.reason,
                )

        if decision.zoom_direction is not None:
            if decision.move_direction is not None and not self._config.control.allow_zoom_during_pan_tilt:
                self._log_local_skip("zoom_blocked_pan_tilt_active", target_state, now)
            elif not self._zoom_cooldown.ready(now):
                self._log_local_skip("zoom_cooldown", target_state, now)
            else:
                result = self._ptz.pulse(decision.zoom_direction, decision.zoom_pulse_ms)
                self._metrics.ptz_commands.inc()
                self._zoom_cooldown.mark(now)
                self._last_ptz_action = decision.zoom_direction.value
                self._last_skip_reason = None
                self._save_action_screenshot(frame, decision.zoom_direction.value, now)
                logger.info(
                    "ptz_zoom_action",
                    direction=decision.zoom_direction.value,
                    pulse_ms=decision.zoom_pulse_ms,
                    success=result.success,
                    detail=result.detail,
                    reason=decision.reason,
                )

    def _decision_skip_reason(
        self,
        target_state: TargetState,
        decision: ControlDecision,
        now: float,
    ) -> str | None:
        if self._config.app.detect_only or not self._config.control.enabled:
            return "control_disabled"
        if target_state.status != TrackStatus.TRACKING:
            return f"target_not_tracking:{target_state.status.value}"
        if self._startup_frames < self._config.control.startup_stable_frames:
            return "startup_stabilization"
        if decision.move_direction is None and decision.zoom_direction is None:
            return f"no_action:{decision.reason}"
        if not self._rate_limiter.allow(now):
            return "rate_limited"
        return None

    def _publish_snapshot(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        detections: list[Detection],
        target_state: TargetState,
        decision: ControlDecision,
        latency_ms: float,
        source_fps: float,
    ) -> None:
        status_map = {TrackStatus.SEARCHING: 0, TrackStatus.TRACKING: 1, TrackStatus.LOST: 2}
        phase_map = {
            TrackingPhase.IDLE: 0,
            TrackingPhase.SEARCHING: 1,
            TrackingPhase.ACQUIRING: 2,
            TrackingPhase.TRACKING: 3,
            TrackingPhase.LOST: 4,
            TrackingPhase.RETURNING_HOME: 5,
            TrackingPhase.ERROR: 6,
        }
        self._metrics.tracking_status.set(status_map[target_state.status])
        self._metrics.tracking_phase.set(phase_map[self._tracking_phase])
        snapshot = TrackingSnapshot(
            frame_index=frame_index,
            timestamp=timestamp,
            tracking_phase=self._tracking_phase,
            detections=detections,
            target=target_state,
            decision=decision,
            inference_latency_ms=latency_ms,
            current_ptz_action=self._last_ptz_action,
            last_skip_reason=self._last_skip_reason,
            return_home_enabled=bool(self._config.camera.home_preset_name),
            return_home_issued=self._return_home_issued,
            extras={
                "fps": source_fps,
                "loss_started_at": self._loss_started_at,
                "return_home_issued": self._return_home_issued,
            },
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

    def _log_state_transition(
        self,
        previous: TargetState,
        current: TargetState,
        now: float,
        previous_phase: TrackingPhase,
    ) -> None:
        if previous_phase != self._tracking_phase:
            logger.info(
                "tracking_phase_transition",
                previous_phase=previous_phase.value,
                phase=self._tracking_phase.value,
                track_id=current.track_id,
                reason=current.selection_reason,
                lost_duration_seconds=current.lost_duration_seconds,
            )
        if previous.track_id != current.track_id and current.status == TrackStatus.TRACKING:
            logger.info(
                "tracking_target_switch",
                previous_track_id=previous.track_id,
                track_id=current.track_id,
                reason=current.selection_reason,
                score=current.candidate_score,
            )
        if previous.status != current.status:
            logger.info(
                "tracking_status_transition",
                previous_status=previous.status.value,
                status=current.status.value,
                previous_phase=previous_phase.value,
                phase=self._tracking_phase.value,
                track_id=current.track_id,
                reason=current.selection_reason,
                lost_duration_seconds=current.lost_duration_seconds,
            )
        if previous.status == TrackStatus.TRACKING and current.status == TrackStatus.LOST:
            logger.info(
                "tracking_target_lost",
                track_id=previous.track_id,
                last_seen_ts=previous.last_seen_ts,
                now=now,
            )
        if previous.status in {TrackStatus.LOST, TrackStatus.SEARCHING} and current.status == TrackStatus.TRACKING:
            logger.info(
                "tracking_target_reacquired",
                track_id=current.track_id,
                reason=current.selection_reason,
                stable=current.stable,
            )

    def _copy_target_state(self, target: TargetState) -> TargetState:
        return TargetState(
            track_id=target.track_id,
            bbox_xyxy=target.bbox_xyxy,
            confidence=target.confidence,
            persist_frames=target.persist_frames,
            last_seen_ts=target.last_seen_ts,
            status=target.status,
            stable=target.stable,
            visible=target.visible,
            selection_reason=target.selection_reason,
            candidate_score=target.candidate_score,
            lost_duration_seconds=target.lost_duration_seconds,
        )

    def _log_local_skip(self, reason: str, target_state: TargetState, now: float) -> None:
        self._last_skip_reason = reason
        if self._skip_log_debouncer.allow(reason, now):
            logger.info("ptz_action_skipped", reason=reason, status=target_state.status.value)

    def _determine_phase(self, target_state: TargetState) -> TrackingPhase:
        if self._tracking_phase == TrackingPhase.ERROR:
            return TrackingPhase.ERROR
        if self._return_home_issued:
            return TrackingPhase.RETURNING_HOME
        if target_state.status == TrackStatus.TRACKING:
            return TrackingPhase.TRACKING
        if target_state.visible and not target_state.stable:
            return TrackingPhase.ACQUIRING
        if target_state.status == TrackStatus.LOST:
            return TrackingPhase.LOST
        return TrackingPhase.SEARCHING

    def ptz_test(self, direction: PtzDirection) -> None:
        self._ptz.pulse(direction, self._config.control.pan_pulse_ms_small)

    def _startup_runtime(self) -> None:
        if self._runtime_started:
            return
        self._tracking_phase = TrackingPhase.SEARCHING
        self._reader.start()
        startup = self._ptz.startup_preset()
        logger.info(
            "startup_preset_result",
            success=startup.success,
            dry_run=startup.dry_run,
            detail=startup.detail,
        )
        self._runtime_started = True

    def _shutdown_runtime(self) -> None:
        if not self._runtime_started:
            return
        self._tracking_phase = TrackingPhase.IDLE
        logger.info("tracking_service_stopping")
        self._reader.stop()
        self._ptz.emergency_stop()
        self._runtime_started = False
        if self._config.app.debug_window:
            cv2.destroyAllWindows()

    def _run_worker(self) -> None:
        try:
            self.run_loop()
        except Exception:
            logger.exception("tracking_worker_exited_with_error")
        finally:
            self._shutdown_runtime()

    def _sleep_for_tick(self, started_at: float) -> None:
        elapsed, remaining = self._loop_regulator.sleep_after(started_at)
        if remaining <= 0 and elapsed > self._loop_regulator.target_period_seconds:
            logger.debug(
                "tracking_loop_overrun",
                elapsed_seconds=round(elapsed, 4),
                target_period_seconds=round(self._loop_regulator.target_period_seconds, 4),
            )
