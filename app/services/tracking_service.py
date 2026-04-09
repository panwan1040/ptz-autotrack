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
from app.control.handoff_manager import HandoffManager
from app.control.lifecycle_manager import LifecycleManager
from app.control.monitoring_policy import MonitoringPolicy
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
from app.utils.geometry import bbox_center, height_ratio, inside_dead_zone, normalized_bbox_center
from app.utils.throttling import Debouncer
from app.utils.timers import CooldownTimer, LoopRegulator, RateLimiter

logger = get_logger(__name__)


class TrackingService:
    """Coordinates RTSP ingestion, detection, tracking, recovery, and PTZ control."""

    ACTIVE_CONTROL_PHASES = {
        TrackingPhase.CENTERING,
        TrackingPhase.ZOOMING_FOR_HANDOFF,
        TrackingPhase.TRACKING,
    }

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
        self._handoff_manager = HandoffManager(config.tracking.handoff)
        self._monitoring_policy = MonitoringPolicy(config.tracking.monitoring)
        self._lifecycle_manager = LifecycleManager(config.tracking)
        self._smoother = EmaSmoother(config.tracking.ema_alpha)
        self._stop_event = threading.Event()
        self._state_store = StateStore()
        self._metrics = MetricsRegistry()
        self._move_cooldown = CooldownTimer(config.control.movement_cooldown_seconds)
        self._zoom_cooldown = CooldownTimer(config.control.zoom_cooldown_seconds)
        self._lost_zoom_cooldown = CooldownTimer(config.control.lost_zoom_out_cooldown_seconds)
        self._recovery_zoom_cooldown = CooldownTimer(config.tracking.recovery.recovery_zoom_cooldown_seconds)
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
                    frame=frame,
                )
                smoothed_bbox = self._smoother.update(target_state.bbox_xyxy)
                if smoothed_bbox is not None and target_state.visible:
                    target_state.bbox_xyxy = smoothed_bbox
                    self._tracker.target_memory.last_smoothed_bbox = smoothed_bbox
                    self._tracker.target_memory.last_center = bbox_center(smoothed_bbox)

                now = time.monotonic()
                frame_age = max(0.0, now - packet.timestamp)
                target_state.frame_age_seconds = frame_age
                target_state.stale_frame = frame_age > self._config.tracking.stale_frame.max_age_seconds
                self._tracker.target_memory.stale_frame_age_seconds = frame_age
                if target_state.stale_frame and self._skip_log_debouncer.allow("stale_frame_warning", now):
                    logger.warning(
                        "stale_frame_detected",
                        frame_age_seconds=round(frame_age, 3),
                        threshold_seconds=self._config.tracking.stale_frame.max_age_seconds,
                    )

                decision = self._control_logic.decide(target_state, frame.shape[1], frame.shape[0])
                zoom_decision = self._zoom_logic.decide(
                    target_state,
                    frame.shape[0],
                    pan_tilt_active=decision.move_direction is not None,
                    normalized_error_x=decision.normalized_error_x,
                    normalized_error_y=decision.normalized_error_y,
                )
                self._merge_zoom_decision(decision, zoom_decision)
                self._handle_tracking_state(
                    frame,
                    self._previous_target,
                    target_state,
                    decision,
                    packet.timestamp,
                    frame.shape[1],
                    frame.shape[0],
                )
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

    def _merge_zoom_decision(self, decision: ControlDecision, zoom_decision: ControlDecision) -> None:
        if zoom_decision.zoom_direction is not None:
            decision.zoom_direction = zoom_decision.zoom_direction
            decision.zoom_pulse_ms = zoom_decision.zoom_pulse_ms
            decision.reason = (
                f"{decision.reason}+{zoom_decision.reason}" if decision.reason != "idle" else zoom_decision.reason
            )
        elif decision.reason == "idle":
            decision.reason = zoom_decision.reason

    def _handle_tracking_state(
        self,
        frame: np.ndarray,
        previous: TargetState,
        current: TargetState,
        decision: ControlDecision,
        now: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        memory = self._tracker.target_memory
        handoff_ready, handoff_reason = self._handoff_manager.evaluate(current, memory, frame_width, frame_height)
        current.handoff_ready = handoff_ready
        handoff_zoom_candidate = False
        if current.visible and current.stable and current.bbox_xyxy is not None:
            nx, ny = normalized_bbox_center(current.bbox_xyxy, frame_width, frame_height)
            centered = inside_dead_zone(
                nx,
                ny,
                self._config.tracking.handoff.inner_dead_zone_x * 1.6,
                self._config.tracking.handoff.inner_dead_zone_y * 1.6,
            )
            target_ratio = height_ratio(current.bbox_xyxy, frame_height)
            handoff_zoom_candidate = centered and target_ratio < self._config.tracking.handoff.min_target_height_ratio

        monitoring_broken = False
        monitoring_reason = "monitoring_not_active"
        if self._tracking_phase in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING}:
            monitoring_broken, monitoring_reason = self._monitoring_policy.should_resume_control(
                current,
                memory,
                frame_width,
                frame_height,
                now,
            )

        phase_before = self._tracking_phase
        self._tracking_phase = self._lifecycle_manager.next_phase(
            phase_before,
            current,
            memory,
            handoff_ready=handoff_ready,
            handoff_zoom_candidate=handoff_zoom_candidate,
            monitoring_broken=monitoring_broken,
            return_home_issued=self._return_home_issued,
            now=now,
        )
        memory.lifecycle_state = self._tracking_phase
        current.status = compatibility_status_for_phase(self._tracking_phase)

        if self._tracking_phase in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING, TrackingPhase.ZOOMING_FOR_HANDOFF}:
            current.selection_reason = handoff_reason
        elif monitoring_broken:
            current.selection_reason = monitoring_reason

        self._log_state_transition(previous, current, now, phase_before)

        if current.visible and current.stable:
            self._startup_frames += 1
            self._return_home_issued = False
            if self._tracking_phase not in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING}:
                memory.handoff_ts = None
            if previous.status != TrackStatus.TRACKING and self._config.snapshots.on_target_acquired:
                self._snapshot_manager.save(frame, "target-acquired", now)
            if self._loss_started_at is not None and previous.status != TrackStatus.TRACKING:
                self._metrics.successful_reacquisition_count.inc()
                self._metrics.time_to_reacquire_seconds.observe(max(0.0, now - self._loss_started_at))
                self._loss_started_at = None
        else:
            self._startup_frames = 0

        if previous.status == TrackStatus.TRACKING and current.status != TrackStatus.TRACKING:
            self._metrics.target_lost.inc()
            if self._config.snapshots.on_target_lost:
                self._snapshot_manager.save(frame, "target-lost", now)
            self._loss_started_at = memory.missing_started_ts or previous.last_seen_ts or now
            self._return_home_issued = False
            self._ptz.stop()

        self._apply_phase_behavior(frame, current, decision, now, frame_width, frame_height)

        if self._reader.stats.reconnect_count:
            self._metrics.reconnects.inc(self._reader.stats.reconnect_count)
            self._reader.stats.reconnect_count = 0

    def _apply_phase_behavior(
        self,
        frame: np.ndarray,
        target_state: TargetState,
        decision: ControlDecision,
        now: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        memory = self._tracker.target_memory
        if self._tracking_phase in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING, TrackingPhase.TEMP_LOST, TrackingPhase.OCCLUDED}:
            self._ptz.stop()
            if self._tracking_phase in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING} and memory.handoff_ts is None:
                memory.handoff_ts = now
            return

        if self._tracking_phase == TrackingPhase.RECOVERY_LOCAL:
            if target_state.frame_age_seconds > self._config.tracking.stale_frame.aggressive_recovery_max_age_seconds:
                self._log_local_skip("stale_frame_blocks_local_recovery", target_state, now)
                return
            if self._maybe_zoom_out_for_recovery(now):
                return
            self._execute_local_recovery(frame, target_state, now, frame_width, frame_height)
            return

        if self._tracking_phase == TrackingPhase.RECOVERY_WIDE:
            if self._maybe_zoom_out_for_recovery(now):
                return
            self._apply_lost_behavior(now)
            return

        if self._tracking_phase == TrackingPhase.LOST:
            self._apply_lost_behavior(now)
            if self._tracker.target_memory.last_confirmed_ts > 0:
                memory_age = now - self._tracker.target_memory.last_confirmed_ts
                if memory_age >= self._config.tracking.target_memory.clear_after_seconds:
                    self._metrics.recovery_failure_count.inc()
                    self._tracker.clear_target_memory()
            return

        if self._tracking_phase not in self.ACTIVE_CONTROL_PHASES:
            self._last_skip_reason = f"phase:{self._tracking_phase.value}"
            return

        memory.recovery_zoom_steps = 0

    def _maybe_zoom_out_for_recovery(self, now: float) -> bool:
        memory = self._tracker.target_memory
        if (
            memory.recovery_zoom_steps >= self._config.tracking.recovery.max_recovery_zoom_steps
            or not self._recovery_zoom_cooldown.ready(now)
        ):
            return False
        if (
            memory.last_zoom_ratio < self._config.tracking.recovery.zoom_out_first_min_height_ratio
            and memory.recovery_zoom_steps == 0
        ):
            return False
        if self._config.app.detect_only or not self._config.control.enabled:
            self._last_skip_reason = "recovery_zoom_out_control_disabled"
            return False
        result = self._ptz.pulse(PtzDirection.ZOOM_OUT, self._config.tracking.recovery.zoom_out_step_pulse_ms)
        if result.success:
            self._metrics.ptz_commands.inc()
            self._recovery_zoom_cooldown.mark(now)
            memory.recovery_zoom_steps += 1
            self._last_ptz_action = PtzDirection.ZOOM_OUT.value
            self._last_skip_reason = None
        logger.info(
            "tracking_recovery_zoom_out",
            success=result.success,
            detail=result.detail,
            step=memory.recovery_zoom_steps,
        )
        return result.success

    def _execute_local_recovery(
        self,
        frame: np.ndarray,
        target_state: TargetState,
        now: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        if self._config.app.detect_only or not self._config.control.enabled:
            self._log_local_skip("recovery_control_disabled", target_state, now)
            return
        if not self._rate_limiter.allow(now):
            self._log_local_skip("recovery_rate_limited", target_state, now)
            return
        if not self._move_cooldown.ready(now):
            self._log_local_skip("recovery_movement_cooldown", target_state, now)
            return
        predicted_bbox = target_state.predicted_window or self._tracker.target_memory.predicted_window
        if predicted_bbox is None:
            self._log_local_skip("no_prediction_for_local_recovery", target_state, now)
            return

        synthetic = TargetState(
            track_id=target_state.track_id,
            bbox_xyxy=predicted_bbox,
            status=TrackStatus.TRACKING,
            stable=True,
            visible=True,
            persist_frames=max(1, target_state.persist_frames),
        )
        decision = self._control_logic.decide(synthetic, frame_width, frame_height)
        if decision.move_direction is None:
            self._log_local_skip("local_recovery_prediction_inside_dead_zone", target_state, now)
            return
        result = self._ptz.pulse(decision.move_direction, decision.move_pulse_ms)
        if result.success:
            self._metrics.ptz_commands.inc()
            self._move_cooldown.mark(now)
            self._last_ptz_action = decision.move_direction.value
            self._last_skip_reason = None
            self._save_action_screenshot(frame, f"recovery-{decision.move_direction.value}", now)
        logger.info(
            "tracking_local_recovery_action",
            direction=decision.move_direction.value,
            pulse_ms=decision.move_pulse_ms,
            success=result.success,
            detail=result.detail,
        )

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
                logger.info("ptz_action_skipped", reason=skip_reason, phase=self._tracking_phase.value)
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
                self._tracker.target_memory.last_ptz_action = decision.move_direction.value
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
                self._tracker.target_memory.last_ptz_action = decision.zoom_direction.value
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
        if self._tracking_phase not in self.ACTIVE_CONTROL_PHASES:
            return f"phase_not_actively_controlled:{self._tracking_phase.value}"
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
        phase_order = list(TrackingPhase)
        phase_map = {phase: index for index, phase in enumerate(phase_order)}
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
                "prediction_confidence": self._tracker.target_memory.prediction_confidence,
                "recovery_zoom_steps": self._tracker.target_memory.recovery_zoom_steps,
                "memory_track_id": self._tracker.target_memory.track_id,
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
                missing_frames=current.missing_frames,
            )
        if previous.track_id != current.track_id and current.track_id is not None and current.visible:
            self._metrics.target_switch_count.inc()
            logger.info(
                "tracking_target_switch",
                previous_track_id=previous.track_id,
                track_id=current.track_id,
                reason=current.selection_reason,
                score=current.candidate_score,
                match_breakdown=current.match_breakdown,
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
        if self._tracking_phase == TrackingPhase.OCCLUDED and previous_phase != TrackingPhase.OCCLUDED:
            self._metrics.occlusion_count.inc()
            logger.info("tracking_occluded", track_id=current.track_id, missing_frames=current.missing_frames)
        if self._tracking_phase == TrackingPhase.RECOVERY_LOCAL and previous_phase != TrackingPhase.RECOVERY_LOCAL:
            self._metrics.local_recovery_count.inc()
            logger.info("tracking_recovery_local_started", track_id=current.track_id)
        if self._tracking_phase == TrackingPhase.RECOVERY_WIDE and previous_phase != TrackingPhase.RECOVERY_WIDE:
            self._metrics.wide_recovery_count.inc()
            logger.info("tracking_recovery_wide_started", track_id=current.track_id)
        if self._tracking_phase == TrackingPhase.HANDOFF and previous_phase != TrackingPhase.HANDOFF:
            self._metrics.handoff_count.inc()
            logger.info("tracking_handoff_initiated", track_id=current.track_id)
        if previous_phase == TrackingPhase.HANDOFF and self._tracking_phase == TrackingPhase.MONITORING:
            logger.info("tracking_handoff_succeeded", track_id=current.track_id)
        if previous_phase in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING} and self._tracking_phase in {
            TrackingPhase.TEMP_LOST,
            TrackingPhase.OCCLUDED,
            TrackingPhase.RECOVERY_LOCAL,
            TrackingPhase.RECOVERY_WIDE,
        }:
            self._metrics.handoff_break_count.inc()
            logger.info("tracking_handoff_broken", track_id=current.track_id, reason=current.selection_reason)
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
            predicted_center=target.predicted_center,
            predicted_window=target.predicted_window,
            appearance_similarity=target.appearance_similarity,
            missing_frames=target.missing_frames,
            visible_frames=target.visible_frames,
            handoff_ready=target.handoff_ready,
            frame_age_seconds=target.frame_age_seconds,
            stale_frame=target.stale_frame,
            match_breakdown=target.match_breakdown.copy(),
        )

    def _log_local_skip(self, reason: str, target_state: TargetState, now: float) -> None:
        self._last_skip_reason = reason
        if self._skip_log_debouncer.allow(reason, now):
            logger.info("ptz_action_skipped", reason=reason, status=target_state.status.value)

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
