from __future__ import annotations

import signal
import threading
import time

import cv2
import numpy as np

from app.api.server import StateStore
from app.camera.rtsp_reader import RtspReader
from app.config import AppConfig
from app.control.control_intent import PtzIntent, PtzIntentKind
from app.control.control_logic import ControlLogic
from app.control.handoff_manager import HandoffManager
from app.control.lifecycle_manager import LifecycleManager
from app.control.monitoring_policy import MonitoringPolicy
from app.control.ptz_client import DahuaPtzClient, PtzCommandResult
from app.control.ptz_runtime_state import PtzScheduleResult
from app.control.ptz_scheduler import PtzScheduler
from app.control.smoothing import EmaSmoother
from app.control.zoom_logic import ZoomController
from app.detection.yolo_detector import Detector
from app.logging_config import get_logger
from app.models.runtime import (
    ControlDecision,
    ControlMode,
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
    """Coordinates RTSP ingestion, detection, tracking, recovery, and PTZ scheduling."""

    ACTIVE_CONTROL_PHASES = {
        TrackingPhase.CENTERING,
        TrackingPhase.ZOOMING_FOR_HANDOFF,
        TrackingPhase.TRACKING,  # legacy compatibility phase
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
        self._ptz_scheduler = PtzScheduler(ptz_client)
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
        self._last_command_outcome: dict[str, object] = {}
        self._runtime_started = False
        self._last_visual_frame: np.ndarray | None = None
        self._last_frame_index: int | None = None
        self._fatal_shutdown_pending = False

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
        base_read_timeout = min(1.0, max(0.05, self._loop_regulator.target_period_seconds or 0.05))
        packet = None
        detections: list[Detection] = []
        target_state: TargetState | None = None
        decision: ControlDecision | None = None
        try:
            while not self._stop_event.is_set():
                loop_started_at = time.monotonic()
                self._finalize_due_pulses(loop_started_at)
                read_timeout = self._read_timeout_for_scheduler(base_read_timeout, loop_started_at)
                packet = self._reader.read(timeout=read_timeout)
                now = time.monotonic()
                self._finalize_due_pulses(now)
                if packet is None:
                    self._record_loop_timing(loop_started_at)
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

                control_now = time.monotonic()
                frame_age = max(0.0, control_now - packet.timestamp)
                target_state.frame_age_seconds = frame_age
                target_state.stale_frame = frame_age > self._config.tracking.stale_frame.max_age_seconds
                target_state.prediction_confidence = self._tracker.target_memory.prediction_confidence
                target_state.centered_frames = self._tracker.target_memory.centered_frames
                self._tracker.target_memory.stale_frame_age_seconds = frame_age
                self._metrics.frame_age_ms.observe(frame_age * 1000.0)
                if target_state.stale_frame and self._skip_log_debouncer.allow("stale_frame_warning", control_now):
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
                    control_now,
                    frame.shape[1],
                    frame.shape[0],
                    len(detections),
                )
                self._execute_decision(frame, target_state, decision, control_now)
                self._publish_snapshot(
                    frame,
                    packet.frame_index,
                    packet.timestamp,
                    detections,
                    target_state,
                    decision,
                    latency_ms,
                    packet.source_fps,
                    control_now,
                )
                self._previous_target = self._copy_target_state(target_state)
                self._record_loop_timing(loop_started_at)
                self._sleep_for_tick(loop_started_at)
        except Exception as exc:
            self._tracking_phase = TrackingPhase.ERROR
            self._handle_fatal_exception(exc, packet, detections, target_state, decision)
            self._shutdown_scheduler(time.monotonic())
            self._ptz.emergency_stop()
            raise

    def _read_timeout_for_scheduler(self, base_timeout: float, now: float) -> float:
        until_stop = self._ptz_scheduler.seconds_until_due_stop(now)
        if until_stop is None:
            return base_timeout
        return max(0.01, min(base_timeout, until_stop))

    def _finalize_due_pulses(self, now: float) -> None:
        for result in self._ptz_scheduler.tick(now):
            self._record_scheduler_result(result, now, None)
        self._update_ptz_runtime_metrics(now)

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
        visible_candidate_count: int = 0,
    ) -> None:
        memory = self._tracker.target_memory
        tight_zoom_detected = self._is_tight_zoom_detected(current, frame_height)
        loss_cause = self._classify_loss_cause(current, memory, frame_width, frame_height, tight_zoom_detected)
        current.tight_zoom_detected = tight_zoom_detected
        current.recovery_settle_ticks_remaining = memory.recovery_settle_ticks_remaining
        current.loss_cause = loss_cause
        memory.tight_zoom_detected = tight_zoom_detected
        memory.loss_cause = loss_cause
        handoff_ready, handoff_reason = self._handoff_manager.evaluate(current, memory, frame_width, frame_height)
        current.handoff_ready = handoff_ready
        current.centered_frames = memory.centered_frames
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
            visible_candidate_count=visible_candidate_count,
            tight_zoom_detected=tight_zoom_detected,
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
            memory.return_home_pending = False
            memory.recovery_settle_ticks_remaining = 0
            memory.recovery_zoom_steps = 0
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
            self._issue_stop(now, "target_lost")

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
        policy = self._lifecycle_manager.policy_for(self._tracking_phase)
        if self._tracking_phase in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING, TrackingPhase.TEMP_LOST, TrackingPhase.OCCLUDED}:
            self._issue_stop(now, self._tracking_phase.value)
            if self._tracking_phase in {TrackingPhase.HANDOFF, TrackingPhase.MONITORING} and memory.handoff_ts is None:
                memory.handoff_ts = now
            return

        if self._tracking_phase == TrackingPhase.RECOVERY_ZOOM_OUT:
            self._execute_recovery_zoom_out(target_state, now)
            return

        if self._tracking_phase == TrackingPhase.RECOVERY_LOCAL:
            if target_state.frame_age_seconds > self._config.tracking.stale_frame.aggressive_recovery_max_age_seconds:
                self._log_local_skip("stale_frame_blocks_local_recovery", target_state, now)
                self._metrics.stale_frame_suppressed_action_count.inc()
                return
            self._execute_local_recovery(frame, target_state, now, frame_width, frame_height)
            return

        if self._tracking_phase == TrackingPhase.RECOVERY_WIDE:
            if self._maybe_zoom_out_for_recovery(now):
                return
            if self._maybe_return_to_recovery_preset(now):
                return
            self._apply_lost_behavior(now)
            return

        if self._tracking_phase == TrackingPhase.RETURNING_HOME:
            memory.return_home_pending = False
            return

        if self._tracking_phase == TrackingPhase.LOST:
            self._apply_lost_behavior(now)
            if memory.last_confirmed_ts > 0:
                memory_age = now - memory.last_confirmed_ts
                if memory_age >= self._config.tracking.target_memory.clear_after_seconds:
                    self._metrics.recovery_failure_count.inc()
                    self._tracker.clear_target_memory()
            return

        if not policy.pan_tilt_allowed and not policy.zoom_allowed:
            self._last_skip_reason = f"phase:{self._tracking_phase.value}"
            return

        memory.recovery_zoom_steps = 0

    def _execute_recovery_zoom_out(
        self,
        target_state: TargetState,
        now: float,
    ) -> None:
        memory = self._tracker.target_memory
        if target_state.frame_age_seconds > self._config.tracking.stale_frame.aggressive_recovery_max_age_seconds:
            self._log_local_skip("stale_frame_blocks_recovery_zoom_out", target_state, now)
            self._metrics.stale_frame_suppressed_action_count.inc()
            return
        if memory.recovery_settle_ticks_remaining > 0:
            memory.recovery_settle_ticks_remaining -= 1
            target_state.recovery_settle_ticks_remaining = memory.recovery_settle_ticks_remaining
            self._log_local_skip("recovery_zoom_out_settling", target_state, now)
            return
        if not memory.tight_zoom_detected or memory.recovery_zoom_steps >= self._config.tracking.recovery.max_recovery_zoom_steps:
            self._log_local_skip("recovery_zoom_out_not_needed", target_state, now)
            return

        intent = PtzIntent(
            kind=PtzIntentKind.ZOOM,
            direction=PtzDirection.ZOOM_OUT,
            pulse_ms=self._config.tracking.recovery.recovery_zoom_out_step_pulse_ms,
            reason="recovery_zoom_out_stage",
            control_mode=ControlMode.RECOVERY,
            allow_interrupt=True,
            priority=2,
        )
        result = self._submit_intent(intent, now, move_cooldown=False, zoom_cooldown=False)
        if result.succeeded:
            if memory.recovery_zoom_steps == 0 and self._loss_started_at is not None:
                self._metrics.loss_to_first_zoomout_seconds.observe(max(0.0, now - self._loss_started_at))
            memory.recovery_zoom_steps += 1
            memory.recovery_settle_ticks_remaining = self._config.tracking.recovery.recovery_zoom_out_settle_ticks
            target_state.recovery_settle_ticks_remaining = memory.recovery_settle_ticks_remaining
            self._recovery_zoom_cooldown.mark(now)
            self._metrics.recovery_zoom_out_step_count.inc()

    def _maybe_return_to_recovery_preset(self, now: float) -> bool:
        recovery = self._config.tracking.recovery
        if not recovery.recovery_return_home_enabled or self._return_home_issued:
            return False
        if self._loss_started_at is None:
            return False
        if now - self._loss_started_at < recovery.recovery_return_home_timeout_seconds:
            return False

        self._issue_stop(now, "recovery_return_preset")
        result, preset_name = self._issue_recovery_return_preset()
        logger.info(
            "tracking_recovery_return_preset",
            success=result.success,
            detail=result.detail,
            preset=preset_name,
        )
        if not result.success:
            return False
        self._return_home_issued = True
        self._metrics.return_home_after_loss_count.inc()
        self._tracking_phase = TrackingPhase.RETURNING_HOME
        self._last_ptz_action = "return_home"
        self._tracker.clear_target_memory()
        self._tracker.target_memory.return_home_pending = True
        self._tracker.target_memory.lifecycle_state = TrackingPhase.RETURNING_HOME
        self._previous_target = TargetState(track_id=None, bbox_xyxy=None)
        return True

    def _issue_recovery_return_preset(self) -> tuple[PtzCommandResult, str | None]:
        recovery = self._config.tracking.recovery
        preset_name = recovery.recovery_return_preset_name
        if preset_name:
            return self._ptz.move_to_preset(preset_name, purpose="recovery_return"), preset_name
        if self._config.camera.home_preset_name:
            return self._ptz.move_home(), self._config.camera.home_preset_name
        if recovery.recovery_return_to_startup_preset_if_home_missing and self._config.camera.startup_preset_name:
            return self._ptz.startup_preset(), self._config.camera.startup_preset_name
        return (
            PtzCommandResult(
                success=False,
                action="preset",
                direction="recovery_return",
                pulse_ms=0,
                dry_run=self._ptz.is_dry_run,
                issued=False,
                detail="recovery_return_preset_not_configured",
                accepted=False,
            ),
            None,
        )

    def _maybe_zoom_out_for_recovery(self, now: float) -> bool:
        memory = self._tracker.target_memory
        if (
            memory.recovery_zoom_steps >= self._config.tracking.recovery.max_recovery_zoom_steps
            or not self._recovery_zoom_cooldown.ready(now)
        ):
            return False
        if (
            memory.last_zoom_ratio < self._config.tracking.recovery.tight_zoom_height_ratio_threshold
            and memory.recovery_zoom_steps == 0
        ):
            return False
        intent = PtzIntent(
            kind=PtzIntentKind.ZOOM,
            direction=PtzDirection.ZOOM_OUT,
            pulse_ms=self._config.tracking.recovery.recovery_zoom_out_step_pulse_ms,
            reason="recovery_zoom_out",
            control_mode=ControlMode.RECOVERY,
            allow_interrupt=True,
            priority=2,
        )
        result = self._submit_intent(intent, now, move_cooldown=False, zoom_cooldown=False)
        if result.succeeded:
            memory.recovery_zoom_steps += 1
            self._recovery_zoom_cooldown.mark(now)
            return True
        return False

    def _execute_local_recovery(
        self,
        frame: np.ndarray,
        target_state: TargetState,
        now: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
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
            prediction_confidence=target_state.prediction_confidence,
        )
        decision = self._control_logic.decide(synthetic, frame_width, frame_height)
        if decision.move_direction is None:
            self._log_local_skip("local_recovery_prediction_inside_dead_zone", target_state, now)
            return
        intent = PtzIntent(
            kind=PtzIntentKind.MOVE,
            direction=decision.move_direction,
            pulse_ms=decision.move_pulse_ms,
            reason=f"recovery:{decision.reason}",
            control_mode=ControlMode.RECOVERY,
            allow_interrupt=True,
            priority=2,
            predicted=decision.prediction_used,
        )
        result = self._submit_intent(intent, now, move_cooldown=True, zoom_cooldown=False)
        if result.succeeded:
            self._save_action_screenshot(frame, f"recovery-{decision.move_direction.value}", now)

    def _apply_lost_behavior(self, now: float) -> None:
        behavior = self._config.control.lost_behavior
        loss_age = now - self._loss_started_at if self._loss_started_at is not None else 0.0

        if behavior in {"zoom_out", "return_home"} and self._config.control.lost_zoom_out_enabled:
            if self._lost_zoom_cooldown.ready(now):
                result = self._submit_intent(
                    PtzIntent(
                        kind=PtzIntentKind.ZOOM,
                        direction=PtzDirection.ZOOM_OUT,
                        pulse_ms=self._config.control.zoom_pulse_ms,
                        reason="lost_zoom_out",
                        control_mode=ControlMode.RECOVERY,
                        allow_interrupt=True,
                        priority=1,
                    ),
                    now,
                    move_cooldown=False,
                    zoom_cooldown=False,
                )
                if result.succeeded:
                    self._lost_zoom_cooldown.mark(now)

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
        self._record_control_mode_metrics(decision)
        if decision.prediction_used:
            self._metrics.prediction_used_count.inc()

        skip_reason = self._decision_skip_reason(target_state, decision, now)
        if skip_reason is not None:
            self._last_skip_reason = skip_reason
            if skip_reason.startswith("stale_frame"):
                self._metrics.stale_frame_suppressed_action_count.inc()
            if self._skip_log_debouncer.allow(skip_reason, now):
                logger.info("ptz_action_skipped", reason=skip_reason, phase=self._tracking_phase.value)
            return

        if decision.move_direction is not None:
            result = self._submit_intent(
                PtzIntent(
                    kind=PtzIntentKind.MOVE,
                    direction=decision.move_direction,
                    pulse_ms=decision.move_pulse_ms,
                    reason=decision.reason,
                    control_mode=decision.control_mode,
                    allow_interrupt=decision.control_mode == ControlMode.COARSE_ALIGN,
                    priority=2 if decision.control_mode == ControlMode.COARSE_ALIGN else 1,
                    predicted=decision.prediction_used,
                ),
                now,
                move_cooldown=True,
                zoom_cooldown=False,
            )
            if result.succeeded:
                self._tracker.target_memory.last_ptz_action = decision.move_direction.value
                self._save_action_screenshot(frame, decision.move_direction.value, now)

        if decision.zoom_direction is not None:
            if decision.move_direction is not None and not self._config.control.allow_zoom_during_pan_tilt:
                self._log_local_skip("zoom_blocked_pan_tilt_active", target_state, now)
            else:
                result = self._submit_intent(
                    PtzIntent(
                        kind=PtzIntentKind.ZOOM,
                        direction=decision.zoom_direction,
                        pulse_ms=decision.zoom_pulse_ms,
                        reason=decision.reason,
                        control_mode=decision.control_mode,
                        allow_interrupt=False,
                        priority=1,
                    ),
                    now,
                    move_cooldown=False,
                    zoom_cooldown=True,
                )
                if result.succeeded:
                    self._tracker.target_memory.last_ptz_action = decision.zoom_direction.value
                    self._save_action_screenshot(frame, decision.zoom_direction.value, now)

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
        if decision.stale_frame_policy_state == "blocked":
            return "stale_frame_blocked"
        if decision.move_direction is None and decision.zoom_direction is None:
            return f"no_action:{decision.reason}"
        return None

    def _submit_intent(
        self,
        intent: PtzIntent,
        now: float,
        *,
        move_cooldown: bool,
        zoom_cooldown: bool,
    ) -> PtzScheduleResult:
        scheduler_state = self._ptz_scheduler.state
        same_active = scheduler_state.pulse_active and scheduler_state.active_direction == (intent.direction.value if intent.direction else None)
        if move_cooldown and not same_active and not self._move_cooldown.ready(now):
            return self._record_skipped_intent(intent, "movement_cooldown")
        if zoom_cooldown and not same_active and not self._zoom_cooldown.ready(now):
            return self._record_skipped_intent(intent, "zoom_cooldown")
        if not same_active and not self._rate_limiter.allow(now):
            return self._record_skipped_intent(intent, "rate_limited")

        result = self._ptz_scheduler.submit(intent, now)
        self._record_scheduler_result(result, now, intent)
        if result.cooldown_eligible and intent.kind == PtzIntentKind.MOVE:
            self._move_cooldown.mark(now)
        if result.cooldown_eligible and intent.kind == PtzIntentKind.ZOOM:
            self._zoom_cooldown.mark(now)
        return result

    def _record_skipped_intent(self, intent: PtzIntent, reason: str) -> PtzScheduleResult:
        result = PtzScheduleResult(
            kind=intent.kind.value,
            direction=intent.direction.value if intent.direction else None,
            skipped=True,
            detail=reason,
        )
        self._record_scheduler_result(result, time.monotonic(), intent)
        return result

    def _issue_stop(self, now: float, reason: str) -> None:
        if not self._ptz_scheduler.state.pulse_active:
            return
        result = self._ptz_scheduler.force_stop(now, reason=reason)
        self._record_scheduler_result(result, now, None)

    def _record_scheduler_result(
        self,
        result: PtzScheduleResult,
        now: float,
        intent: PtzIntent | None,
    ) -> None:
        self._last_command_outcome = self._ptz_scheduler.state.last_command_outcome or {
            "kind": result.kind,
            "direction": result.direction,
            "detail": result.detail,
            "skipped": result.skipped,
            "succeeded": result.succeeded,
        }
        if result.attempted:
            self._metrics.ptz_command_attempt_count.inc()
        if result.kind == PtzIntentKind.MOVE.value:
            if result.attempted:
                self._metrics.ptz_move_attempt_count.inc()
        if result.kind == PtzIntentKind.ZOOM.value:
            if result.attempted:
                self._metrics.ptz_zoom_attempt_count.inc()
        accepted_without_issue = result.accepted and not result.succeeded and not result.partial_failure and not result.skipped
        if result.skipped or accepted_without_issue:
            self._metrics.ptz_command_skipped_count.inc()
            self._last_skip_reason = result.detail
        if result.interrupted:
            self._metrics.ptz_command_interrupted_count.inc()
        if result.partial_failure:
            self._metrics.ptz_partial_failure_count.inc()
        if result.succeeded:
            self._metrics.ptz_command_success_count.inc()
            if result.kind == PtzIntentKind.MOVE.value:
                self._metrics.ptz_move_success_count.inc()
                self._metrics.ptz_commands.inc()
            elif result.kind == PtzIntentKind.ZOOM.value:
                self._metrics.ptz_zoom_success_count.inc()
                self._metrics.ptz_commands.inc()
        elif result.attempted and not result.skipped and not accepted_without_issue:
            self._metrics.ptz_command_failure_count.inc()

        self._last_ptz_action = self._ptz_scheduler.state.active_direction
        if result.succeeded:
            self._last_skip_reason = None
        if result.kind == "stop" and not self._ptz_scheduler.state.pulse_active:
            self._last_ptz_action = None
        logger.info(
            "ptz_scheduler_result",
            kind=result.kind,
            direction=result.direction,
            attempted=result.attempted,
            issued=result.issued,
            accepted=result.accepted,
            succeeded=result.succeeded,
            partial_failure=result.partial_failure,
            skipped=result.skipped,
            interrupted=result.interrupted,
            detail=result.detail,
            active_direction=self._ptz_scheduler.state.active_direction,
        )

    def _record_control_mode_metrics(self, decision: ControlDecision) -> None:
        if decision.control_mode == ControlMode.COARSE_ALIGN:
            self._metrics.coarse_align_count.inc()
        elif decision.control_mode == ControlMode.FINE_ALIGN:
            self._metrics.fine_align_count.inc()
        elif decision.control_mode == ControlMode.HOLD_STABLE:
            self._metrics.stable_hold_count.inc()

    def _update_ptz_runtime_metrics(self, now: float) -> None:
        state = self._ptz_scheduler.state
        if state.pulse_active and state.pulse_started_ts is not None and state.pulse_due_stop_ts is not None:
            self._metrics.active_pulse_duration_ms.set((state.pulse_due_stop_ts - state.pulse_started_ts) * 1000.0)
        else:
            self._metrics.active_pulse_duration_ms.set(0.0)

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
        now: float,
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
            ptz_runtime={
                "active_ptz_direction": self._ptz_scheduler.state.active_direction,
                "pulse_active": self._ptz_scheduler.state.pulse_active,
                "pulse_due_stop_ts": self._ptz_scheduler.state.pulse_due_stop_ts,
                "pulse_started_ts": self._ptz_scheduler.state.pulse_started_ts,
                "stop_pending": self._ptz_scheduler.state.stop_pending,
            },
            last_command_outcome=self._last_command_outcome,
            return_home_enabled=bool(
                self._config.tracking.recovery.recovery_return_preset_name
                or self._config.camera.home_preset_name
                or (
                    self._config.tracking.recovery.recovery_return_to_startup_preset_if_home_missing
                    and self._config.camera.startup_preset_name
                )
            ),
            return_home_issued=self._return_home_issued,
            extras={
                "fps": source_fps,
                "loss_started_at": self._loss_started_at,
                "loss_age_seconds": max(0.0, now - self._loss_started_at) if self._loss_started_at is not None else 0.0,
                "return_home_issued": self._return_home_issued,
                "return_home_pending": self._tracker.target_memory.return_home_pending,
                "prediction_confidence": self._tracker.target_memory.prediction_confidence,
                "recovery_zoom_steps": self._tracker.target_memory.recovery_zoom_steps,
                "settle_ticks_remaining": self._tracker.target_memory.recovery_settle_ticks_remaining,
                "tight_zoom_detected": self._tracker.target_memory.tight_zoom_detected,
                "recovery_stage": self._tracking_phase.value,
                "loss_cause": self._tracker.target_memory.loss_cause,
                "memory_track_id": self._tracker.target_memory.track_id,
                "average_frame_age_ms": round(target_state.frame_age_seconds * 1000.0, 2),
                "current_control_mode": decision.control_mode.value,
                "stale_frame_policy_state": decision.stale_frame_policy_state,
            },
        )
        self._state_store.set_snapshot(snapshot)
        self._last_frame_index = frame_index
        if self._config.app.overlay:
            overlay = draw_overlay(frame, snapshot, self._config.control)
            self._last_visual_frame = overlay.copy()
            if self._config.app.debug_window:
                cv2.imshow("ptz-autotrack", overlay)
                cv2.waitKey(1)
        else:
            self._last_visual_frame = frame.copy()
        if (
            self._config.snapshots.periodic_debug_frame_seconds > 0
            and timestamp - self._last_snapshot_ts >= self._config.snapshots.periodic_debug_frame_seconds
        ):
            self._snapshot_manager.save(frame, "periodic", timestamp)
            self._last_snapshot_ts = timestamp

    def _handle_fatal_exception(
        self,
        exc: Exception,
        packet: object | None,
        detections: list[Detection],
        target_state: TargetState | None,
        decision: ControlDecision | None,
    ) -> None:
        self._fatal_shutdown_pending = True
        active_target = target_state or self._tracker.state or self._previous_target
        crash_snapshot_path = self._save_crash_snapshot(packet)
        logger.exception(
            "tracking_service_exception",
            error_type=type(exc).__name__,
            error=str(exc),
            phase=self._tracking_phase.value,
            frame_index=getattr(packet, "frame_index", self._last_frame_index),
            detection_count=len(detections),
            track_id=active_target.track_id,
            target_visible=active_target.visible,
            target_stable=active_target.stable,
            target_status=active_target.status.value,
            selection_reason=active_target.selection_reason,
            target_bbox=active_target.bbox_xyxy,
            predicted_center=active_target.predicted_center,
            missing_frames=active_target.missing_frames,
            match_breakdown=active_target.match_breakdown,
            decision_reason=decision.reason if decision is not None else None,
            control_mode=decision.control_mode.value if decision is not None else None,
            last_ptz_action=self._last_ptz_action,
            last_skip_reason=self._last_skip_reason,
            crash_snapshot_path=crash_snapshot_path,
        )
        if self._config.app.debug_window:
            logger.error(
                "tracking_service_debug_window_closing_after_exception",
                frame_index=getattr(packet, "frame_index", self._last_frame_index),
                crash_snapshot_path=crash_snapshot_path,
            )

    def _save_crash_snapshot(self, packet: object | None) -> str | None:
        frame = None
        if packet is not None:
            frame = getattr(packet, "frame", None)
        if frame is None:
            frame = self._last_visual_frame
        if frame is None:
            return None
        try:
            path = self._snapshot_manager.save(frame, "fatal-crash", time.time())
            return str(path)
        except Exception:
            logger.exception("tracking_service_fatal_snapshot_failed")
            return None

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
        if self._tracking_phase == TrackingPhase.RECOVERY_ZOOM_OUT and previous_phase != TrackingPhase.RECOVERY_ZOOM_OUT:
            self._metrics.recovery_zoom_out_count.inc()
            logger.info(
                "tracking_recovery_zoom_out_started",
                track_id=current.track_id,
                tight_zoom_detected=current.tight_zoom_detected,
                loss_cause=current.loss_cause,
            )
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
            TrackingPhase.RECOVERY_ZOOM_OUT,
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
        if previous_phase == TrackingPhase.RECOVERY_ZOOM_OUT and current.status == TrackStatus.TRACKING:
            self._metrics.recovery_zoom_out_success_count.inc()
        if previous_phase == TrackingPhase.RECOVERY_ZOOM_OUT and self._tracking_phase in {
            TrackingPhase.RECOVERY_LOCAL,
            TrackingPhase.RECOVERY_WIDE,
            TrackingPhase.RETURNING_HOME,
            TrackingPhase.LOST,
        }:
            self._metrics.recovery_zoom_out_abort_count.inc()

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
            centered_frames=target.centered_frames,
            frame_age_seconds=target.frame_age_seconds,
            stale_frame=target.stale_frame,
            prediction_confidence=target.prediction_confidence,
            tight_zoom_detected=target.tight_zoom_detected,
            recovery_settle_ticks_remaining=target.recovery_settle_ticks_remaining,
            loss_cause=target.loss_cause,
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

    def _shutdown_scheduler(self, now: float) -> None:
        result = self._ptz_scheduler.shutdown(now)
        self._record_scheduler_result(result, now, None)

    def _shutdown_runtime(self) -> None:
        if not self._runtime_started:
            return
        self._tracking_phase = TrackingPhase.IDLE
        logger.info(
            "tracking_service_stopping",
            reason="fatal_exception" if self._fatal_shutdown_pending else "normal_shutdown",
        )
        self._reader.stop()
        self._shutdown_scheduler(time.monotonic())
        self._ptz.emergency_stop()
        self._runtime_started = False
        self._fatal_shutdown_pending = False
        if self._config.app.debug_window:
            cv2.destroyAllWindows()

    def _run_worker(self) -> None:
        try:
            self.run_loop()
        except Exception:
            logger.exception("tracking_worker_exited_with_error")
        finally:
            self._shutdown_runtime()

    def _record_loop_timing(self, started_at: float) -> None:
        elapsed = (time.monotonic() - started_at) * 1000.0
        self._metrics.control_loop_elapsed_ms.observe(elapsed)

    def _sleep_for_tick(self, started_at: float) -> None:
        elapsed = self._loop_regulator.elapsed_since(started_at)
        remaining = self._loop_regulator.remaining_sleep(elapsed)
        due_stop = self._ptz_scheduler.seconds_until_due_stop(time.monotonic())
        if due_stop is not None:
            remaining = min(remaining, due_stop)
        if remaining > 0:
            time.sleep(remaining)
        if remaining <= 0 and elapsed > self._loop_regulator.target_period_seconds:
            self._metrics.control_loop_overrun_count.inc()
            logger.debug(
                "tracking_loop_overrun",
                elapsed_seconds=round(elapsed, 4),
                target_period_seconds=round(self._loop_regulator.target_period_seconds, 4),
            )

    def _is_tight_zoom_detected(self, current: TargetState, frame_height: int) -> bool:
        ratio = 0.0
        if current.visible and current.bbox_xyxy is not None:
            ratio = height_ratio(current.bbox_xyxy, frame_height)
        elif self._tracker.target_memory.last_zoom_ratio > 0:
            ratio = self._tracker.target_memory.last_zoom_ratio
        return ratio >= self._config.tracking.recovery.tight_zoom_height_ratio_threshold

    def _classify_loss_cause(
        self,
        current: TargetState,
        memory: TargetMemory,
        frame_width: int,
        frame_height: int,
        tight_zoom_detected: bool,
    ) -> str:
        if current.visible:
            return "visible"
        if current.stale_frame:
            return "stale_frame"
        if tight_zoom_detected:
            return "over_zoom"
        if memory.last_center is not None:
            nx = memory.last_center[0] / max(1.0, frame_width)
            ny = memory.last_center[1] / max(1.0, frame_height)
            if nx <= 0.12 or nx >= 0.88 or ny <= 0.12 or ny >= 0.88:
                return "off_frame_exit"
        if memory.likely_occluded or memory.consecutive_missing_frames <= self._config.tracking.recovery.missing_frame_count_occluded:
            return "occlusion"
        return "detector_miss"
