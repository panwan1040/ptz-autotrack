from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, generate_latest


@dataclass(slots=True)
class MetricsRegistry:
    frames_received: Counter = Counter("ptz_frames_received_total", "Frames received")
    frames_processed: Counter = Counter("ptz_frames_processed_total", "Frames processed")
    ptz_commands: Counter = Counter("ptz_commands_total", "PTZ commands issued")
    ptz_command_attempt_count: Counter = Counter("ptz_command_attempt_total", "PTZ command attempts")
    ptz_command_success_count: Counter = Counter("ptz_command_success_total", "PTZ command successes")
    ptz_command_failure_count: Counter = Counter("ptz_command_failure_total", "PTZ command failures")
    ptz_command_skipped_count: Counter = Counter("ptz_command_skipped_total", "PTZ command skips")
    ptz_command_interrupted_count: Counter = Counter("ptz_command_interrupted_total", "PTZ command interruptions")
    ptz_partial_failure_count: Counter = Counter("ptz_partial_failure_total", "PTZ partial failures")
    ptz_move_attempt_count: Counter = Counter("ptz_move_attempt_total", "PTZ move attempts")
    ptz_move_success_count: Counter = Counter("ptz_move_success_total", "PTZ move successes")
    ptz_zoom_attempt_count: Counter = Counter("ptz_zoom_attempt_total", "PTZ zoom attempts")
    ptz_zoom_success_count: Counter = Counter("ptz_zoom_success_total", "PTZ zoom successes")
    target_lost: Counter = Counter("ptz_target_lost_total", "Target lost events")
    occlusion_count: Counter = Counter("ptz_occlusion_total", "Occlusion events")
    recovery_zoom_out_count: Counter = Counter("ptz_recovery_zoom_out_total", "Recovery zoom-out phase entries")
    recovery_zoom_out_step_count: Counter = Counter("ptz_recovery_zoom_out_step_total", "Recovery zoom-out steps")
    recovery_zoom_out_success_count: Counter = Counter(
        "ptz_recovery_zoom_out_success_total",
        "Recovery zoom-out phases that led to reacquisition",
    )
    recovery_zoom_out_abort_count: Counter = Counter(
        "ptz_recovery_zoom_out_abort_total",
        "Recovery zoom-out phases exited without reacquisition",
    )
    local_recovery_count: Counter = Counter("ptz_local_recovery_total", "Local recovery entries")
    wide_recovery_count: Counter = Counter("ptz_wide_recovery_total", "Wide recovery entries")
    return_home_after_loss_count: Counter = Counter(
        "ptz_return_home_after_loss_total",
        "Return-to-preset actions issued after failed recovery",
    )
    successful_reacquisition_count: Counter = Counter(
        "ptz_successful_reacquisition_total",
        "Successful target reacquisitions",
    )
    target_switch_count: Counter = Counter("ptz_target_switch_total", "Target switch events")
    handoff_count: Counter = Counter("ptz_handoff_total", "Handoff attempts")
    handoff_break_count: Counter = Counter("ptz_handoff_break_total", "Handoff break events")
    recovery_failure_count: Counter = Counter("ptz_recovery_failure_total", "Recovery failures")
    reconnects: Counter = Counter("ptz_reconnects_total", "RTSP reconnect events")
    inference_latency_ms: Histogram = Histogram(
        "ptz_inference_latency_ms", "YOLO inference latency in ms"
    )
    control_loop_elapsed_ms: Histogram = Histogram("ptz_control_loop_elapsed_ms", "Control loop elapsed time in ms")
    control_loop_overrun_count: Counter = Counter("ptz_control_loop_overrun_total", "Control loop overruns")
    frame_age_ms: Histogram = Histogram("ptz_frame_age_ms", "Frame age in ms at decision time")
    time_to_reacquire_seconds: Histogram = Histogram(
        "ptz_time_to_reacquire_seconds",
        "Time from target loss to reacquisition in seconds",
    )
    loss_to_first_zoomout_seconds: Histogram = Histogram(
        "ptz_loss_to_first_zoomout_seconds",
        "Time from target loss to first recovery zoom-out step in seconds",
    )
    active_pulse_duration_ms: Gauge = Gauge("ptz_active_pulse_duration_ms", "Current active pulse duration in ms")
    prediction_used_count: Counter = Counter("ptz_prediction_used_total", "Predictive control decisions")
    stale_frame_suppressed_action_count: Counter = Counter(
        "ptz_stale_frame_suppressed_action_total",
        "Actions suppressed because of stale frames",
    )
    fine_align_count: Counter = Counter("ptz_fine_align_total", "Fine alignment decisions")
    coarse_align_count: Counter = Counter("ptz_coarse_align_total", "Coarse alignment decisions")
    stable_hold_count: Counter = Counter("ptz_stable_hold_total", "Stable hold decisions")
    tracking_status: Gauge = Gauge("ptz_tracking_status", "0=searching,1=tracking,2=lost")
    tracking_phase: Gauge = Gauge(
        "ptz_tracking_phase",
        "Lifecycle phase encoded as an integer enum index",
    )

    def render(self) -> bytes:
        return generate_latest()
