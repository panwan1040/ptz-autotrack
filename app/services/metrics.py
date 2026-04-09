from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, generate_latest


@dataclass(slots=True)
class MetricsRegistry:
    frames_received: Counter = Counter("ptz_frames_received_total", "Frames received")
    frames_processed: Counter = Counter("ptz_frames_processed_total", "Frames processed")
    ptz_commands: Counter = Counter("ptz_commands_total", "PTZ commands issued")
    target_lost: Counter = Counter("ptz_target_lost_total", "Target lost events")
    occlusion_count: Counter = Counter("ptz_occlusion_total", "Occlusion events")
    local_recovery_count: Counter = Counter("ptz_local_recovery_total", "Local recovery entries")
    wide_recovery_count: Counter = Counter("ptz_wide_recovery_total", "Wide recovery entries")
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
    time_to_reacquire_seconds: Histogram = Histogram(
        "ptz_time_to_reacquire_seconds",
        "Time from target loss to reacquisition in seconds",
    )
    tracking_status: Gauge = Gauge("ptz_tracking_status", "0=searching,1=tracking,2=lost")
    tracking_phase: Gauge = Gauge(
        "ptz_tracking_phase",
        "Lifecycle phase encoded as an integer enum index",
    )

    def render(self) -> bytes:
        return generate_latest()
