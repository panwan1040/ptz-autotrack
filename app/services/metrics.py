from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, generate_latest


@dataclass(slots=True)
class MetricsRegistry:
    frames_received: Counter = Counter("ptz_frames_received_total", "Frames received")
    frames_processed: Counter = Counter("ptz_frames_processed_total", "Frames processed")
    ptz_commands: Counter = Counter("ptz_commands_total", "PTZ commands issued")
    target_lost: Counter = Counter("ptz_target_lost_total", "Target lost events")
    reconnects: Counter = Counter("ptz_reconnects_total", "RTSP reconnect events")
    inference_latency_ms: Histogram = Histogram(
        "ptz_inference_latency_ms", "YOLO inference latency in ms"
    )
    tracking_status: Gauge = Gauge("ptz_tracking_status", "0=searching,1=tracking,2=lost")
    tracking_phase: Gauge = Gauge(
        "ptz_tracking_phase",
        "0=idle,1=searching,2=acquiring,3=tracking,4=lost,5=returning_home,6=error",
    )

    def render(self) -> bytes:
        return generate_latest()
