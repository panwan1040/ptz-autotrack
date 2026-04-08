from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TrackStatus(str, Enum):
    SEARCHING = "searching"
    TRACKING = "tracking"
    LOST = "lost"


class TrackingPhase(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    LOST = "lost"
    RETURNING_HOME = "returning_home"
    ERROR = "error"


class PtzDirection(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"
    UP = "Up"
    DOWN = "Down"
    LEFT_UP = "LeftUp"
    RIGHT_UP = "RightUp"
    LEFT_DOWN = "LeftDown"
    RIGHT_DOWN = "RightDown"
    ZOOM_IN = "ZoomTele"
    ZOOM_OUT = "ZoomWide"
    STOP = "Stop"


@dataclass(slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str
    tracker_id: int | None = None


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray
    timestamp: float
    source_fps: float
    frame_index: int


@dataclass(slots=True)
class TargetState:
    track_id: int | None
    bbox_xyxy: tuple[float, float, float, float] | None
    confidence: float = 0.0
    persist_frames: int = 0
    last_seen_ts: float = 0.0
    status: TrackStatus = TrackStatus.SEARCHING
    stable: bool = False
    visible: bool = False
    selection_reason: str = "idle"
    candidate_score: float = 0.0
    lost_duration_seconds: float = 0.0


def compatibility_status_for_phase(phase: TrackingPhase) -> TrackStatus:
    if phase == TrackingPhase.TRACKING:
        return TrackStatus.TRACKING
    if phase in {TrackingPhase.LOST, TrackingPhase.RETURNING_HOME}:
        return TrackStatus.LOST
    return TrackStatus.SEARCHING


@dataclass(slots=True)
class ControlDecision:
    move_direction: PtzDirection | None = None
    move_pulse_ms: int = 0
    zoom_direction: PtzDirection | None = None
    zoom_pulse_ms: int = 0
    reason: str = "idle"
    normalized_error_x: float = 0.0
    normalized_error_y: float = 0.0
    target_height_ratio: float = 0.0


@dataclass(slots=True)
class TrackingSnapshot:
    frame_index: int
    timestamp: float
    tracking_phase: TrackingPhase = TrackingPhase.SEARCHING
    detections: list[Detection] = field(default_factory=list)
    target: TargetState = field(default_factory=lambda: TargetState(track_id=None, bbox_xyxy=None))
    decision: ControlDecision = field(default_factory=ControlDecision)
    inference_latency_ms: float = 0.0
    current_ptz_action: str | None = None
    last_skip_reason: str | None = None
    return_home_enabled: bool = False
    return_home_issued: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def target_to_detection(self) -> Detection:
        return Detection(
            bbox_xyxy=self.target.bbox_xyxy or (0.0, 0.0, 0.0, 0.0),
            confidence=self.target.confidence,
            class_name="person",
            tracker_id=self.target.track_id,
        )
