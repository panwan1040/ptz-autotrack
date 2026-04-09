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
    CANDIDATE_LOCK = "candidate_lock"
    CENTERING = "centering"
    ZOOMING_FOR_HANDOFF = "zooming_for_handoff"
    HANDOFF = "handoff"
    MONITORING = "monitoring"
    TEMP_LOST = "temp_lost"
    OCCLUDED = "occluded"
    RECOVERY_LOCAL = "recovery_local"
    RECOVERY_WIDE = "recovery_wide"
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
    predicted_center: tuple[float, float] | None = None
    predicted_window: tuple[float, float, float, float] | None = None
    appearance_similarity: float = 0.0
    missing_frames: int = 0
    visible_frames: int = 0
    handoff_ready: bool = False
    frame_age_seconds: float = 0.0
    stale_frame: bool = False
    match_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TargetMemory:
    track_id: int | None = None
    last_confirmed_ts: float = 0.0
    last_confirmed_bbox: tuple[float, float, float, float] | None = None
    last_smoothed_bbox: tuple[float, float, float, float] | None = None
    last_center: tuple[float, float] | None = None
    last_velocity: tuple[float, float] = (0.0, 0.0)
    last_direction: tuple[float, float] = (0.0, 0.0)
    bbox_size_ratio_trend: float = 0.0
    confidence_average: float = 0.0
    appearance_signature: list[float] | None = None
    last_ptz_action: str | None = None
    last_zoom_ratio: float = 0.0
    consecutive_visible_frames: int = 0
    consecutive_missing_frames: int = 0
    lifecycle_state: TrackingPhase = TrackingPhase.IDLE
    acquisition_ts: float = 0.0
    handoff_ts: float | None = None
    centered_frames: int = 0
    recovery_zoom_steps: int = 0
    predicted_center: tuple[float, float] | None = None
    predicted_window: tuple[float, float, float, float] | None = None
    prediction_confidence: float = 0.0
    appearance_similarity: float = 0.0
    last_match_score: float = 0.0
    recent_confidences: list[float] = field(default_factory=list)
    recent_centers: list[tuple[float, float]] = field(default_factory=list)
    recent_timestamps: list[float] = field(default_factory=list)
    missing_started_ts: float | None = None
    likely_occluded: bool = False
    stale_frame_age_seconds: float = 0.0


def compatibility_status_for_phase(phase: TrackingPhase) -> TrackStatus:
    if phase in {
        TrackingPhase.CENTERING,
        TrackingPhase.ZOOMING_FOR_HANDOFF,
        TrackingPhase.HANDOFF,
        TrackingPhase.MONITORING,
        TrackingPhase.TRACKING,
    }:
        return TrackStatus.TRACKING
    if phase in {
        TrackingPhase.TEMP_LOST,
        TrackingPhase.OCCLUDED,
        TrackingPhase.RECOVERY_LOCAL,
        TrackingPhase.RECOVERY_WIDE,
        TrackingPhase.LOST,
        TrackingPhase.RETURNING_HOME,
    }:
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
