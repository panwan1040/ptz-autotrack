from __future__ import annotations

import cv2
import numpy as np

from app.config import ControlSection
from app.models.runtime import Detection, TrackingSnapshot
from app.utils.geometry import bbox_center


def draw_overlay(
    frame: np.ndarray,
    snapshot: TrackingSnapshot,
    control: ControlSection,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    cx, cy = width // 2, height // 2
    dead_x = int(control.dead_zone_x * width)
    dead_y = int(control.dead_zone_y * height)
    cv2.rectangle(output, (cx - dead_x, cy - dead_y), (cx + dead_x, cy + dead_y), (0, 255, 255), 1)
    cv2.drawMarker(output, (cx, cy), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20)
    for det in snapshot.detections:
        _draw_detection(output, det, (0, 255, 0))
    if snapshot.target.bbox_xyxy is not None:
        _draw_detection(output, snapshot.target_to_detection(), (0, 0, 255))
    if snapshot.target.predicted_window is not None:
        x1, y1, x2, y2 = [int(v) for v in snapshot.target.predicted_window]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 128, 0), 1)
    if snapshot.target.predicted_center is not None:
        px, py = [int(v) for v in snapshot.target.predicted_center]
        cv2.drawMarker(output, (px, py), (255, 128, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16)
    if snapshot.target.bbox_xyxy is not None:
        cx, cy = bbox_center(snapshot.target.bbox_xyxy)
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    text_lines = [
        f"phase={snapshot.tracking_phase.value}",
        f"status={snapshot.target.status.value}",
        f"target_id={snapshot.target.track_id}",
        f"stable={snapshot.target.stable}",
        f"visible={snapshot.target.visible} missing={snapshot.target.missing_frames}",
        f"handoff_ready={snapshot.target.handoff_ready}",
        f"ptz={snapshot.decision.move_direction.value if snapshot.decision.move_direction else 'idle'}",
        f"zoom={snapshot.decision.zoom_direction.value if snapshot.decision.zoom_direction else 'idle'}",
        f"reason={snapshot.target.selection_reason}",
        f"appearance={snapshot.target.appearance_similarity:.2f}",
        f"frame_age={snapshot.target.frame_age_seconds:.2f}s stale={snapshot.target.stale_frame}",
        f"fps={snapshot.extras.get('fps', 0):.1f}",
    ]
    for idx, line in enumerate(text_lines):
        cv2.putText(output, line, (10, 25 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return output


def _draw_detection(frame: np.ndarray, detection: Detection, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in detection.bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cx, cy = bbox_center(detection.bbox_xyxy)
    cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
    cv2.putText(
        frame,
        f"id={detection.tracker_id} conf={detection.confidence:.2f}",
        (x1, max(15, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )
