from __future__ import annotations

from math import sqrt


BBox = tuple[float, float, float, float]


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_width(bbox: BBox) -> float:
    x1, _y1, x2, _y2 = bbox
    return max(0.0, x2 - x1)


def bbox_height(bbox: BBox) -> float:
    _x1, y1, _x2, y2 = bbox
    return max(0.0, y2 - y1)


def bbox_area(bbox: BBox) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def normalize_point(x: float, y: float, frame_width: float, frame_height: float) -> tuple[float, float]:
    return (x / frame_width, y / frame_height)


def normalized_bbox_center(bbox: BBox, frame_width: float, frame_height: float) -> tuple[float, float]:
    cx, cy = bbox_center(bbox)
    return normalize_point(cx, cy, frame_width, frame_height)


def height_ratio(bbox: BBox, frame_height: float) -> float:
    return bbox_height(bbox) / frame_height if frame_height > 0 else 0.0


def center_distance_normalized(a: BBox, b: BBox, frame_width: float, frame_height: float) -> float:
    ax, ay = normalized_bbox_center(a, frame_width, frame_height)
    bx, by = normalized_bbox_center(b, frame_width, frame_height)
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def inside_dead_zone(
    normalized_x: float,
    normalized_y: float,
    dead_zone_x: float,
    dead_zone_y: float,
) -> bool:
    return abs(normalized_x - 0.5) <= dead_zone_x and abs(normalized_y - 0.5) <= dead_zone_y
