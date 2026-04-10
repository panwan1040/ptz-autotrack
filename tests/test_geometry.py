import pytest

from app.tracking.models import TrackCandidate
from app.utils.geometry import (
    bbox_area,
    bbox_center,
    bbox_iou,
    height_ratio,
    inside_dead_zone,
    normalized_bbox_center,
    validate_bbox_input,
)


def test_bbox_center() -> None:
    assert bbox_center((10, 20, 30, 40)) == (20.0, 30.0)


def test_inside_dead_zone() -> None:
    assert inside_dead_zone(0.5, 0.5, 0.1, 0.1) is True
    assert inside_dead_zone(0.7, 0.5, 0.1, 0.1) is False


def test_height_ratio() -> None:
    assert height_ratio((0, 0, 50, 100), 200) == 0.5


def test_bbox_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_validate_bbox_input_accepts_valid_bbox_list() -> None:
    assert validate_bbox_input("bbox_center", [1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)


def test_bbox_center_rejects_track_candidate_with_clear_type_error() -> None:
    candidate = TrackCandidate(
        track_id=1,
        bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
        confidence=0.9,
        persist_frames=3,
        total_visible_frames=3,
        age_frames=3,
        missed_frames=0,
        last_seen_ts=1.0,
        confirmed=True,
    )
    with pytest.raises(TypeError, match="bbox_center expected bbox tuple\\[4\\], got TrackCandidate"):
        bbox_center(candidate)


def test_bbox_area_rejects_invalid_object_with_clear_type_error() -> None:
    with pytest.raises(TypeError, match="bbox_area expected bbox tuple\\[4\\], got dict"):
        bbox_area({"x1": 0, "y1": 0, "x2": 10, "y2": 10})


def test_normalized_bbox_center_rejects_invalid_object_with_clear_type_error() -> None:
    with pytest.raises(TypeError, match="normalized_bbox_center expected bbox tuple\\[4\\], got str"):
        normalized_bbox_center("not-a-bbox", 1920, 1080)
