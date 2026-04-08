from app.config import TrackingSection
from app.models.runtime import Detection, TargetState
from app.tracking.target_selector import TargetSelector


def test_select_largest() -> None:
    selector = TargetSelector(TrackingSection(strategy="largest"))
    detections = [
        Detection((0, 0, 50, 100), 0.7, "person", 1),
        Detection((0, 0, 80, 200), 0.6, "person", 2),
    ]
    state = selector.select(detections, TargetState(track_id=None, bbox_xyxy=None), 960, 540, 1.0)
    assert state.track_id == 2


def test_stick_to_previous_target() -> None:
    selector = TargetSelector(TrackingSection(strategy="stick_nearest", max_association_distance=0.2))
    prev = TargetState(track_id=10, bbox_xyxy=(100, 100, 200, 300), last_seen_ts=1.0)
    detections = [
        Detection((105, 110, 205, 310), 0.8, "person", 10),
        Detection((700, 100, 800, 300), 0.95, "person", 11),
    ]
    state = selector.select(detections, prev, 960, 540, 2.0)
    assert state.track_id == 10
