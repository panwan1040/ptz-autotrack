from app.config import TrackingSection
from app.models.runtime import Detection, TrackStatus
from app.tracking.tracker import Tracker


def test_tracker_keeps_stable_id_across_small_motion() -> None:
    tracker = Tracker(TrackingSection(min_persist_frames=2, lost_timeout_seconds=1.0))
    first = tracker.update([Detection((100, 100, 200, 300), 0.8, "person")], 960, 540, 1.0)
    second = tracker.update([Detection((108, 102, 208, 302), 0.82, "person")], 960, 540, 1.1)
    third = tracker.update([Detection((116, 110, 216, 310), 0.85, "person")], 960, 540, 1.2)

    assert first.track_id is not None
    assert second.track_id == first.track_id
    assert third.track_id == first.track_id
    assert third.status == TrackStatus.TRACKING
    assert third.stable is True


def test_tracker_marks_target_lost_then_reacquires_same_track() -> None:
    tracker = Tracker(TrackingSection(min_persist_frames=1, lost_timeout_seconds=0.8))
    acquired = tracker.update([Detection((200, 80, 320, 360), 0.9, "person")], 960, 540, 1.0)
    lost = tracker.update([], 960, 540, 1.4)
    reacquired = tracker.update([Detection((210, 85, 330, 365), 0.88, "person")], 960, 540, 1.6)

    assert acquired.status == TrackStatus.TRACKING
    assert lost.status == TrackStatus.LOST
    assert reacquired.status == TrackStatus.TRACKING
    assert reacquired.track_id == acquired.track_id
