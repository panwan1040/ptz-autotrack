import numpy as np

from app.config import TrackingSection
from app.models.runtime import TargetMemory
from app.tracking.appearance_extractor import AppearanceExtractor
from app.tracking.models import TrackCandidate
from app.tracking.target_matcher import TargetMatcher


def test_appearance_extractor_similarity_tracks_same_color_patch() -> None:
    extractor = AppearanceExtractor(TrackingSection().appearance)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[10:40, 10:30] = (0, 0, 255)
    frame[10:40, 40:60] = (0, 0, 255)
    frame[45:75, 10:30] = (0, 255, 0)

    red_a = extractor.extract(frame, (10, 10, 30, 40))
    red_b = extractor.extract(frame, (40, 10, 60, 40))
    green = extractor.extract(frame, (10, 45, 30, 75))

    assert extractor.similarity(red_a, red_b) > extractor.similarity(red_a, green)


def test_target_matcher_prefers_candidate_near_prediction_and_appearance() -> None:
    tracking = TrackingSection()
    extractor = AppearanceExtractor(tracking.appearance)
    matcher = TargetMatcher(tracking, extractor)
    memory = TargetMemory(
        track_id=5,
        predicted_center=(200.0, 180.0),
        last_center=(190.0, 170.0),
        last_direction=(10.0, 8.0),
        last_confirmed_bbox=(150.0, 100.0, 240.0, 280.0),
        appearance_signature=[1.0, 0.0, 0.0],
    )
    strong = TrackCandidate(
        track_id=5,
        bbox_xyxy=(160.0, 110.0, 250.0, 290.0),
        confidence=0.86,
        persist_frames=5,
        total_visible_frames=5,
        age_frames=5,
        missed_frames=0,
        last_seen_ts=1.0,
        confirmed=True,
        appearance_signature=[1.0, 0.0, 0.0],
    )
    weak = TrackCandidate(
        track_id=6,
        bbox_xyxy=(500.0, 110.0, 620.0, 310.0),
        confidence=0.90,
        persist_frames=5,
        total_visible_frames=5,
        age_frames=5,
        missed_frames=0,
        last_seen_ts=1.0,
        confirmed=True,
        appearance_signature=[0.0, 1.0, 0.0],
    )

    strong_score, _ = matcher.score(strong, memory, 960, 540)
    weak_score, _ = matcher.score(weak, memory, 960, 540)

    assert strong_score > weak_score
