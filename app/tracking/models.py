from __future__ import annotations

from dataclasses import dataclass

from app.models.runtime import Detection
from app.utils.geometry import BBox


@dataclass(slots=True)
class TrackRecord:
    track_id: int
    bbox_xyxy: BBox
    confidence: float
    created_ts: float
    last_seen_ts: float
    hits: int = 1
    visible_streak: int = 1
    missed_frames: int = 0
    age_frames: int = 1
    confirmed: bool = False
    visible: bool = True
    appearance_signature: list[float] | None = None

    def update(
        self,
        detection: Detection,
        now: float,
        min_persist_frames: int,
        appearance_signature: list[float] | None = None,
    ) -> None:
        self.bbox_xyxy = detection.bbox_xyxy
        self.confidence = detection.confidence
        self.last_seen_ts = now
        self.hits += 1
        self.visible_streak += 1
        self.missed_frames = 0
        self.age_frames += 1
        self.visible = True
        if appearance_signature is not None:
            self.appearance_signature = appearance_signature
        if self.visible_streak >= max(1, min_persist_frames):
            self.confirmed = True

    def mark_missed(self) -> None:
        self.visible = False
        self.visible_streak = 0
        self.missed_frames += 1
        self.age_frames += 1


@dataclass(slots=True)
class TrackCandidate:
    track_id: int
    bbox_xyxy: BBox
    confidence: float
    persist_frames: int
    total_visible_frames: int
    age_frames: int
    missed_frames: int
    last_seen_ts: float
    confirmed: bool
    visible: bool = True
    appearance_signature: list[float] | None = None
    match_breakdown: dict[str, float] | None = None

    @classmethod
    def from_record(cls, record: TrackRecord) -> "TrackCandidate":
        return cls(
            track_id=record.track_id,
            bbox_xyxy=record.bbox_xyxy,
            confidence=record.confidence,
            persist_frames=record.visible_streak,
            total_visible_frames=record.hits,
            age_frames=record.age_frames,
            missed_frames=record.missed_frames,
            last_seen_ts=record.last_seen_ts,
            confirmed=record.confirmed,
            visible=record.visible,
            appearance_signature=record.appearance_signature,
        )


@dataclass(slots=True)
class SelectionResult:
    candidate: TrackCandidate | None
    reason: str
    score: float = 0.0
    previous_score: float = 0.0
    switched: bool = False
