from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from config import MAX_EVENT_HISTORY

@dataclass
class DetectionEvent:
    event_id: int
    timestamp: float
    track_id: Optional[int]
    status: str
    weapon_class: Optional[str]
    confidence: float
    positive_frames: int
    observed_frames: int
    frame_index: Optional[int] = None

    @property
    def label(self) -> str:
        return self.weapon_class.title() if self.weapon_class else "Unknown"

    def to_dict(self) -> dict:
        return asdict(self)

class EventManager:
    """Store confirmed events and prevent duplicate alerts for active tracks."""

    def __init__(self, max_events: int = MAX_EVENT_HISTORY):
        if max_events < 1:
            raise ValueError("max_events must be at least 1.")
        self.max_events = max_events
        self._events: list[DetectionEvent] = []
        self._next_id = 1
        self._active_tracks: set[int] = set()

    @property
    def events(self) -> list[DetectionEvent]:
        return list(self._events)

    def reset(self) -> None:
        self._events.clear()
        self._next_id = 1
        self._active_tracks.clear()

    def create_event(
        self,
        temporal_result: dict,
        timestamp: float,
        frame_index: Optional[int] = None,
    ) -> Optional[DetectionEvent]:
        if not temporal_result.get("confirmed", False):
            return None

        track_id = temporal_result.get("track_id")
        if track_id is not None and track_id in self._active_tracks:
            return None

        event = DetectionEvent(
            event_id=self._next_id,
            timestamp=float(timestamp),
            track_id=track_id,
            status="confirmed",
            weapon_class=temporal_result.get("likely_class"),
            confidence=float(temporal_result.get("latest_confidence", 0.0)),
            positive_frames=int(temporal_result.get("positive_frames", 0)),
            observed_frames=int(temporal_result.get("observed_frames", 0)),
            frame_index=frame_index,
        )
        self._next_id += 1
        self._events.insert(0, event)

        if track_id is not None:
            self._active_tracks.add(track_id)

        if len(self._events) > self.max_events:
            self._events = self._events[:self.max_events]
        return event

    def close_track(self, track_id: int) -> None:
        self._active_tracks.discard(track_id)

    def close_tracks(self, track_ids: set[int] | list[int]) -> None:
        for track_id in track_ids:
            self.close_track(track_id)

    def close_all_tracks(self) -> None:
        self._active_tracks.clear()

    def get_recent_events(self, limit: int = 10) -> list[DetectionEvent]:
        if limit < 1:
            return []
        return self._events[:limit]

    def get_event(self, event_id: int) -> Optional[DetectionEvent]:
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None

    def summary(self) -> dict:
        confirmed = [
            event for event in self._events
            if event.status == "confirmed"
        ]
        weapon_classes = {}
        for event in confirmed:
            label = event.weapon_class or "unknown"
            weapon_classes[label] = weapon_classes.get(label, 0) + 1

        return {
            "total_events": len(self._events),
            "confirmed_alerts": len(confirmed),
            "weapon_classes": weapon_classes,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
