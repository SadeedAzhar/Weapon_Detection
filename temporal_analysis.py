from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from config import TEMPORAL_REQUIRED_FRAMES, TEMPORAL_WINDOW

@dataclass
class TemporalState:
    track_id: int
    observations: deque
    confirmed: bool = False
    last_status: str = "unknown"

class TemporalAnalyzer:
    """Keep weapon evidence separate from inconclusive frames."""

    def __init__(
        self,
        window_size: int = TEMPORAL_WINDOW,
        required_frames: int = TEMPORAL_REQUIRED_FRAMES,
    ):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if required_frames < 1 or required_frames > window_size:
            raise ValueError("required_frames must be between 1 and window_size.")
        self.window_size = window_size
        self.required_frames = required_frames
        self._states: dict[int, TemporalState] = {}

    def reset(self) -> None:
        self._states.clear()

    def _get_state(self, track_id: int) -> TemporalState:
        if track_id not in self._states:
            self._states[track_id] = TemporalState(
                track_id=track_id,
                observations=deque(maxlen=self.window_size),
            )
        return self._states[track_id]

    @staticmethod
    def _best_class(top_classes: list[dict]) -> Optional[str]:
        if not top_classes:
            return None
        return top_classes[0].get("class")

    @staticmethod
    def _best_class_confidence(top_classes: list[dict]) -> float:
        if not top_classes:
            return 0.0
        return float(top_classes[0].get("confidence", 0.0))

    @staticmethod
    def _observation_status(weapon_result: dict) -> str:
        status = weapon_result.get("status", "clear")
        if status in {"insufficient_resolution", "low_resolution_upscaled"}:
            return "unusable"
        return "positive" if weapon_result.get("present", False) else "negative"

    def update(self, track_id: int, weapon_result: dict) -> dict:
        state = self._get_state(track_id)
        observation_status = self._observation_status(weapon_result)
        top_classes = weapon_result.get("top_classes", [])
        observation = {
            "status": observation_status,
            "confidence": float(weapon_result.get("confidence", 0.0)),
            "class": self._best_class(top_classes),
            "class_confidence": self._best_class_confidence(top_classes),
        }
        state.observations.append(observation)

        observations = list(state.observations)
        positive = [item for item in observations if item["status"] == "positive"]
        negative = [item for item in observations if item["status"] == "negative"]
        unusable = [item for item in observations if item["status"] == "unusable"]

        positive_count = len(positive)
        negative_count = len(negative)
        usable_count = positive_count + negative_count
        confirmed = positive_count >= self.required_frames

        if confirmed:
            state.last_status = "confirmed"
        elif positive_count > 0:
            state.last_status = "possible"
        elif usable_count > 0:
            state.last_status = "clear"
        else:
            state.last_status = "inconclusive"

        class_votes = defaultdict(float)
        for item in positive:
            if item["class"]:
                class_votes[item["class"]] += item["class_confidence"]

        likely_class = max(class_votes, key=class_votes.get) if class_votes else None
        state.confirmed = confirmed

        return {
            "track_id": track_id,
            "status": state.last_status,
            "confirmed": confirmed,
            "positive_frames": positive_count,
            "negative_frames": negative_count,
            "unusable_frames": len(unusable),
            "observed_frames": usable_count,
            "window_frames": len(observations),
            "required_frames": self.required_frames,
            "confirmation_ratio": (
                positive_count / usable_count if usable_count else 0.0
            ),
            "likely_class": likely_class,
            "latest_confidence": float(weapon_result.get("confidence", 0.0)),
            "latest_top_classes": top_classes,
        }

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def get_state(self, track_id: int) -> Optional[dict]:
        state = self._states.get(track_id)
        if state is None:
            return None

        observations = list(state.observations)
        positive = sum(1 for item in observations if item["status"] == "positive")
        negative = sum(1 for item in observations if item["status"] == "negative")
        unusable = sum(1 for item in observations if item["status"] == "unusable")

        return {
            "track_id": track_id,
            "confirmed": state.confirmed,
            "status": state.last_status,
            "positive_frames": positive,
            "negative_frames": negative,
            "unusable_frames": unusable,
            "observed_frames": positive + negative,
            "required_frames": self.required_frames,
        }

    def get_all_states(self) -> list[dict]:
        return [
            self.get_state(track_id)
            for track_id in self._states
        ]
