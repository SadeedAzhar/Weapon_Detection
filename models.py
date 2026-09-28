from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    track_id: Optional[int] = None
    source: str = "weapon"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "source": self.source,
        }


@dataclass
class PersonDetection:
    """Lightweight person detection returned by the advanced person detector."""
    track_id: Optional[int]
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    frame_width: int
    frame_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }

@dataclass
class WeaponResult:
    present: bool = False
    confidence: float = 0.0
    status: str = "clear"
    detections: list[Detection] = field(default_factory=list)
    top_classes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def best_detection(self) -> Optional[Detection]:
        if not self.detections:
            return None
        return max(self.detections, key=lambda item: item.confidence)

    def to_dict(self) -> dict[str, Any]:
        best = self.best_detection
        return {
            "present": self.present,
            "confidence": self.confidence,
            "status": self.status,
            "bbox_xyxy": best.bbox if best else None,
            "detections": [item.to_dict() for item in self.detections],
            "top_classes": self.top_classes,
        }

@dataclass
class PersonResult:
    bbox: tuple[int, int, int, int]
    confidence: float
    track_id: Optional[int] = None
    person_index: Optional[int] = None
    size_category: Optional[str] = None
    height_ratio: float = 0.0
    roi_bbox: Optional[tuple[int, int, int, int]] = None
    roi_scale_factor: float = 1.0
    roi_status: str = "ready"
    weapon: WeaponResult = field(default_factory=WeaponResult)
    temporal: Optional[dict[str, Any]] = None
    event_id: Optional[int | str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": self.bbox,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "person_index": self.person_index,
            "size_category": self.size_category,
            "height_ratio": self.height_ratio,
            "roi_bbox": self.roi_bbox,
            "roi_scale_factor": self.roi_scale_factor,
            "roi_status": self.roi_status,
            "weapon": self.weapon.to_dict(),
            "temporal": self.temporal,
            "event_id": self.event_id,
        }

@dataclass
class ImageAnalysisResult:
    image: Any
    annotated_image: Any
    persons: list[PersonResult] = field(default_factory=list)
    weapon_results: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class VideoAnalysisResult:
    frames: list[Any] = field(default_factory=list)
    annotated_frames: list[Any] = field(default_factory=list)
    frame_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EventView:
    event_id: int | str
    timestamp: float | str
    track_id: int
    status: str
    weapon_class: Optional[str] = None
    confidence: float = 0.0
    positive_frames: int = 0
    observed_frames: int = 0
    frame_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "status": self.status,
            "weapon_class": self.weapon_class,
            "confidence": self.confidence,
            "positive_frames": self.positive_frames,
            "observed_frames": self.observed_frames,
            "frame_index": self.frame_index,
        }

@dataclass
class ModelInfo:
    weapon_model_path: str
    person_model_path: str
    weapon_model_loaded: bool = False
    person_model_loaded: bool = False
    device: Optional[str] = None

@dataclass
class RuntimeSettings:
    weapon_confidence: float
    person_confidence: float
    top_k: int
    roi_expansion: float
    roi_mode: str = "original"
    sample_every_n: int = 1
    temporal_window: int = 5
    temporal_required_frames: int = 3
