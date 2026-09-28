from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import (
    DEFAULT_ROI_EXPANSION,
    MIN_PERSON_HEIGHT_PX,
    MIN_PERSON_HEIGHT_RATIO,
    PERSON_SIZE_LARGE,
    PERSON_SIZE_MEDIUM,
    PERSON_SIZE_SMALL,
)

@dataclass
class ROIResult:
    status: str
    size_category: str
    person_bbox: tuple[int, int, int, int]
    roi_bbox: Optional[tuple[int, int, int, int]]
    roi: Optional[np.ndarray]
    scale_factor: float
    height_ratio: float

def classify_person_size(height_ratio: float) -> str:
    if height_ratio >= PERSON_SIZE_LARGE:
        return "large"
    if height_ratio >= PERSON_SIZE_MEDIUM:
        return "medium"
    if height_ratio >= PERSON_SIZE_SMALL:
        return "small"
    return "very small"

def has_sufficient_resolution(person_height: int, height_ratio: float) -> bool:
    return (
        person_height >= MIN_PERSON_HEIGHT_PX
        and height_ratio >= MIN_PERSON_HEIGHT_RATIO
    )

def clip_bbox(bbox, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), frame_width - 1))
    y1 = max(0, min(int(round(y1)), frame_height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), frame_width))
    y2 = max(y1 + 1, min(int(round(y2)), frame_height))
    return x1, y1, x2, y2

def get_expansion_ratio(
    size_category: str,
    base_expansion: float = DEFAULT_ROI_EXPANSION,
) -> float:
    if size_category in {"large", "medium"}:
        return base_expansion
    if size_category == "small":
        return max(base_expansion, 0.25)
    return max(base_expansion, 0.30)

def expand_person_bbox(
    person_bbox,
    frame_width: int,
    frame_height: int,
    size_category: str,
    base_expansion: float = DEFAULT_ROI_EXPANSION,
):
    x1, y1, x2, y2 = person_bbox
    width = x2 - x1
    height = y2 - y1
    expansion = get_expansion_ratio(size_category, base_expansion)
    pad_x = width * expansion
    pad_y = height * expansion
    return clip_bbox(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        frame_width,
        frame_height,
    )

def get_upscale_factor(
    person_height: int,
    target_person_height: int = 160,
    max_scale: float = 4.0,
) -> float:
    if person_height <= 0:
        return 1.0
    scale = target_person_height / person_height
    return float(max(1.0, min(scale, max_scale)))

def crop_roi(frame: np.ndarray, roi_bbox) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Cannot crop an empty frame.")
    x1, y1, x2, y2 = roi_bbox
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        raise ValueError("ROI crop is empty.")
    return roi

def upscale_roi(roi: np.ndarray, scale_factor: float) -> np.ndarray:
    if roi is None or roi.size == 0:
        raise ValueError("Cannot upscale an empty ROI.")
    if scale_factor <= 1.0:
        return roi
    height, width = roi.shape[:2]
    return cv2.resize(
        roi,
        (
            max(1, int(round(width * scale_factor))),
            max(1, int(round(height * scale_factor))),
        ),
        interpolation=cv2.INTER_CUBIC,
    )

def prepare_person_roi(
    frame: np.ndarray,
    person_bbox,
    base_expansion: float = DEFAULT_ROI_EXPANSION,
    roi_mode: str = "original",
) -> ROIResult:
    """Prepare the ROI; original resolution is the default, upscaling is optional."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot process an empty frame.")
    if roi_mode not in {"original", "upscaled"}:
        raise ValueError("roi_mode must be 'original' or 'upscaled'.")

    frame_height, frame_width = frame.shape[:2]
    person_bbox = clip_bbox(person_bbox, frame_width, frame_height)
    person_height = max(0, person_bbox[3] - person_bbox[1])
    height_ratio = person_height / max(frame_height, 1)
    size_category = classify_person_size(height_ratio)
    sufficient = has_sufficient_resolution(person_height, height_ratio)

    if not sufficient and roi_mode == "original":
        return ROIResult(
            status="insufficient_resolution",
            size_category=size_category,
            person_bbox=person_bbox,
            roi_bbox=None,
            roi=None,
            scale_factor=1.0,
            height_ratio=height_ratio,
        )

    roi_bbox = expand_person_bbox(
        person_bbox,
        frame_width=frame_width,
        frame_height=frame_height,
        size_category=size_category,
        base_expansion=base_expansion,
    )
    roi = crop_roi(frame, roi_bbox)
    scale_factor = get_upscale_factor(person_height) if roi_mode == "upscaled" else 1.0
    roi = upscale_roi(roi, scale_factor)

    if not sufficient:
        status = "low_resolution_upscaled"
    elif roi_mode == "upscaled":
        status = "upscaled"
    else:
        status = "standard"

    return ROIResult(
        status=status,
        size_category=size_category,
        person_bbox=person_bbox,
        roi_bbox=roi_bbox,
        roi=roi,
        scale_factor=scale_factor,
        height_ratio=height_ratio,
    )
