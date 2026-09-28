from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_DEVICE, DEFAULT_IMAGE_SIZE, WEAPON_CLASSES

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ModelLoadError(RuntimeError):
    pass

class InferenceError(RuntimeError):
    pass

@lru_cache(maxsize=4)
def load_model(model_path: str):
    if YOLO is None:
        raise ModelLoadError("Ultralytics is not installed. Run: pip install ultralytics")
    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")
    return YOLO(str(path))

class YOLODetector:
    def __init__(self, model_path: str | Path, device=DEFAULT_DEVICE):
        self.model_path = Path(model_path)
        self.device = device
        self.model = load_model(str(self.model_path.resolve()))

    def predict(self, image: np.ndarray, conf: float, imgsz: int = DEFAULT_IMAGE_SIZE):
        if image is None or image.size == 0:
            raise InferenceError("Cannot run inference on an empty image.")
        kwargs = {
            "source": image,
            "conf": conf,
            "imgsz": imgsz,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        try:
            return self.model.predict(**kwargs)
        except Exception as e:
            raise InferenceError(
                f"Inference failed for {self.model_path.name}: {e}"
            ) from e

def get_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id)).strip().lower()
    return str(names[int(class_id)]).strip().lower()

def clip_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be positive.")
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return x1, y1, x2, y2

def bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1), max(0, y2 - y1)

def extract_person_detections(result, frame_width: int, frame_height: int):
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for bbox, conf, class_id in zip(boxes, confs, classes):
        if get_class_name(result.names, class_id) != "person":
            continue
        bbox = clip_bbox(bbox, frame_width, frame_height)
        width, height = bbox_size(bbox)
        detections.append({
            "confidence": float(conf),
            "bbox_xyxy": bbox,
            "width": width,
            "height": height,
            "height_ratio": height / max(frame_height, 1),
        })

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return detections

def extract_weapon_predictions(
    result,
    roi_frame_bbox,
    frame_width: int,
    frame_height: int,
    top_k: int = 3,
    roi_scale_factor: float = 1.0,
):
    empty = {
        "present": False,
        "confidence": 0.0,
        "status": "clear",
        "top_classes": [],
        "detections": [],
        "bbox_xyxy": None,
    }

    if result.boxes is None or len(result.boxes) == 0:
        return empty

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    candidates = []

    # Convert the model output back to the original ROI, then to the full frame.
    for bbox, conf, class_id in zip(boxes, confs, classes):
        class_name = get_class_name(result.names, class_id)
        if class_name not in WEAPON_CLASSES:
            continue

        frame_bbox = roi_bbox_to_frame_bbox(
            bbox,
            roi_frame_bbox,
            frame_width,
            frame_height,
            roi_scale_factor=roi_scale_factor,
        )
        candidates.append({
            "class": class_name,
            "confidence": float(conf),
            "bbox_xyxy": frame_bbox,
        })

    if not candidates:
        return empty

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    best_per_class = {}

    for item in candidates:
        name = item["class"]
        if name not in best_per_class or item["confidence"] > best_per_class[name]["confidence"]:
            best_per_class[name] = item

    top_classes = sorted(
        best_per_class.values(),
        key=lambda x: x["confidence"],
        reverse=True,
    )[:max(1, top_k)]

    return {
        "present": True,
        "confidence": candidates[0]["confidence"],
        "status": "detected",
        "top_classes": top_classes,
        "detections": candidates,
        "bbox_xyxy": candidates[0]["bbox_xyxy"],
    }

def roi_bbox_to_frame_bbox(
    roi_bbox,
    roi_frame_bbox,
    frame_width: int,
    frame_height: int,
    roi_scale_factor: float = 1.0,
):
    rx1, ry1, _, _ = roi_frame_bbox
    x1, y1, x2, y2 = roi_bbox
    scale = max(float(roi_scale_factor), 1.0)

    # Undo optional ROI upscaling before applying the ROI offset.
    x1 /= scale
    y1 /= scale
    x2 /= scale
    y2 /= scale

    roi_width = roi_frame_bbox[2] - roi_frame_bbox[0]
    roi_height = roi_frame_bbox[3] - roi_frame_bbox[1]
    return clip_bbox(
        (
            rx1 + x1,
            ry1 + y1,
            rx1 + x2,
            ry1 + y2,
        ),
        frame_width,
        frame_height,
    )

def draw_bbox(
    image: np.ndarray,
    bbox,
    label: str,
    color=(0, 180, 255),
    thickness: int = 2,
):
    output = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        output,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return output

def draw_overlays(image: np.ndarray, persons: list[dict], weapon_results: list[dict]):
    output = image.copy()

    for index, person in enumerate(persons, start=1):
        label = f"Person #{index} | {person['confidence']:.0%}"
        output = draw_bbox(output, person["bbox_xyxy"], label, color=(255, 255, 0))

    for weapon in weapon_results:
        if not weapon.get("present") or weapon.get("bbox_xyxy") is None:
            continue
        top_classes = weapon.get("top_classes", [])
        class_name = top_classes[0]["class"] if top_classes else "weapon"
        label = f"{class_name} | {weapon['confidence']:.0%}"
        output = draw_bbox(
            output,
            weapon["bbox_xyxy"],
            label,
            color=(0, 0, 255),
            thickness=3,
        )

    return output

def validate_weapon_classes(model_names):
    names = list(model_names.values()) if isinstance(model_names, dict) else list(model_names)
    normalized = [str(name).strip().casefold() for name in names]
    expected = [str(name).strip().casefold() for name in WEAPON_CLASSES]
    valid = len(normalized) == len(expected) and all(
        found == expected_name
        for found, expected_name in zip(normalized, expected)
    )
    return {
        "expected": expected,
        "found": normalized,
        "valid": valid,
    }
