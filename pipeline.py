import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import (
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_ROI_EXPANSION,
    DEFAULT_SAMPLE_EVERY_N,
    DEFAULT_TOP_K,
    PERSON_MODEL_PATH,
    WEAPON_MODEL_PATH,
)
from events import EventManager
from inference import YOLODetector, extract_person_detections, extract_weapon_predictions
from models import ImageAnalysisResult, PersonDetection, VideoAnalysisResult
from roi_processor import prepare_person_roi
from temporal_analysis import TemporalAnalyzer
from tracking import PersonTracker
from video_processor import VideoProcessor, get_video_metadata_from_bytes, save_uploaded_video


def _decode_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Uploaded file is not a valid image.")
    return frame


def _load_models(
    person_model_path=PERSON_MODEL_PATH,
    weapon_model_path=WEAPON_MODEL_PATH,
    device=DEFAULT_DEVICE,
):
    person_detector = YOLODetector(person_model_path, device=device)
    weapon_detector = YOLODetector(weapon_model_path, device=device)
    return person_detector, weapon_detector


def _correct_upscaled_bbox(
    bbox,
    roi_bbox,
    scale_factor,
    frame_width,
    frame_height,
):
    """Undo ROI upscaling before keeping the detection in full-frame coordinates."""
    if scale_factor <= 1.0:
        return bbox

    rx1, ry1, _, _ = roi_bbox
    x1, y1, x2, y2 = bbox

    corrected = (
        rx1 + (x1 - rx1) / scale_factor,
        ry1 + (y1 - ry1) / scale_factor,
        rx1 + (x2 - rx1) / scale_factor,
        ry1 + (y2 - ry1) / scale_factor,
    )

    x1, y1, x2, y2 = corrected
    x1 = max(0, min(int(round(x1)), frame_width - 1))
    y1 = max(0, min(int(round(y1)), frame_height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), frame_width))
    y2 = max(y1 + 1, min(int(round(y2)), frame_height))
    return x1, y1, x2, y2


def _empty_weapon_result(status="clear"):
    return {
        "present": False,
        "confidence": 0.0,
        "top_classes": [],
        "bbox_xyxy": None,
        "status": status,
    }


def analyze_weapon_only_frame(
    frame_bgr,
    weapon_detector,
    confidence_threshold=0.50,
    top_k=DEFAULT_TOP_K,
):
    """Run the weapon detector directly on the full frame.

    This is the default/primary path. It deliberately does not run the person
    detector or ROI preparation. Person-aware analysis is an advanced mode.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Cannot analyze an empty frame.")

    frame_height, frame_width = frame_bgr.shape[:2]
    outputs = weapon_detector.predict(
        frame_bgr,
        conf=confidence_threshold,
        imgsz=DEFAULT_IMAGE_SIZE,
    )

    if outputs:
        weapon = extract_weapon_predictions(
            outputs[0],
            roi_frame_bbox=(0, 0, frame_width, frame_height),
            frame_width=frame_width,
            frame_height=frame_height,
            top_k=top_k,
        )
    else:
        weapon = _empty_weapon_result()

    return weapon


def _draw_weapon_only_result(frame_bgr, weapon):
    """Draw only weapon evidence; no person/ROI boxes."""
    output = frame_bgr.copy()
    if not weapon or not weapon.get("present"):
        return output

    bbox = weapon.get("bbox_xyxy")
    if bbox is None:
        return output

    x1, y1, x2, y2 = bbox
    top_classes = weapon.get("top_classes", [])
    label = "Weapon"
    if top_classes:
        label = f"{top_classes[0]['class']} | {weapon['confidence']:.0%}"

    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        output,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def analyze_frame(
    frame_bgr,
    person_detector,
    weapon_detector,
    confidence_threshold=0.50,
    person_confidence=0.35,
    top_k=DEFAULT_TOP_K,
    roi_expansion=DEFAULT_ROI_EXPANSION,
):
    """Run person-first weapon inference on one original full-frame image."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Cannot analyze an empty frame.")

    frame_height, frame_width = frame_bgr.shape[:2]
    person_outputs = person_detector.predict(
        frame_bgr,
        conf=person_confidence,
        imgsz=DEFAULT_IMAGE_SIZE,
    )

    if not person_outputs:
        return [], frame_bgr.copy()

    raw_persons = extract_person_detections(
        person_outputs[0],
        frame_width=frame_width,
        frame_height=frame_height,
    )

    people = []
    annotated = frame_bgr.copy()

    for person_index, person in enumerate(raw_persons, start=1):
        bbox = person["bbox_xyxy"]
        roi_result = prepare_person_roi(
            frame_bgr,
            bbox,
            base_expansion=roi_expansion,
        )

        person_record = {
            **person,
            "track_id": None,
            "person_index": person_index,
            "roi_status": roi_result.status,
            "roi_bbox": roi_result.roi_bbox,
            "roi_scale_factor": roi_result.scale_factor,
            "size_category": roi_result.size_category,
        }

        if roi_result.roi is None:
            person_record["weapon"] = _empty_weapon_result(roi_result.status)
            people.append(person_record)
            continue

        weapon_outputs = weapon_detector.predict(
            roi_result.roi,
            conf=confidence_threshold,
            imgsz=DEFAULT_IMAGE_SIZE,
        )

        if weapon_outputs:
            weapon_result = extract_weapon_predictions(
                weapon_outputs[0],
                roi_frame_bbox=roi_result.roi_bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                top_k=top_k,
            )
        else:
            weapon_result = _empty_weapon_result()

        if weapon_result.get("bbox_xyxy") is not None:
            weapon_result["bbox_xyxy"] = _correct_upscaled_bbox(
                weapon_result["bbox_xyxy"],
                roi_result.roi_bbox,
                roi_result.scale_factor,
                frame_width,
                frame_height,
            )

        for prediction in weapon_result.get("top_classes", []):
            if prediction.get("bbox_xyxy") is not None:
                prediction["bbox_xyxy"] = _correct_upscaled_bbox(
                    prediction["bbox_xyxy"],
                    roi_result.roi_bbox,
                    roi_result.scale_factor,
                    frame_width,
                    frame_height,
                )

        person_record["weapon"] = weapon_result
        people.append(person_record)

    return people, annotated


def _draw_results(frame_bgr, people):
    """Draw person, ROI and weapon information on the original frame."""
    output = frame_bgr.copy()

    for index, person in enumerate(people, start=1):
        x1, y1, x2, y2 = person["bbox_xyxy"]
        track_id = person.get("track_id")
        person_label = f"Person #{track_id or index} | {person['confidence']:.0%}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            output,
            person_label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        roi_bbox = person.get("roi_bbox")
        if roi_bbox is not None:
            rx1, ry1, rx2, ry2 = roi_bbox
            cv2.rectangle(
                output,
                (rx1, ry1),
                (rx2, ry2),
                (180, 180, 180),
                1,
            )

        weapon = person.get("weapon", {})
        if not weapon.get("present"):
            continue

        weapon_bbox = weapon.get("bbox_xyxy")
        if weapon_bbox is None:
            continue

        wx1, wy1, wx2, wy2 = weapon_bbox
        top_classes = weapon.get("top_classes", [])
        label = "Weapon"
        if top_classes:
            label = f"{top_classes[0]['class']} | {weapon['confidence']:.0%}"

        cv2.rectangle(
            output,
            (wx1, wy1),
            (wx2, wy2),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            output,
            label,
            (wx1, min(output.shape[0] - 10, wy2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def analyze_image(
    image_bytes,
    confidence_threshold=0.50,
    person_confidence=0.35,
    top_k=DEFAULT_TOP_K,
    roi_expansion=DEFAULT_ROI_EXPANSION,
    person_model_path=PERSON_MODEL_PATH,
    weapon_model_path=WEAPON_MODEL_PATH,
    device=DEFAULT_DEVICE,
    advanced_person_analysis=False,
):
    """Analyze one image. Weapon-only is the default; person-aware is advanced."""
    frame_bgr = _decode_image(image_bytes)

    if advanced_person_analysis:
        person_detector, weapon_detector = _load_models(
            person_model_path,
            weapon_model_path,
            device=device,
        )
        people, _ = analyze_frame(
            frame_bgr,
            person_detector,
            weapon_detector,
            confidence_threshold=confidence_threshold,
            person_confidence=person_confidence,
            top_k=top_k,
            roi_expansion=roi_expansion,
        )
        annotated = _draw_results(frame_bgr, people)
        weapon_results = [person["weapon"] for person in people]
        weapon_count = sum(1 for weapon in weapon_results if weapon.get("present"))
    else:
        weapon_detector = YOLODetector(weapon_model_path, device=device)
        weapon = analyze_weapon_only_frame(
            frame_bgr,
            weapon_detector,
            confidence_threshold=confidence_threshold,
            top_k=top_k,
        )
        people = []
        weapon_results = [weapon]
        weapon_count = int(bool(weapon.get("present")))
        annotated = _draw_weapon_only_result(frame_bgr, weapon)

    summary = {
        "status": "Weapon detected" if weapon_count else "No weapon detected",
        "persons": len(people),
        "potential_weapons": weapon_count,
        "confirmed_alerts": 0,
        "resolution": f"{frame_bgr.shape[1]} × {frame_bgr.shape[0]}",
    }

    person_results = [
        PersonDetection(
            track_id=person.get("track_id"),
            confidence=person["confidence"],
            bbox_xyxy=person["bbox_xyxy"],
            frame_width=frame_bgr.shape[1],
            frame_height=frame_bgr.shape[0],
        )
        for person in people
    ]

    return ImageAnalysisResult(
        image=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        annotated_image=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        persons=person_results,
        weapon_results=weapon_results,
        summary=summary,
    )


def get_video_metadata(video_bytes: bytes):
    """Return video metadata using the shared video processor."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata = get_video_metadata_from_bytes(
            video_bytes,
            temp_dir=temp_dir,
        )

    return {
        "fps": metadata.fps,
        "frame_count": metadata.frame_count,
        "resolution": metadata.resolution,
        "duration": metadata.duration,
    }


def process_video(
    video_bytes,
    confidence_threshold=0.50,
    person_confidence=0.35,
    top_k=DEFAULT_TOP_K,
    roi_expansion=DEFAULT_ROI_EXPANSION,
    person_model_path=PERSON_MODEL_PATH,
    weapon_model_path=WEAPON_MODEL_PATH,
    sample_every_n=DEFAULT_SAMPLE_EVERY_N,
    max_frames=DEFAULT_MAX_VIDEO_FRAMES,
    device=DEFAULT_DEVICE,
    advanced_person_analysis=False,
):
    """Analyze sampled video frames. Person-aware analysis is opt-in."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = save_uploaded_video(
            video_bytes,
            temp_dir,
            filename="uploaded_video.mp4",
        )
        processor = VideoProcessor(video_path)
        metadata = processor.metadata()

        weapon_detector = YOLODetector(weapon_model_path, device=device)
        person_detector = None

        if advanced_person_analysis:
            person_detector = YOLODetector(person_model_path, device=device)

        tracker = PersonTracker() if advanced_person_analysis else None
        temporal = TemporalAnalyzer() if advanced_person_analysis else None
        events = EventManager() if advanced_person_analysis else None

        frames = []
        annotated_frames = []
        frame_results = []

        for frame_data in processor.frames(
            sample_every_n=sample_every_n,
            max_frames=max_frames,
        ):
            if advanced_person_analysis:
                people, _ = analyze_frame(
                    frame_data.frame,
                    person_detector,
                    weapon_detector,
                    confidence_threshold=confidence_threshold,
                    person_confidence=person_confidence,
                    top_k=top_k,
                    roi_expansion=roi_expansion,
                )

                tracked_people = tracker.update(people)
                temporal_results = []

                for person in tracked_people:
                    track_id = person["track_id"]
                    temporal_result = temporal.update(
                        track_id,
                        person["weapon"],
                    )
                    person["temporal"] = temporal_result
                    temporal_results.append(temporal_result)

                    event = events.create_event(
                        temporal_result,
                        timestamp=frame_data.timestamp,
                        frame_index=frame_data.frame_index,
                    )
                    if event is not None:
                        person["event_id"] = event.event_id

                annotated = _draw_results(frame_data.frame, tracked_people)

                for person in tracked_people:
                    temporal_result = person.get("temporal")
                    if not temporal_result or not temporal_result.get("confirmed"):
                        continue
                    x1, y1, _, _ = person["bbox_xyxy"]
                    track_id = person["track_id"]
                    label = (
                        f"CONFIRMED | Person #{track_id} | "
                        f"{temporal_result.get('likely_class') or 'weapon'}"
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1, min(annotated.shape[0] - 10, y1 + 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                frame_results.append(
                    {
                        "frame_index": frame_data.frame_index,
                        "timestamp": frame_data.timestamp,
                        "persons": tracked_people,
                        "temporal_results": temporal_results,
                        "weapon": _best_weapon_from_people(tracked_people),
                        "events": [
                            event.to_dict()
                            for event in events.get_recent_events()
                            if event.frame_index == frame_data.frame_index
                        ],
                    }
                )
            else:
                weapon = analyze_weapon_only_frame(
                    frame_data.frame,
                    weapon_detector,
                    confidence_threshold=confidence_threshold,
                    top_k=top_k,
                )
                annotated = _draw_weapon_only_result(frame_data.frame, weapon)
                frame_results.append(
                    {
                        "frame_index": frame_data.frame_index,
                        "timestamp": frame_data.timestamp,
                        "persons": [],
                        "temporal_results": [],
                        "weapon": weapon,
                        "events": [],
                    }
                )

            frames.append(cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB))
            annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        if advanced_person_analysis:
            events.close_all_tracks()
            event_list = [event.to_dict() for event in events.events]
            event_summary = events.summary()
        else:
            event_list = []
            detected_count = sum(
                1 for item in frame_results
                if item.get("weapon", {}).get("present")
            )
            event_summary = {
                "confirmed_alerts": 0,
                "weapon_detections": detected_count,
            }

    return VideoAnalysisResult(
        frames=frames,
        annotated_frames=annotated_frames,
        frame_results=frame_results,
        metadata={
            "fps": metadata.fps,
            "resolution": metadata.resolution,
            "frame_count": metadata.frame_count,
            "duration": metadata.duration,
            "frames_analyzed": len(frames),
            "advanced_person_analysis": advanced_person_analysis,
            "events": event_list,
            "event_summary": event_summary,
        },
    )


def _best_weapon_from_people(people):
    best = None
    for person in people:
        weapon = person.get("weapon", {})
        if weapon.get("present") and (
            best is None or weapon.get("confidence", 0) > best.get("confidence", 0)
        ):
            best = weapon
    return best or _empty_weapon_result()
