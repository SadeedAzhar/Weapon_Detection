from dataclasses import dataclass
import math

@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    missed_frames: int = 0
    age: int = 1
    hits: int = 1

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union

def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

class PersonTracker:
    def __init__(self, iou_threshold=0.25, center_distance_ratio=1.5, max_missed_frames=8):
        self.iou_threshold = iou_threshold
        self.center_distance_ratio = center_distance_ratio
        self.max_missed_frames = max_missed_frames
        self._tracks = {}
        self._next_id = 1
        self._expired_ids = []

    @property
    def tracks(self):
        return self._tracks

    @property
    def expired_track_ids(self):
        return self._expired_ids.copy()

    def consume_expired_tracks(self):
        expired = self._expired_ids.copy()
        self._expired_ids.clear()
        return expired

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._expired_ids.clear()

    def _new_track(self, detection):
        track = Track(
            track_id=self._next_id,
            bbox=tuple(detection["bbox_xyxy"]),
            confidence=float(detection.get("confidence", 0.0)),
        )
        self._tracks[track.track_id] = track
        self._next_id += 1
        return track

    def _center_match(self, detection_bbox, track_bbox):
        dx1, dy1, dx2, dy2 = detection_bbox
        tx1, ty1, tx2, ty2 = track_bbox

        dcx, dcy = bbox_center(detection_bbox)
        tcx, tcy = bbox_center(track_bbox)

        distance = math.hypot(dcx - tcx, dcy - tcy)
        track_height = max(1, ty2 - ty1)
        detection_height = max(1, dy2 - dy1)
        reference = max(track_height, detection_height)

        return distance <= reference * self.center_distance_ratio

    def update(self, detections):
        self._expired_ids.clear()

        detections = [dict(item) for item in detections]
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self._tracks.keys())
        matches = []

        # Match the strongest overlaps first.
        candidates = []
        for detection_index, detection in enumerate(detections):
            bbox = tuple(detection["bbox_xyxy"])
            for track_id, track in self._tracks.items():
                score = bbox_iou(bbox, track.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, detection_index, track_id))

        for score, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index not in unmatched_detections:
                continue
            if track_id not in unmatched_tracks:
                continue

            matches.append((detection_index, track_id))
            unmatched_detections.remove(detection_index)
            unmatched_tracks.remove(track_id)

        # Use center distance when IoU is temporarily weak.
        for detection_index in list(unmatched_detections):
            detection_bbox = tuple(detections[detection_index]["bbox_xyxy"])
            best_track_id = None
            best_distance = float("inf")

            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                if self._center_match(detection_bbox, track.bbox):
                    dcx, dcy = bbox_center(detection_bbox)
                    tcx, tcy = bbox_center(track.bbox)
                    distance = math.hypot(dcx - tcx, dcy - tcy)
                    if distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id

            if best_track_id is not None:
                matches.append((detection_index, best_track_id))
                unmatched_detections.remove(detection_index)
                unmatched_tracks.remove(best_track_id)

        # Update matched tracks.
        for detection_index, track_id in matches:
            detection = detections[detection_index]
            track = self._tracks[track_id]
            track.bbox = tuple(detection["bbox_xyxy"])
            track.confidence = float(detection.get("confidence", 0.0))
            track.missed_frames = 0
            track.age += 1
            track.hits += 1
            detection["track_id"] = track_id

        # Start tracks for new people.
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = self._new_track(detection)
            detection["track_id"] = track.track_id

        # Age tracks that were not seen.
        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.missed_frames += 1
            track.age += 1

            if track.missed_frames > self.max_missed_frames:
                self._expired_ids.append(track_id)
                del self._tracks[track_id]

        return detections
