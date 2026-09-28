import time
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from config import CCTV_BUFFER_SIZE, CCTV_FPS_LIMIT


@dataclass
class CCTVMetadata:
    source: str
    fps: float = 0.0
    width: int = 0
    height: int = 0
    opened: bool = False

    @property
    def resolution(self) -> str:
        if not self.width or not self.height:
            return "Unknown"
        return f"{self.width} × {self.height}"


@dataclass
class CCTVIncident:
    start_time: float
    end_time: float
    peak_time: float
    frame_index: int
    weapon_class: str
    peak_confidence: float
    detection_count: int
    snapshot: bytes

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "peak_time": self.peak_time,
            "frame_index": self.frame_index,
            "weapon_class": self.weapon_class,
            "peak_confidence": self.peak_confidence,
            "detection_count": self.detection_count,
        }

    @property
    def label(self) -> str:
        return self.weapon_class.replace("_", " ").title()

    @property
    def time_range(self) -> str:
        if self.duration < 0.15:
            return f"{self.start_time:.2f}s"
        return f"{self.start_time:.2f}s – {self.end_time:.2f}s"


class CCTVIncidentManager:
    def __init__(self, group_gap: float = 1.5, max_incidents: int = 50):
        self.group_gap = max(0.1, float(group_gap))
        self.max_incidents = max(1, int(max_incidents))
        self.incidents: list[CCTVIncident] = []
        self._active: Optional[CCTVIncident] = None

    def _label(self, weapon: dict) -> str:
        top_classes = weapon.get("top_classes", []) or []
        if top_classes:
            return str(top_classes[0].get("class", "weapon"))
        return "weapon"

    def _snapshot(self, frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 78],
        )
        if not ok:
            return b""
        return encoded.tobytes()

    def update(
        self,
        timestamp: float,
        frame_index: int,
        weapon: dict,
        frame: np.ndarray,
    ) -> None:
        if not weapon or not weapon.get("present"):
            return

        timestamp = float(timestamp)
        confidence = float(weapon.get("confidence", 0.0))
        label = self._label(weapon)

        if (
            self._active is None
            or timestamp - self._active.end_time > self.group_gap
        ):
            self._close_active()
            self._active = CCTVIncident(
                start_time=timestamp,
                end_time=timestamp,
                peak_time=timestamp,
                frame_index=int(frame_index),
                weapon_class=label,
                peak_confidence=confidence,
                detection_count=1,
                snapshot=self._snapshot(frame),
            )
            return

        self._active.end_time = timestamp
        self._active.detection_count += 1

        if confidence > self._active.peak_confidence:
            self._active.peak_confidence = confidence
            self._active.peak_time = timestamp
            self._active.frame_index = int(frame_index)
            self._active.weapon_class = label
            self._active.snapshot = self._snapshot(frame)

    def _close_active(self) -> None:
        if self._active is None:
            return
        self.incidents.append(self._active)
        self.incidents = self.incidents[-self.max_incidents:]
        self._active = None

    def finalize(self) -> None:
        self._close_active()

    def all_incidents(self, include_active: bool = True) -> list[CCTVIncident]:
        if include_active and self._active is not None:
            return [*self.incidents, self._active]
        return list(self.incidents)

    def reset(self) -> None:
        self.incidents.clear()
        self._active = None

    def flush_active(self) -> None:
        """Finalize the current live incident without clearing history."""
        self._close_active()


class CCTVProcessor:
    def __init__(
        self,
        source: str | int,
        buffer_size: int = CCTV_BUFFER_SIZE,
        fps_limit: float = CCTV_FPS_LIMIT,
    ):
        self.source = source
        self.buffer_size = max(1, int(buffer_size))
        self.fps_limit = max(0.0, float(fps_limit))
        self.capture: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0

    def connect(self) -> bool:
        self.release()

        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            self.release()
            return False

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return True

    def is_connected(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def metadata(self) -> CCTVMetadata:
        if not self.is_connected():
            return CCTVMetadata(
                source=str(self.source),
                opened=False,
            )

        return CCTVMetadata(
            source=str(self.source),
            fps=float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0),
            width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            opened=True,
        )

    def _wait_for_frame(self) -> None:
        if self.fps_limit <= 0:
            return

        interval = 1.0 / self.fps_limit
        elapsed = time.monotonic() - self._last_frame_time

        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected():
            return False, None

        self._wait_for_frame()

        ok, frame = self.capture.read()

        if not ok:
            return False, None

        self._last_frame_time = time.monotonic()
        return True, frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        return self._read_frame()

    def read_latest(
        self,
        drain_reads: int = 6,
    ) -> tuple[bool, Optional[np.ndarray]]:
        """Read a fresh frame and drop stale buffered frames."""
        if not self.is_connected():
            return False, None

        self._wait_for_frame()

        ok, frame = self.capture.read()
        if not ok:
            return False, None

        for _ in range(max(0, int(drain_reads))):
            if not self.capture.grab():
                break

            ok, latest = self.capture.retrieve()
            if ok:
                frame = latest

        self._last_frame_time = time.monotonic()
        return True, frame

    def frames(
        self,
        max_frames: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        count = 0

        while self.is_connected():
            if max_frames is not None and count >= max_frames:
                break

            ok, frame = self.read()

            if not ok:
                break

            yield frame
            count += 1

    def reconnect(
        self,
        attempts: int = 3,
        delay: float = 1.0,
    ) -> bool:
        attempts = max(1, int(attempts))

        for attempt in range(attempts):
            if self.connect():
                return True

            if attempt < attempts - 1:
                time.sleep(max(0.0, delay))

        return False

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self._last_frame_time = 0.0


def normalize_cctv_source(source: str) -> str | int:
    value = str(source).strip()

    if value.isdigit():
        return int(value)

    return value


def validate_cctv_source(source: str | int) -> tuple[bool, str]:
    if isinstance(source, int):
        if source >= 0:
            return True, ""
        return False, "Camera index must be 0 or greater."

    value = str(source).strip()

    if not value:
        return False, "CCTV source cannot be empty."

    if value.isdigit():
        return True, ""

    if value.startswith((
        "rtsp://",
        "rtmp://",
        "http://",
        "https://",
    )):
        return True, ""

    return False, (
        "Use a camera index or a supported RTSP/HTTP/RTMP stream URL."
    )
