from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

@dataclass
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float

    @property
    def resolution(self) -> str:
        return f"{self.width} × {self.height}"

@dataclass
class FrameData:
    frame_index: int
    timestamp: float
    frame: np.ndarray

class VideoProcessor:
    def __init__(self, source: Union[str, Path]):
        self.source = Path(source)

        if not self.source.exists():
            raise FileNotFoundError(f"Video not found: {self.source}")

    def metadata(self) -> VideoMetadata:
        capture = cv2.VideoCapture(str(self.source))

        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {self.source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        capture.release()

        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=str(self.source),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration=duration,
        )

    def frames(
        self,
        sample_every_n: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Iterator[FrameData]:
        if sample_every_n < 1:
            raise ValueError("sample_every_n must be at least 1.")

        if start_frame < 0:
            raise ValueError("start_frame cannot be negative.")

        if max_frames is not None and max_frames < 1:
            return

        capture = cv2.VideoCapture(str(self.source))

        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {self.source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if end_frame is None:
            end_frame = total_frames

        end_frame = min(end_frame, total_frames)

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        yielded = 0
        frame_index = start_frame

        try:
            while frame_index < end_frame:
                ok, frame = capture.read()

                if not ok:
                    break

                if (frame_index - start_frame) % sample_every_n == 0:
                    timestamp = frame_index / fps if fps > 0 else 0.0

                    yield FrameData(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        frame=frame,
                    )

                    yielded += 1

                    if max_frames is not None and yielded >= max_frames:
                        break

                frame_index += 1
        finally:
            capture.release()

def save_uploaded_video(
    video_bytes: bytes,
    directory: Union[str, Path],
    filename: str = "uploaded_video.mp4",
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / Path(filename).name
    path.write_bytes(video_bytes)

    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("Uploaded video could not be saved.")

    return path

def get_video_metadata_from_bytes(
    video_bytes: bytes,
    temp_dir: Union[str, Path],
    filename: str = "uploaded_video.mp4",
) -> VideoMetadata:
    path = save_uploaded_video(video_bytes, temp_dir, filename)
    return VideoProcessor(path).metadata()
