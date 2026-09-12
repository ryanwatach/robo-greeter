from dataclasses import dataclass
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np

from config import CameraConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left) in original frame coords
    encoding: np.ndarray  # 128-d embedding
    frame_timestamp: float

    @property
    def center(self) -> Tuple[int, int]:
        top, right, bottom, left = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def area(self) -> int:
        top, right, bottom, left = self.bbox
        return (right - left) * (bottom - top)


class FrameProcessor:
    def __init__(self, config: CameraConfig):
        self.scale = config.processing_scale
        self.inv_scale = int(1.0 / config.processing_scale)

    def process(self, frame: np.ndarray, timestamp: float) -> List[DetectedFace]:
        small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        if not locations:
            return []

        encodings = face_recognition.face_encodings(rgb_small, locations)

        faces = []
        for loc, enc in zip(locations, encodings):
            top, right, bottom, left = loc
            original_bbox = (
                top * self.inv_scale,
                right * self.inv_scale,
                bottom * self.inv_scale,
                left * self.inv_scale,
            )
            faces.append(DetectedFace(
                bbox=original_bbox,
                encoding=enc,
                frame_timestamp=timestamp,
            ))

        return faces
