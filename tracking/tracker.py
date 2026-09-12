from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import face_recognition
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import TrackingConfig
from video.processor import DetectedFace
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


@dataclass
class TrackedFace:
    track_id: int
    bbox: Tuple[int, int, int, int]
    encoding: np.ndarray
    encoding_history: deque = field(default_factory=lambda: deque(maxlen=16))
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_visible: int = 0
    frames_disappeared: int = 0

    @property
    def center(self) -> Tuple[int, int]:
        top, right, bottom, left = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def area(self) -> int:
        top, right, bottom, left = self.bbox
        return (right - left) * (bottom - top)

    @property
    def mean_encoding(self) -> np.ndarray:
        if self.encoding_history:
            return np.mean(list(self.encoding_history), axis=0)
        return self.encoding


class FaceTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracks: Dict[int, TrackedFace] = {}
        self._next_id = 0

    def update(self, detections: List[DetectedFace]) -> Dict[int, TrackedFace]:
        if not self.tracks:
            for det in detections:
                self._register(det)
            return self.tracks

        if not detections:
            to_remove = []
            for tid, track in self.tracks.items():
                track.frames_disappeared += 1
                if track.frames_disappeared > self.config.max_disappeared:
                    to_remove.append(tid)
            for tid in to_remove:
                log.debug("Dropping track %d (disappeared)", tid)
                del self.tracks[tid]
            return self.tracks

        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        for tid, det_idx in matched.items():
            det = detections[det_idx]
            track = self.tracks[tid]
            track.bbox = det.bbox
            track.encoding = det.encoding
            track.encoding_history.append(det.encoding)
            track.last_seen = det.frame_timestamp
            track.frames_visible += 1
            track.frames_disappeared = 0

        # For unmatched detections, try encoding match against disappeared tracks
        still_unmatched = []
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            matched_tid = self._encoding_match(det, unmatched_tracks)
            if matched_tid is not None:
                track = self.tracks[matched_tid]
                track.bbox = det.bbox
                track.encoding = det.encoding
                track.encoding_history.append(det.encoding)
                track.last_seen = det.frame_timestamp
                track.frames_visible += 1
                track.frames_disappeared = 0
                unmatched_tracks.remove(matched_tid)
                log.debug("Re-linked detection to track %d via encoding", matched_tid)
            else:
                still_unmatched.append(det_idx)

        for det_idx in still_unmatched:
            self._register(detections[det_idx])

        to_remove = []
        for tid in unmatched_tracks:
            self.tracks[tid].frames_disappeared += 1
            if self.tracks[tid].frames_disappeared > self.config.max_disappeared:
                to_remove.append(tid)
        for tid in to_remove:
            log.debug("Dropping track %d (disappeared)", tid)
            del self.tracks[tid]

        return self.tracks

    def get_active_tracks(self) -> Dict[int, TrackedFace]:
        return {tid: t for tid, t in self.tracks.items() if t.frames_disappeared == 0}

    def _register(self, det: DetectedFace):
        tid = self._next_id
        self._next_id += 1
        track = TrackedFace(
            track_id=tid,
            bbox=det.bbox,
            encoding=det.encoding,
            first_seen=det.frame_timestamp,
            last_seen=det.frame_timestamp,
            frames_visible=1,
        )
        track.encoding_history.append(det.encoding)
        self.tracks[tid] = track
        log.debug("Registered new track %d", tid)

    def _encoding_match(self, det: DetectedFace, candidate_tids: List[int]) -> Optional[int]:
        """Try to match a detection to a disappeared track by face encoding."""
        if not candidate_tids:
            return None

        best_tid = None
        best_dist = 0.50  # threshold — must be closer than this

        for tid in candidate_tids:
            track = self.tracks[tid]
            dist = face_recognition.face_distance([track.mean_encoding], det.encoding)[0]
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        return best_tid

    def _match(self, detections: List[DetectedFace]):
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].bbox for tid in track_ids]
        det_boxes = [d.bbox for d in detections]

        cost = np.zeros((len(track_boxes), len(det_boxes)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou = self._compute_iou(tb, db)
                cost[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)

        matched = {}
        used_dets = set()
        used_tracks = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < (1.0 - self.config.iou_threshold):
                matched[track_ids[r]] = c
                used_dets.add(c)
                used_tracks.add(track_ids[r])

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]

        return matched, unmatched_dets, unmatched_tracks

    @staticmethod
    def _compute_iou(box_a, box_b) -> float:
        a_top, a_right, a_bottom, a_left = box_a
        b_top, b_right, b_bottom, b_left = box_b

        inter_top = max(a_top, b_top)
        inter_left = max(a_left, b_left)
        inter_bottom = min(a_bottom, b_bottom)
        inter_right = min(a_right, b_right)

        if inter_bottom <= inter_top or inter_right <= inter_left:
            return 0.0

        inter_area = (inter_bottom - inter_top) * (inter_right - inter_left)
        a_area = (a_bottom - a_top) * (a_right - a_left)
        b_area = (b_bottom - b_top) * (b_right - b_left)

        return inter_area / (a_area + b_area - inter_area)
