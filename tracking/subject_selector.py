from typing import Dict, Optional, Tuple

from config import TrackingConfig
from tracking.tracker import TrackedFace


class PrimarySubjectSelector:
    def __init__(self, config: TrackingConfig, frame_shape: Tuple[int, int]):
        self.config = config
        self.frame_h, self.frame_w = frame_shape

    def update_frame_shape(self, frame_shape: Tuple[int, int]):
        self.frame_h, self.frame_w = frame_shape

    def select(
        self,
        tracks: Dict[int, TrackedFace],
        current_subject_id: Optional[int] = None,
    ) -> Optional[int]:
        active = {tid: t for tid, t in tracks.items() if t.frames_disappeared == 0}
        if not active:
            return None

        best_id = None
        best_score = -1.0

        max_area = max(t.area for t in active.values()) or 1
        max_frames = max(t.frames_visible for t in active.values()) or 1

        cx, cy = self.frame_w / 2, self.frame_h / 2

        for tid, track in active.items():
            fx, fy = track.center
            dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
            max_dist = (cx ** 2 + cy ** 2) ** 0.5
            center_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0.5

            size_score = track.area / max_area
            persist_score = track.frames_visible / max_frames

            score = (
                self.config.center_weight * center_score
                + self.config.size_weight * size_score
                + self.config.persistence_weight * persist_score
            )

            if tid == current_subject_id:
                score += self.config.hysteresis_bonus

            if score > best_score:
                best_score = score
                best_id = tid

        return best_id
