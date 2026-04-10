from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import face_recognition
import numpy as np

from config import IdentityConfig
from identity.database import FaceDatabase
from tracking.tracker import TrackedFace
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


@dataclass
class MatchResult:
    person_id: Optional[int]
    person_name: Optional[str]
    confidence: float
    status: str  # "confirmed", "pending", "unknown"


class IdentityMatcher:
    def __init__(self, config: IdentityConfig, database: FaceDatabase):
        self.config = config
        self.database = database
        self._known: List[Tuple[int, str, np.ndarray]] = []
        self._vote_buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.embedding_buffer_size)
        )
        self.reload_database()

    def reload_database(self):
        self._known = self.database.get_all_persons()
        log.info("Loaded %d known persons from database", len(self._known))

    def process_track(self, track: TrackedFace) -> MatchResult:
        encoding = track.encoding
        tid = track.track_id

        if not self._known:
            self._vote_buffers[tid].append(None)
            return self._tally_votes(tid)

        known_encodings = [k[2] for k in self._known]
        distances = face_recognition.face_distance(known_encodings, encoding)

        best_idx = int(np.argmin(distances))
        best_dist = distances[best_idx]
        best_name = self._known[best_idx][1]

        if best_dist < self.config.distance_threshold:
            vote = self._known[best_idx][0]  # person_id
            log.debug(f"Track {tid}: matched {best_name} (dist={best_dist:.3f}, threshold={self.config.distance_threshold})")
        else:
            vote = None
            log.debug(f"Track {tid}: no match, closest is {best_name} (dist={best_dist:.3f}, threshold={self.config.distance_threshold})")

        self._vote_buffers[tid].append(vote)
        result = self._tally_votes(tid)
        if result.status == "confirmed":
            log.info(f"Track {tid} CONFIRMED as {result.person_name} (confidence={result.confidence:.0%})")
        elif result.status == "unknown":
            log.info(f"Track {tid} marked UNKNOWN (confidence={result.confidence:.0%})")
        return result

    def _tally_votes(self, track_id: int) -> MatchResult:
        buf = self._vote_buffers[track_id]

        if len(buf) < self.config.confirmation_frames:
            return MatchResult(None, None, 0.0, "pending")

        recent = list(buf)[-self.config.confirmation_frames:]
        total = len(recent)

        none_count = recent.count(None)
        if none_count / total >= self.config.confirmation_ratio:
            return MatchResult(None, None, none_count / total, "unknown")

        vote_counts: Dict[int, int] = {}
        for v in recent:
            if v is not None:
                vote_counts[v] = vote_counts.get(v, 0) + 1

        if not vote_counts:
            return MatchResult(None, None, 0.0, "pending")

        best_pid = max(vote_counts, key=vote_counts.get)
        best_count = vote_counts[best_pid]
        confidence = best_count / total

        if confidence >= self.config.confirmation_ratio:
            name = None
            for pid, n, _ in self._known:
                if pid == best_pid:
                    name = n
                    break
            return MatchResult(best_pid, name, confidence, "confirmed")

        return MatchResult(None, None, confidence, "pending")

    def reset_track(self, track_id: int):
        if track_id in self._vote_buffers:
            del self._vote_buffers[track_id]

    def reset_all(self):
        self._vote_buffers.clear()
