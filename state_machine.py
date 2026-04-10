import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from config import StateConfig
from identity.matcher import IdentityMatcher, MatchResult
from tracking.tracker import TrackedFace
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


class State(Enum):
    IDLE = auto()
    SCANNING = auto()       # faces detected, collecting identity info
    READY_TO_GREET = auto() # enough info, ready to speak
    GREETING = auto()       # speech in progress
    COOLDOWN = auto()       # post-speech cooldown
    TRACKING = auto()       # greeted, just tracking


@dataclass
class GreetingRequest:
    """One combined greeting for all detected faces."""
    known_names: List[Tuple[str, int]]   # (name, person_id) pairs
    unknown_count: int
    tracks: Dict[int, TrackedFace]


@dataclass
class StateMachineOutput:
    ptz_target_bbox: Optional[Tuple[int, int, int, int]] = None
    should_disengage_ptz: bool = False
    greeting_request: Optional[GreetingRequest] = None


class GreeterStateMachine:
    def __init__(self, config: StateConfig, matcher: IdentityMatcher):
        self.config = config
        self.matcher = matcher

        self._state: State = State.IDLE
        self._state_enter_time: float = time.monotonic()
        self._greeted_person_ids: set = set()
        self._greeted_track_ids: set = set()  # tracks we've already greeted (known or unknown)
        self._primary_track_id: Optional[int] = None
        self._scan_results: Dict[int, MatchResult] = {}  # track_id -> latest match

    @property
    def state(self) -> State:
        return self._state

    @property
    def current_subject_id(self) -> Optional[int]:
        return self._primary_track_id

    def mark_greeting_done(self):
        self._transition(State.COOLDOWN)

    def freeze_for_speech(self):
        """Called externally to indicate speech is active."""
        if self._state == State.READY_TO_GREET:
            self._transition(State.GREETING)

    def tick(
        self,
        active_tracks: Dict[int, TrackedFace],
        primary_id: Optional[int],
    ) -> StateMachineOutput:
        self._primary_track_id = primary_id

        handler = {
            State.IDLE: self._handle_idle,
            State.SCANNING: self._handle_scanning,
            State.READY_TO_GREET: self._handle_ready,
            State.GREETING: self._handle_greeting,
            State.COOLDOWN: self._handle_cooldown,
            State.TRACKING: self._handle_tracking,
        }[self._state]

        return handler(active_tracks, primary_id)

    def _transition(self, new_state: State):
        if new_state != self._state:
            log.info("State: %s -> %s", self._state.name, new_state.name)
        self._state = new_state
        self._state_enter_time = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self._state_enter_time

    # --- Handlers ---

    def _handle_idle(self, tracks, primary_id) -> StateMachineOutput:
        if tracks:
            self._scan_results.clear()
            self._transition(State.SCANNING)
        return StateMachineOutput(should_disengage_ptz=True)

    def _handle_scanning(self, tracks, primary_id) -> StateMachineOutput:
        if not tracks:
            # Don't immediately drop — give 3 seconds for face to reappear
            if self._elapsed() > 3.0:
                self._transition(State.IDLE)
            return StateMachineOutput(should_disengage_ptz=True)

        # Run matcher on all active (non-greeted) tracks
        ungreeted = {tid: t for tid, t in tracks.items()
                     if tid not in self._greeted_track_ids and t.frames_disappeared == 0}

        if not ungreeted:
            # Everyone visible has been greeted already
            self._transition(State.TRACKING)
            return self._ptz_output(tracks, primary_id)

        for tid, track in ungreeted.items():
            result = self.matcher.process_track(track)
            self._scan_results[tid] = result

        # Check if we have enough info to greet (all ungreeted are confirmed or unknown)
        all_resolved = all(
            r.status in ("confirmed", "unknown")
            for tid, r in self._scan_results.items()
            if tid in ungreeted
        )

        if all_resolved and self._elapsed() > self.config.acquire_timeout:
            self._transition(State.READY_TO_GREET)

        if self._elapsed() > self.config.identify_timeout:
            # Timed out, greet with what we have
            self._transition(State.READY_TO_GREET)

        return self._ptz_output(tracks, primary_id)

    def _handle_ready(self, tracks, primary_id) -> StateMachineOutput:
        # Build a combined greeting request
        known_names = []
        unknown_count = 0

        for tid, result in self._scan_results.items():
            if tid in self._greeted_track_ids:
                continue
            if result.status == "confirmed" and result.person_id is not None:
                if result.person_id not in self._greeted_person_ids:
                    known_names.append((result.person_name, result.person_id))
                    self._greeted_person_ids.add(result.person_id)
                self._greeted_track_ids.add(tid)
            elif result.status == "unknown":
                unknown_count += 1
                self._greeted_track_ids.add(tid)

        if not known_names and unknown_count == 0:
            # Nothing new to greet
            self._transition(State.TRACKING)
            return self._ptz_output(tracks, primary_id)

        request = GreetingRequest(
            known_names=known_names,
            unknown_count=unknown_count,
            tracks=tracks,
        )

        return StateMachineOutput(
            ptz_target_bbox=self._get_primary_bbox(tracks, primary_id),
            greeting_request=request,
        )

    def _handle_greeting(self, tracks, primary_id) -> StateMachineOutput:
        # Just track, don't do anything else — speech is in progress
        return self._ptz_output(tracks, primary_id)

    def _handle_cooldown(self, tracks, primary_id) -> StateMachineOutput:
        if self._elapsed() > self.config.disengage_cooldown:
            if tracks:
                self._transition(State.TRACKING)
            else:
                self._transition(State.IDLE)
        return self._ptz_output(tracks, primary_id)

    def _handle_tracking(self, tracks, primary_id) -> StateMachineOutput:
        if not tracks:
            if self._elapsed() > self.config.interaction_timeout:
                self._greeted_track_ids.clear()
                self._scan_results.clear()
                self._transition(State.IDLE)
            return StateMachineOutput(should_disengage_ptz=True)

        # Check if any NEW (ungreeted) faces appeared
        ungreeted = {tid: t for tid, t in tracks.items()
                     if tid not in self._greeted_track_ids and t.frames_disappeared == 0}
        if ungreeted:
            self._scan_results.clear()
            self._transition(State.SCANNING)

        return self._ptz_output(tracks, primary_id)

    # --- Helpers ---

    def _ptz_output(self, tracks, primary_id) -> StateMachineOutput:
        bbox = self._get_primary_bbox(tracks, primary_id)
        return StateMachineOutput(ptz_target_bbox=bbox)

    def _get_primary_bbox(self, tracks, primary_id):
        if primary_id is not None and primary_id in tracks:
            t = tracks[primary_id]
            if t.frames_disappeared == 0:
                return t.bbox
        return None
