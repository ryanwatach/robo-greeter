import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from audio.audio_manager import AudioManager
from identity.database import FaceDatabase
from identity.matcher import IdentityMatcher
from interaction.conversation_manager import ConversationManager
from tracking.tracker import TrackedFace
from utils.logger import setup_logger

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "faces")  # legacy snapshot dir (kept)
PEOPLE_DIR = os.path.join(DATA_DIR, "people")    # per-person folders


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def _person_dir(person_id: int, name: str) -> str:
    return os.path.join(PEOPLE_DIR, f"{person_id:03d}_{_safe_name(name)}")

log = setup_logger("robo-greeter")

GREETER_NAME = "Jarvis"


class GreeterLogic:
    def __init__(
        self,
        audio: AudioManager,
        database: FaceDatabase,
        matcher: IdentityMatcher,
        gemini_api_key: str,
    ):
        self.audio = audio
        self.database = database
        self.matcher = matcher
        self._current_frame: Optional[np.ndarray] = None
        # Face presence: when a face was last seen (monotonic time)
        self._face_last_seen: float = time.monotonic()
        self.conversation_manager = ConversationManager(
            audio, gemini_api_key,
            face_absent_seconds=self.face_absent_seconds,
            tools=self._build_db_tools(),
        )

    def _build_db_tools(self):
        """Return plain-Python callables exposed to Gemini for function
        calling. The genai SDK introspects signatures + docstrings to build
        the tool schema and invokes them automatically when the model asks.
        These run synchronously in the conversation thread."""
        db = self.database

        def get_visit_count(name: str) -> int:
            """Return how many times the named person has checked in. Returns 0 if the person is not known."""
            row = db.conn.execute(
                "SELECT visit_count FROM persons WHERE LOWER(name) = LOWER(?)",
                (name,),
            ).fetchone()
            return int(row[0]) if row else 0

        def get_last_visit(name: str) -> str:
            """Return the most recent check-in timestamp for a person as an ISO-like string, or 'never' if the person has never checked in."""
            row = db.conn.execute(
                "SELECT last_seen FROM persons WHERE LOWER(name) = LOWER(?)",
                (name,),
            ).fetchone()
            return row[0] if row and row[0] else "never"

        def count_checkins_today() -> int:
            """Return the number of check-ins recorded today (local time)."""
            row = db.conn.execute(
                "SELECT COUNT(*) FROM check_ins WHERE date(checked_in_at) = date('now', 'localtime')"
            ).fetchone()
            return int(row[0]) if row else 0

        def list_known_people() -> list:
            """Return the names of all people enrolled in the system."""
            rows = db.conn.execute("SELECT name FROM persons ORDER BY name").fetchall()
            return [r[0] for r in rows]

        def get_recent_checkins(limit: int = 10) -> list:
            """Return the most recent check-ins as a list of {name, time} entries (newest first). limit is capped at 50."""
            limit = max(1, min(50, int(limit)))
            rows = db.conn.execute(
                "SELECT name, checked_in_at FROM check_ins ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [{"name": n, "time": t} for n, t in rows]

        return [
            get_visit_count,
            get_last_visit,
            count_checkins_today,
            list_known_people,
            get_recent_checkins,
        ]

    def update_face_presence(self, face_present: bool):
        """Called by main loop each frame so the conversation flow knows
        whether the user is currently in front of the camera."""
        if face_present:
            self._face_last_seen = time.monotonic()

    def face_absent_seconds(self) -> float:
        return time.monotonic() - self._face_last_seen

    def save_person_record(
        self,
        person_id: int,
        name: str,
        track: Optional[TrackedFace] = None,
        frame: Optional[np.ndarray] = None,
        write_snapshot: bool = False,
    ):
        """Maintain a per-person folder at data/people/{id}_{name}/ with a
        face snapshot (likeness) and a details.json (name, id, dates,
        visit_count, recent check-ins). Called on enrollment and on every
        check-in so details.json stays current."""
        try:
            pdir = _person_dir(person_id, name)
            os.makedirs(pdir, exist_ok=True)

            snap_path = os.path.join(pdir, "snapshot.jpg")
            if write_snapshot and frame is not None and track is not None:
                top, right, bottom, left = track.bbox
                h, w = frame.shape[:2]
                pad = 40
                top = max(0, top - pad)
                left = max(0, left - pad)
                bottom = min(h, bottom + pad)
                right = min(w, right + pad)
                face_img = frame[top:bottom, left:right]
                if face_img.size > 0:
                    cv2.imwrite(snap_path, face_img)

            info = self.database.get_person_by_id(person_id) or {}
            recent = self.database.get_check_ins_for_person(person_id, limit=20)
            details = {
                "id": person_id,
                "name": name,
                "created_at": info.get("created_at"),
                "last_seen": info.get("last_seen"),
                "visit_count": info.get("visit_count"),
                "snapshot": os.path.basename(snap_path) if os.path.exists(snap_path) else None,
                "recent_check_ins": [{"id": cid, "at": at} for cid, at in recent],
            }
            with open(os.path.join(pdir, "details.json"), "w") as f:
                json.dump(details, f, indent=2)
        except Exception as e:
            log.warning("save_person_record failed for %s: %s", name, e)

    def save_face_snapshot(self, name: str, track: TrackedFace, frame: Optional[np.ndarray] = None):
        """Save a cropped face image for reference."""
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        if frame is not None:
            top, right, bottom, left = track.bbox
            # Add padding
            h, w = frame.shape[:2]
            pad = 40
            top = max(0, top - pad)
            left = max(0, left - pad)
            bottom = min(h, bottom + pad)
            right = min(w, right + pad)
            face_img = frame[top:bottom, left:right]
        else:
            return

        safe_name = re.sub(r'[^\w\-]', '_', name)
        ts = int(time.time())
        path = os.path.join(SNAPSHOT_DIR, f"{safe_name}_{ts}.jpg")
        cv2.imwrite(path, face_img)
        log.info("Saved face snapshot: %s", path)

    def set_current_frame(self, frame: np.ndarray):
        """Called by main loop to give greeter access to the latest frame."""
        self._current_frame = frame

    def greet_and_enroll(
        self,
        known_names: List[Tuple[str, int]],
        unknown_count: int,
        unknown_tracks: List[TrackedFace],
    ):
        """
        Full conversation flow. This runs in a thread and handles everything:
        - Greet known people
        - For unknowns: ask name, get response, confirm, enroll
        All in one continuous conversation.
        """
        names = [n for n, _ in known_names]
        log.info(f"greet_and_enroll: known_names={names}, unknown_count={unknown_count}")

        # Step 1: Greet known people
        if names and unknown_count == 0:
            self._greet_known_only(names)
        elif not names and unknown_count == 1:
            self._enroll_single_unknown(unknown_tracks[0] if unknown_tracks else None)
        elif not names and unknown_count > 1:
            self._enroll_multiple_unknowns(unknown_tracks)
        elif names and unknown_count > 0:
            self._greet_mixed(names, unknown_tracks)

        # Update visit counts, record a check-in, refresh per-person folder
        for name, pid in known_names:
            self.database.update_last_seen(pid)
            self.database.record_check_in(pid, name)
            self.save_person_record(pid, name)

    def _greet_known_only(self, names: List[str]):
        if len(names) == 1:
            self.audio.say(random.choice([
                f"Ah, {names[0]}. Good to see you. Let me sign you in.",
                f"{names[0]}, welcome. Signing you in now.",
                f"Good to see you, {names[0]}. I'll get you checked in.",
            ]))
        else:
            name_list = self._join_names(names)
            self.audio.say(f"Hello {name_list}. Good to see you. Let me get you signed in.")

        # Start a conversation
        self.conversation_manager.start_conversation(names)

    def _enroll_single_unknown(self, track: Optional[TrackedFace]):
        # Ask their name
        name = self.audio.ask(
            f"Good day. I'm {GREETER_NAME}. I don't believe we've met. Please say 'My name is' followed by your name."
        )

        if not name or not name.strip():
            self.audio.say("No worries. I'll be here when you're ready.")
            return

        # Extract name only if prefixed with "my name is"
        name = self._extract_name_from_introduction(name)
        if not name:
            self.audio.say("I didn't catch that. Please say 'My name is' followed by your name.")
            return

        # Confirm
        confirmed = self.audio.ask_yes_no(f"{name}, is that right?")
        if confirmed is False:
            name2 = self.audio.ask("My apologies. Please say 'My name is' followed by your name.")
            if name2 and name2.strip():
                name2 = self._extract_name_from_introduction(name2)
                if name2:
                    name = name2
                else:
                    self.audio.say("That's alright. We'll sort it out next time.")
                    return
            else:
                self.audio.say("That's alright. We'll sort it out next time.")
                return

        # Enroll
        if track:
            embeddings = list(track.encoding_history)
            if not embeddings:
                embeddings = [track.encoding]
            person_id = self.database.add_person(name, embeddings)
            self.matcher.reload_database()
            self.save_face_snapshot(name, track, self._current_frame)
            self.database.record_check_in(person_id, name)
            self.save_person_record(person_id, name, track, self._current_frame, write_snapshot=True)
            self.audio.say(
                f"A pleasure to meet you, {name}. "
                "I'll remember you from now on. You're all checked in."
            )
            log.info("Enrolled: %s (id=%d, %d embeddings)", name, person_id, len(embeddings))

            # Start a conversation with the new person
            self.conversation_manager.start_conversation([name])
        else:
            self.audio.say(f"Nice to meet you, {name}. Welcome.")

    def _enroll_multiple_unknowns(self, tracks: List[TrackedFace]):
        self.audio.say(
            f"Good day. I'm {GREETER_NAME}. "
            "I don't think I've met any of you yet. "
            "Let's get you all checked in."
        )
        for i, track in enumerate(tracks):
            name = self.audio.ask(f"Please say 'My name is' followed by your name.")
            if not name or not name.strip():
                continue
            name = self._extract_name_from_introduction(name)
            if not name:
                self.audio.say("I didn't catch that. Please say 'My name is' followed by your name.")
                continue
            embeddings = list(track.encoding_history) or [track.encoding]
            pid = self.database.add_person(name, embeddings)
            self.matcher.reload_database()
            self.save_face_snapshot(name, track, self._current_frame)
            self.database.record_check_in(pid, name)
            self.save_person_record(pid, name, track, self._current_frame, write_snapshot=True)
            self.audio.say(f"Got it, {name}. You're checked in.")
            log.info("Enrolled: %s (id=%d)", name, pid)

            # Conversation with each new person
            self.conversation_manager.start_conversation([name])

    def _greet_mixed(self, known_names: List[str], unknown_tracks: List[TrackedFace]):
        name_list = self._join_names(known_names)
        self.audio.say(f"Hello {name_list}, good to see you. Let me sign you in.")

        # Have a conversation with the known people
        self.conversation_manager.start_conversation(known_names)

        # Now enroll the unknowns
        if len(unknown_tracks) == 1:
            self.audio.say("And I see someone new.")
            self._enroll_single_unknown(unknown_tracks[0])
        elif len(unknown_tracks) > 1:
            self.audio.say("And I see some new faces.")
            self._enroll_multiple_unknowns(unknown_tracks)

    @staticmethod
    def _join_names(names: List[str]) -> str:
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return ", ".join(names[:-1]) + f", and {names[-1]}"

    @staticmethod
    def _extract_name_from_introduction(raw: str) -> Optional[str]:
        """
        Extract name ONLY if the input starts with 'my name is'.
        Returns None if the required prefix is not found.
        """
        text = raw.strip()
        lower = text.lower()

        # MUST start with "my name is"
        if not lower.startswith("my name is"):
            return None

        # Extract everything after "my name is"
        name = text[10:].strip()  # len("my name is") == 10

        # Clean up the name
        name = re.sub(r"[.,!?]+$", "", name).strip()
        name = re.sub(r"[^\w\s\-']", "", name).strip()

        if name:
            name = " ".join(w.capitalize() for w in name.split())
            return name

        return None

    @staticmethod
    def _normalize_name(raw: str) -> str:
        filler = [
            "my name is", "i'm", "im", "i am", "it's", "its",
            "they call me", "call me", "you can call me",
            "hi i'm", "hey i'm", "hello i'm",
            "hi", "hello", "hey", "um", "uh", "so",
        ]
        text = raw.strip()
        lower = text.lower()
        for f in sorted(filler, key=len, reverse=True):
            if lower.startswith(f):
                text = text[len(f):].strip()
                lower = text.lower()
        text = re.sub(r"[.,!?]+$", "", text).strip()
        text = re.sub(r"[^\w\s\-']", "", text).strip()
        if text:
            text = " ".join(w.capitalize() for w in text.split())
        return text if text else "Friend"
