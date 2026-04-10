#!/usr/bin/env python3
"""
Jarvis Robo-Greeter
PTZ face tracking, recognition, voice interaction, persistent identity DB.

Press 'q' to quit. Press 'b' to build DB from ryan.jpg.
"""
import os
import signal
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from config import AppConfig
from video.capture import ThreadedCamera
from video.processor import FrameProcessor
from tracking.tracker import FaceTracker
from tracking.subject_selector import PrimarySubjectSelector
from identity.database import FaceDatabase
from identity.matcher import IdentityMatcher
from camera_control.ptz import PTZController
from audio.tts import TTSEngine
from audio.stt import STTEngine
from audio.audio_manager import AudioManager
from interaction.greeter import GreeterLogic, GREETER_NAME
from state_machine import GreeterStateMachine, State, StateMachineOutput
from utils.logger import setup_logger


# ── UI Drawing ──────────────────────────────────────────────────

# Colors (BGR)
C_GREEN = (0, 220, 100)
C_CYAN = (220, 200, 0)
C_ORANGE = (0, 150, 255)
C_RED = (60, 60, 220)
C_PURPLE = (180, 60, 160)
C_GRAY = (160, 160, 160)
C_WHITE = (240, 240, 240)
C_BLACK = (0, 0, 0)
C_DARK = (30, 30, 30)
C_PANEL = (40, 40, 40)

STATE_COLORS = {
    State.IDLE: C_GRAY,
    State.SCANNING: C_CYAN,
    State.READY_TO_GREET: C_ORANGE,
    State.GREETING: C_GREEN,
    State.COOLDOWN: C_PURPLE,
    State.TRACKING: C_GREEN,
}

STATE_LABELS = {
    State.IDLE: "WAITING",
    State.SCANNING: "IDENTIFYING",
    State.READY_TO_GREET: "READY",
    State.GREETING: "SPEAKING",
    State.COOLDOWN: "COOLDOWN",
    State.TRACKING: "TRACKING",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN


# ── Chat Log ─────────────────────────────────────────────────────

class ChatLog:
    def __init__(self, maxlen=100):
        self._messages = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, role: str, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._messages.append((role, text, ts))

    def get_messages(self):
        with self._lock:
            return list(self._messages)


def _wrap_text(text: str, max_chars: int) -> list:
    """Wrap text at word boundaries."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > max_chars:
            lines.append(current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)
    return lines or [""]


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_face_box(frame, bbox, color, label, is_primary):
    top, right, bottom, left = bbox
    thickness = 2 if is_primary else 1

    # Corner brackets instead of full rectangle for a cleaner look
    corner_len = max(15, (right - left) // 5)

    # Top-left
    cv2.line(frame, (left, top), (left + corner_len, top), color, thickness + 1)
    cv2.line(frame, (left, top), (left, top + corner_len), color, thickness + 1)
    # Top-right
    cv2.line(frame, (right, top), (right - corner_len, top), color, thickness + 1)
    cv2.line(frame, (right, top), (right, top + corner_len), color, thickness + 1)
    # Bottom-left
    cv2.line(frame, (left, bottom), (left + corner_len, bottom), color, thickness + 1)
    cv2.line(frame, (left, bottom), (left, bottom - corner_len), color, thickness + 1)
    # Bottom-right
    cv2.line(frame, (right, bottom), (right - corner_len, bottom), color, thickness + 1)
    cv2.line(frame, (right, bottom), (right, bottom - corner_len), color, thickness + 1)

    if is_primary:
        # Thin connecting lines
        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)

    # Label background
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
        lx = left
        ly = top - 8
        cv2.rectangle(frame, (lx - 2, ly - th - 6), (lx + tw + 6, ly + 4), C_DARK, -1)
        cv2.putText(frame, label, (lx + 2, ly), FONT, 0.6, color, 1, cv2.LINE_AA)


def draw_crosshair(frame, bbox, color):
    """Draw a small crosshair at face center."""
    top, right, bottom, left = bbox
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    size = 8
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)


def draw_dpad(canvas, h, w):
    """Draw D-pad arrow buttons for manual PTZ control (bottom-left of video)."""
    btn = 36
    gap = 8
    cx = 90
    cy = h - 100

    dirs = {
        "up": (cx, cy - btn - gap),
        "down": (cx, cy + btn + gap),
        "left": (cx - btn - gap, cy),
        "right": (cx + btn + gap, cy),
    }
    arrow_pts = {
        "up": [(0, -btn + 6), (-btn + 8, btn - 8), (btn - 8, btn - 8)],
        "down": [(0, btn - 6), (-btn + 8, -btn + 8), (btn - 8, -btn + 8)],
        "left": [(-btn + 6, 0), (btn - 8, -btn + 8), (btn - 8, btn - 8)],
        "right": [(btn - 6, 0), (-btn + 8, -btn + 8), (-btn + 8, btn - 8)],
    }
    labels = {"up": "W", "down": "S", "left": "A", "right": "D"}

    for direction, (bx, by) in dirs.items():
        draw_rounded_rect(canvas, (bx - btn, by - btn), (bx + btn, by + btn), (60, 60, 60), radius=6)
        pts_rel = arrow_pts[direction]
        pts = np.array([[bx + dx, by + dy] for dx, dy in pts_rel], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], C_CYAN)
        cv2.putText(canvas, labels[direction], (bx - 5, by + 5), FONT_SMALL, 0.9, C_WHITE, 1, cv2.LINE_AA)


def draw_chat_panel(canvas, px, panel_w, y_start, y_end, chat_log, chat_input_active, chat_input_buffer, conversation_active):
    """Draw live captions and chat input box."""
    # Divider + header
    py = y_start
    cv2.line(canvas, (px, py), (px + panel_w - 30, py), C_GRAY, 1)
    py += 15
    cv2.putText(canvas, "CAPTIONS", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)

    msg_area_end = y_end - 42
    messages = chat_log.get_messages()

    # Wrap and render messages from bottom up
    rendered_rows = []
    for role, text, ts in messages:
        wrapped = _wrap_text(text, 36)
        for i, line in enumerate(wrapped):
            prefix = "J: " if role == "Jarvis" else "You: "
            role_color = C_CYAN if role == "Jarvis" else C_GREEN
            rendered_rows.append((prefix, line, role_color, ts if i == 0 else ""))

    # Take the last N rows that fit (20px per row)
    max_rows = (msg_area_end - py - 20) // 20
    visible_rows = rendered_rows[-max_rows:] if len(rendered_rows) > max_rows else rendered_rows

    # Render from bottom up
    msg_y = msg_area_end - 5
    for prefix, line, color, ts in reversed(visible_rows):
        # Prefix + text
        prefix_text = prefix
        cv2.putText(canvas, prefix_text, (px + 5, msg_y), FONT_SMALL, 0.8, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, line, (px + 40, msg_y), FONT_SMALL, 0.8, C_WHITE, 1, cv2.LINE_AA)
        # Timestamp on first line of message
        if ts:
            cv2.putText(canvas, ts, (px + panel_w - 85, msg_y), FONT_SMALL, 0.7, C_GRAY, 1, cv2.LINE_AA)
        msg_y -= 20

    # Input box at bottom
    input_y_top = y_end - 40
    input_y_bot = y_end - 10
    input_x_left = px
    input_x_right = px + panel_w - 30

    if chat_input_active:
        # Active input box: cyan border, show buffer + cursor
        draw_rounded_rect(canvas, (input_x_left, input_y_top), (input_x_right, input_y_bot), C_DARK, radius=4)
        cv2.rectangle(canvas, (input_x_left, input_y_top), (input_x_right, input_y_bot), C_CYAN, 2)
        cursor = "_" if int(time.monotonic() * 2) % 2 == 0 else " "
        cv2.putText(canvas, chat_input_buffer + cursor, (input_x_left + 8, input_y_bot - 8),
                    FONT_SMALL, 0.8, C_WHITE, 1, cv2.LINE_AA)
    elif conversation_active:
        # Inactive but conversation active: hint text
        draw_rounded_rect(canvas, (input_x_left, input_y_top), (input_x_right, input_y_bot), (50, 50, 50), radius=4)
        cv2.putText(canvas, "type to reply...", (input_x_left + 8, input_y_bot - 8),
                    FONT_SMALL, 0.7, C_GRAY, 1, cv2.LINE_AA)
    else:
        # No conversation: dimmed
        draw_rounded_rect(canvas, (input_x_left, input_y_top), (input_x_right, input_y_bot), (35, 35, 35), radius=4)
        cv2.putText(canvas, "no active conversation", (input_x_left + 8, input_y_bot - 8),
                    FONT_SMALL, 0.6, (80, 80, 80), 1, cv2.LINE_AA)


def build_dashboard(frame, tracks, primary_id, state, match_result, db_count, fps, is_muted=False,
                    chat_log=None, chat_input_active=False, chat_input_buffer="", conversation_active=False):
    h, w = frame.shape[:2]

    # -- Side panel --
    panel_w = 420
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = C_PANEL

    px = w + 15  # panel x start
    py = 30      # panel y start
    chat_y_start = 0  # will be set after target info

    # Title
    cv2.putText(canvas, GREETER_NAME.upper(), (px, py), FONT, 0.9, C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "CHECK-IN SYSTEM", (px, py + 25), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)

    # Divider
    py += 45
    cv2.line(canvas, (px, py), (px + panel_w - 30, py), C_GRAY, 1)

    # State
    py += 30
    state_color = STATE_COLORS.get(state, C_WHITE)
    state_label = STATE_LABELS.get(state, state.name)
    cv2.circle(canvas, (px + 6, py - 5), 6, state_color, -1)
    cv2.putText(canvas, state_label, (px + 20, py), FONT, 0.55, state_color, 1, cv2.LINE_AA)

    # Active faces
    py += 35
    active_count = sum(1 for t in tracks.values() if t.frames_disappeared == 0)
    cv2.putText(canvas, "FACES DETECTED", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)
    py += 22
    cv2.putText(canvas, str(active_count), (px, py), FONT, 0.7, C_WHITE, 2, cv2.LINE_AA)

    # Database
    py += 35
    cv2.putText(canvas, "KNOWN PEOPLE", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)
    py += 22
    cv2.putText(canvas, str(db_count), (px, py), FONT, 0.7, C_WHITE, 2, cv2.LINE_AA)

    # Current target
    py += 35
    cv2.line(canvas, (px, py), (px + panel_w - 30, py), C_GRAY, 1)
    py += 25
    cv2.putText(canvas, "ACTIVE TARGET", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)
    py += 25

    if primary_id is not None and match_result:
        if match_result.person_name:
            cv2.putText(canvas, match_result.person_name, (px, py), FONT, 0.7, C_GREEN, 2, cv2.LINE_AA)
            py += 22
            cv2.putText(canvas, f"Confidence: {match_result.confidence:.0%}", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)
        elif match_result.status == "unknown":
            cv2.putText(canvas, "Unknown Person", (px, py), FONT, 0.6, C_ORANGE, 1, cv2.LINE_AA)
        elif match_result.status == "pending":
            cv2.putText(canvas, "Analyzing...", (px, py), FONT, 0.6, C_CYAN, 1, cv2.LINE_AA)
            py += 22
            cv2.putText(canvas, f"Confidence: {match_result.confidence:.0%}", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)
    elif primary_id is not None:
        cv2.putText(canvas, f"Track #{primary_id}", (px, py), FONT, 0.6, C_CYAN, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "None", (px, py), FONT, 0.6, C_GRAY, 1, cv2.LINE_AA)

    # Chat section starts here
    chat_y_start = py + 35

    # Mute button
    py = h - 75
    mute_color = C_RED if is_muted else C_GREEN
    mute_label = "MUTE: ON " if is_muted else "MUTE: OFF"
    draw_rounded_rect(canvas, (px, py - 16), (px + panel_w - 30, py + 8), mute_color, radius=5)
    cv2.putText(canvas, mute_label, (px + 8, py), FONT, 0.5, C_DARK, 1, cv2.LINE_AA)

    # FPS
    py = h - 40
    cv2.putText(canvas, f"FPS: {fps:.1f}", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)
    py += 20
    cv2.putText(canvas, "Q: Quit  B: Build  M: Mute", (px, py), FONT_SMALL, 1.0, C_GRAY, 1, cv2.LINE_AA)

    # Chat panel
    if chat_log:
        draw_chat_panel(canvas, px, panel_w, chat_y_start, h - 90, chat_log,
                        chat_input_active, chat_input_buffer, conversation_active)

    # -- Draw face boxes on the video portion --
    for tid, track in tracks.items():
        if track.frames_disappeared > 0:
            continue
        is_primary = tid == primary_id

        if is_primary:
            if match_result and match_result.person_name:
                color = C_GREEN
                label = match_result.person_name
            elif match_result and match_result.status == "unknown":
                color = C_ORANGE
                label = "Unknown"
            elif match_result and match_result.status == "pending":
                color = C_CYAN
                label = "Analyzing..."
            else:
                color = C_CYAN
                label = f"#{tid}"
        else:
            color = C_GRAY
            label = ""

        draw_face_box(canvas, track.bbox, color, label, is_primary)
        if is_primary:
            draw_crosshair(canvas, track.bbox, color)

    # Center crosshair (shows PTZ target zone)
    ch, cw = h // 2, w // 2
    cv2.line(canvas, (cw - 20, ch), (cw + 20, ch), (80, 80, 80), 1)
    cv2.line(canvas, (cw, ch - 20), (cw, ch + 20), (80, 80, 80), 1)

    # D-pad controls
    draw_dpad(canvas, h, w)

    return canvas


# ── Helpers ─────────────────────────────────────────────────────

def seed_database(database, matcher):
    import face_recognition
    if database.person_count() > 0:
        return
    ryan_path = os.path.join(os.path.dirname(__file__), "ryan.jpg")
    if not os.path.exists(ryan_path):
        return
    print("Seeding database with ryan.jpg...")
    img = face_recognition.load_image_file(ryan_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        database.add_person("Ryan", [encodings[0]])
        matcher.reload_database()
        print("Seeded: Ryan")


# ── Main Loop ───────────────────────────────────────────────────

def main():
    # Single-instance enforcement
    LOCKFILE = "/tmp/robo-greeter.pid"
    _my_pid = os.getpid()

    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, "r") as _f:
                _old_pid = int(_f.read().strip())
            if _old_pid != _my_pid:
                os.kill(_old_pid, signal.SIGTERM)
                time.sleep(0.5)
        except (ValueError, ProcessLookupError, PermissionError):
            pass

    with open(LOCKFILE, "w") as _f:
        _f.write(str(_my_pid))

    config = AppConfig()
    log = setup_logger("robo-greeter", config.log_level)
    log.info("Initializing Jarvis...")

    # Components
    database = FaceDatabase(config.db_path)
    camera = ThreadedCamera(config.camera)
    processor = FrameProcessor(config.camera)
    tracker = FaceTracker(config.tracking)
    selector = PrimarySubjectSelector(config.tracking, frame_shape=(1080, 1920))
    matcher = IdentityMatcher(config.identity, database)
    ptz = PTZController(config.ptz, config.camera)
    tts = TTSEngine(config.audio)
    stt = STTEngine(config.audio)

    # Chat log for captions
    chat_log = ChatLog()
    stt.transcription_callback = chat_log.add

    audio = AudioManager(tts, stt, config.audio, speak_callback=chat_log.add)
    greeter = GreeterLogic(audio, database, matcher, config.audio.gemini_api_key)
    sm = GreeterStateMachine(config.state, matcher)

    seed_database(database, matcher)

    # Processing state
    last_process_time = 0.0
    process_interval = 1.0 / config.camera.target_fps
    interaction_thread = None
    current_match_result = None
    fps = 0.0
    fps_counter = 0
    fps_timer = time.monotonic()
    db_count = database.person_count()

    # Chat input state
    chat_input_active = False
    chat_input_buffer = ""

    log.info("Jarvis online. Press 'q' to quit.")

    try:
        while True:
            ret, frame, ts = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()

            # FPS counter
            fps_counter += 1
            if now - fps_timer >= 1.0:
                fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now
                db_count = database.person_count()

            # Rate-limited face processing
            if now - last_process_time >= process_interval:
                last_process_time = now
                selector.update_frame_shape(frame.shape[:2])

                detections = processor.process(frame, ts)
                tracks = tracker.update(detections)
                active_tracks = tracker.get_active_tracks()

                primary_id = selector.select(tracks, sm.current_subject_id)

                # Update match result for UI
                if primary_id is not None and primary_id in active_tracks:
                    primary = active_tracks[primary_id]
                    if sm.state in (State.SCANNING, State.TRACKING, State.GREETING):
                        current_match_result = matcher.process_track(primary)
                else:
                    primary = None
                if sm.state == State.IDLE:
                    current_match_result = None

                # Is a conversation currently happening?
                conversation_active = (
                    interaction_thread is not None and interaction_thread.is_alive()
                )

                if conversation_active:
                    # Freeze state machine, just track for PTZ
                    output = StateMachineOutput()
                    if primary:
                        output.ptz_target_bbox = primary.bbox
                else:
                    output = sm.tick(active_tracks, primary_id)

                # PTZ
                if output.ptz_target_bbox:
                    ptz.update_target(output.ptz_target_bbox, frame.shape[:2])
                elif output.should_disengage_ptz:
                    ptz.clear_target()

                # Handle greeting request — launch full conversation thread
                if output.greeting_request and not conversation_active:
                    req = output.greeting_request
                    sm.freeze_for_speech()

                    # Collect unknown tracks for enrollment
                    unknown_tracks = []
                    for tid, result in sm._scan_results.items():
                        if result.status == "unknown" and tid in active_tracks:
                            unknown_tracks.append(active_tracks[tid])

                    def _converse(r, ut):
                        greeter.greet_and_enroll(
                            r.known_names, r.unknown_count, ut
                        )
                        sm.mark_greeting_done()

                    interaction_thread = threading.Thread(
                        target=_converse, args=(req, unknown_tracks), daemon=True
                    )
                    interaction_thread.start()

            else:
                tracks = tracker.tracks
                primary_id = sm.current_subject_id

            # Pass current frame to greeter for snapshots
            greeter.set_current_frame(frame)

            # Build and show UI
            canvas = build_dashboard(
                frame, tracks, primary_id, sm.state,
                current_match_result, db_count, fps,
                is_muted=tts.is_muted(),
                chat_log=chat_log,
                chat_input_active=chat_input_active,
                chat_input_buffer=chat_input_buffer,
                conversation_active=conversation_active,
            )

            try:
                cv2.imshow("Jarvis", canvas)
            except Exception as e:
                log.warning(f"Display error (window might be closed): {e}")
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                pass
            elif chat_input_active:
                if key == 13:  # Enter — submit
                    if chat_input_buffer.strip():
                        chat_log.add("You", chat_input_buffer.strip())
                    stt.feed_key("\r")
                    chat_input_buffer = ""
                    chat_input_active = False
                elif key in (8, 127):  # Backspace
                    chat_input_buffer = chat_input_buffer[:-1]
                    stt.feed_key("\x08")
                elif key == 27:  # Escape — cancel
                    chat_input_buffer = ""
                    chat_input_active = False
                elif 32 <= key < 127:
                    chat_input_buffer += chr(key)
                    stt.feed_key(chr(key))
            elif stt.waiting_for_input:
                if key == 13:
                    if stt.current_buffer.strip():
                        chat_log.add("You", stt.current_buffer.strip())
                    stt.feed_key("\r")
                elif key in (8, 127):
                    stt.feed_key("\x08")
                elif 32 <= key < 127:
                    stt.feed_key(chr(key))
            elif key == ord("q"):
                break
            elif key == ord("b"):
                seed_database(database, matcher)
            elif key == ord("m"):
                tts.toggle_mute()
            elif key == 27:
                pass
            elif conversation_active and 32 <= key < 127:
                # Any printable key during conversation → activate chat input
                chat_input_active = True
                chat_input_buffer = chr(key)
                stt.activate_keyboard_input()
                stt.feed_key(chr(key))
            else:
                # PTZ
                if key == 81:  # left arrow
                    ptz.manual_move("left")
                elif key == 82:  # up arrow
                    ptz.manual_move("up")
                elif key == 83:  # right arrow
                    ptz.manual_move("right")
                elif key == 84:  # down arrow
                    ptz.manual_move("down")
                elif key == ord("a"):
                    ptz.manual_move("left")
                elif key == ord("w"):
                    ptz.manual_move("up")
                elif key == ord("d"):
                    ptz.manual_move("right")

    except KeyboardInterrupt:
        log.info("Shutting down...")
    except Exception as e:
        log.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        camera.stop()
        ptz.clear_target()
        ptz.stop()
        database.close()
        try:
            os.remove(LOCKFILE)
        except FileNotFoundError:
            pass
        cv2.destroyAllWindows()

    log.info("Jarvis offline.")


if __name__ == "__main__":
    main()
