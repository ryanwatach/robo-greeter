import threading
import time
from typing import Optional, Tuple

from amcrest import AmcrestCamera

from config import CameraConfig, PTZConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")

HOME_TIMEOUT = 7.0  # seconds with no face before returning home


class PTZController:
    def __init__(self, ptz_config: PTZConfig, camera_config: CameraConfig):
        self.config = ptz_config
        self.cam = AmcrestCamera(
            camera_config.host,
            camera_config.port,
            camera_config.user,
            camera_config.password,
        ).camera

        self._target_offset: Optional[Tuple[float, float]] = None
        self._target_time: float = 0.0
        self._smoothed_x: float = 0.0
        self._smoothed_y: float = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._last_command_time: float = 0.0
        self._stale_threshold: float = 0.5
        self._last_active_time: float = time.monotonic()
        self._is_home: bool = True
        # Hysteresis: once the face is parked inside dead_zone, require a
        # larger movement (unlock_threshold) before correcting again. This
        # kills tiny rubber-banding at center while staying responsive when
        # the face actually moves.
        self._locked_x: bool = True
        self._locked_y: bool = True

        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        log.info("PTZ controller started")

    def update_target(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]):
        top, right, bottom, left = bbox
        face_cx = (left + right) / 2.0
        face_cy = (top + bottom) / 2.0
        frame_h, frame_w = frame_shape

        offset_x = (face_cx - frame_w / 2.0) / (frame_w / 2.0)
        offset_y = (face_cy - frame_h / 2.0) / (frame_h / 2.0)

        with self._lock:
            self._target_offset = (offset_x, offset_y)
            self._target_time = time.monotonic()
            self._last_active_time = time.monotonic()
            self._is_home = False

    def clear_target(self):
        with self._lock:
            self._target_offset = None
            self._smoothed_x = 0.0
            self._smoothed_y = 0.0

    def go_home(self):
        try:
            self.cam.go_to_preset(preset_point_number=1)
            log.info("PTZ returning to home position")
        except Exception as e:
            log.warning("PTZ go_home failed: %s", e)

    def stop(self):
        self._running = False
        log.info("PTZ controller stopped")

    def manual_move(self, direction: str, speed: int = 2, duration: float = 0.15):
        """Immediately move camera in given direction for manual keyboard control.
        Bypasses auto-tracking loop. Thread-safe."""
        threading.Thread(
            target=self._move,
            args=(direction, speed, duration),
            daemon=True
        ).start()
        with self._lock:
            self._last_active_time = time.monotonic()
            self._is_home = False

    def _control_loop(self):
        while self._running:
            time.sleep(0.05)
            now = time.monotonic()

            # Auto-return home if no face for HOME_TIMEOUT seconds
            with self._lock:
                idle_time = now - self._last_active_time
                is_home = self._is_home

            if idle_time > HOME_TIMEOUT and not is_home:
                self.clear_target()
                self.go_home()
                with self._lock:
                    self._is_home = True
                continue

            # Rate limit
            if now - self._last_command_time < self.config.update_interval:
                continue

            with self._lock:
                offset = self._target_offset
                if offset is None:
                    continue
                if now - self._target_time > self._stale_threshold:
                    self._target_offset = None
                    self._smoothed_x = 0.0
                    self._smoothed_y = 0.0
                    continue
                raw_x, raw_y = offset

            alpha = self.config.smoothing_alpha
            self._smoothed_x = alpha * raw_x + (1 - alpha) * self._smoothed_x
            self._smoothed_y = alpha * raw_y + (1 - alpha) * self._smoothed_y

            dx = self._smoothed_x
            dy = self._smoothed_y

            moved = False
            max_dur = self.config.move_duration
            gain = self.config.move_gain
            max_spd_h = self.config.horizontal_speed
            max_spd_v = self.config.vertical_speed

            # Hysteresis: when "locked" (parked near center), require a larger
            # offset to re-engage. Once we move, drop back to dead_zone for
            # tight tracking until we re-center.
            engage_x = self.config.unlock_threshold_x if self._locked_x else self.config.dead_zone_x
            engage_y = self.config.unlock_threshold_y if self._locked_y else self.config.dead_zone_y

            if abs(dx) > engage_x:
                mag = abs(dx)
                spd = max(1, min(max_spd_h, int(mag * max_spd_h + 0.5)))
                dur_x = min(max_dur, mag * gain)
                direction = "right" if dx > 0 else "left"
                self._move(direction, spd, dur_x)
                moved = True
                self._locked_x = False
            elif abs(dx) < self.config.dead_zone_x:
                self._locked_x = True

            if abs(dy) > engage_y:
                mag = abs(dy)
                spd = max(1, min(max_spd_v, int(mag * max_spd_v + 0.5)))
                dur_y = min(max_dur, mag * gain)
                direction = "down" if dy > 0 else "up"
                self._move(direction, spd, dur_y)
                moved = True
                self._locked_y = False
            elif abs(dy) < self.config.dead_zone_y:
                self._locked_y = True

            if moved:
                self._last_command_time = now

    def _move(self, direction: str, speed: int, duration: float):
        try:
            if direction in ("up", "down"):
                fn = self.cam.move_up if direction == "up" else self.cam.move_down
                fn(start=True, vertical_speed=speed)
                time.sleep(duration)
                fn(start=False)
            else:
                fn = self.cam.move_left if direction == "left" else self.cam.move_right
                fn(start=True, horizontal_speed=speed)
                time.sleep(duration)
                fn(start=False)
        except Exception as e:
            log.warning("PTZ move %s failed: %s", direction, e)
