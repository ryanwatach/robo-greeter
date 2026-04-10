import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CameraConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


class ThreadedCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.timestamp: float = 0.0
        self.ret: bool = False
        self.lock = threading.Lock()
        self.running = True
        self._connect()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        log.info("Connecting to camera: %s", self.config.rtsp_url)
        self.cap = cv2.VideoCapture(self.config.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        if self.cap.isOpened():
            log.info("Camera connected")
        else:
            log.error("Failed to open camera stream")

    def _capture_loop(self):
        failures = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                failures += 1
                wait = min(failures * 2, 30)
                log.warning("Camera disconnected, reconnecting in %ds...", wait)
                time.sleep(wait)
                self._connect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                failures += 1
                if failures > 20:
                    log.warning("Too many read failures, reconnecting...")
                    self.cap.release()
                    self._connect()
                    failures = 0
                continue

            failures = 0
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.timestamp = time.monotonic()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        with self.lock:
            if self.frame is None:
                return False, None, 0.0
            return self.ret, self.frame.copy(), self.timestamp

    def is_healthy(self) -> bool:
        with self.lock:
            if not self.ret:
                return False
            return (time.monotonic() - self.timestamp) < 5.0

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        log.info("Camera stopped")
