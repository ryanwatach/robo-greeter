import threading
from typing import Optional

import pyttsx3

from config import AudioConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


class TTSEngine:
    def __init__(self, config: AudioConfig):
        self.config = config
        self._speak_lock = threading.Lock()  # only one utterance at a time
        self._muted: bool = False
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed: slower for clarity
        self.engine.setProperty('volume', 0.9)  # Volume: 0-1
        log.info("TTS: using pyttsx3 (local text-to-speech)")

    def speak(self, text: str, blocking: bool = True):
        if blocking:
            self._do_speak(text)
        else:
            t = threading.Thread(target=self._do_speak, args=(text,), daemon=True)
            t.start()

    def _do_speak(self, text: str):
        # Block until any current speech finishes — prevents overlapping audio
        with self._speak_lock:
            if self._muted:
                return
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                log.error("TTS error: %s", e)

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass

    def toggle_mute(self):
        self._muted = not self._muted
        log.info("TTS: mute %s", "ON" if self._muted else "OFF")

    def is_muted(self) -> bool:
        return self._muted

    def is_speaking(self) -> bool:
        return self._speak_lock.locked()
