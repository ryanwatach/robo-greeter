import threading
import time
import os
from typing import Optional

import numpy as np

from config import AudioConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


def _has_microphone() -> bool:
    try:
        import sounddevice as sd
        sd.check_input_settings()
        return True
    except Exception:
        return False


class STTEngine:
    """
    Speech-to-text engine.
    Uses Whisper if a mic is available, otherwise provides a shared text buffer
    that the main loop can write to via OpenCV key capture.
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        self._model = None
        self._model_lock = threading.Lock()
        self._has_mic = _has_microphone()

        # Shared state for UI-based text input (no mic fallback)
        self._input_buffer = ""
        self._input_ready = threading.Event()
        self._waiting_for_input = False
        self._input_lock = threading.Lock()
        self._keyboard_override = False
        self.transcription_callback = None

        if self._has_mic:
            log.info("STT: microphone detected, using Whisper")
        else:
            log.info("STT: no microphone, using on-screen keyboard input")

    @property
    def waiting_for_input(self) -> bool:
        return self._waiting_for_input

    @property
    def current_buffer(self) -> str:
        with self._input_lock:
            return self._input_buffer

    def feed_key(self, char: str):
        """Called by the main loop when a key is pressed in the OpenCV window."""
        with self._input_lock:
            if not self._waiting_for_input:
                return
            if char == "\r" or char == "\n":
                # Enter pressed — submit
                self._input_ready.set()
            elif char == "\x08" or char == "\x7f":
                # Backspace
                self._input_buffer = self._input_buffer[:-1]
            elif char.isprintable():
                self._input_buffer += char

    def activate_keyboard_input(self):
        """Override mic recording to accept keyboard input instead."""
        with self._input_lock:
            self._input_buffer = ""
            self._input_ready.clear()
            self._waiting_for_input = True
            self._keyboard_override = True

    def listen(self, timeout: Optional[float] = None) -> Optional[str]:
        timeout = timeout or self.config.listen_timeout

        if self._has_mic:
            audio = self._record_until_silence(timeout)
            if audio is None:
                if self._waiting_for_input:
                    return self._ui_input(timeout, already_waiting=True)
                return None
            return self._transcribe(audio)
        else:
            return self._ui_input(timeout)

    def _ui_input(self, timeout: float, already_waiting: bool = False) -> Optional[str]:
        """Wait for text input from the OpenCV window."""
        if not already_waiting:
            with self._input_lock:
                self._input_buffer = ""
                self._input_ready.clear()
                self._waiting_for_input = True

        log.info("STT: waiting for on-screen input (%.0fs timeout)...", timeout)

        # Wait for Enter key or timeout
        got_input = self._input_ready.wait(timeout=timeout)

        with self._input_lock:
            self._waiting_for_input = False
            text = self._input_buffer.strip()
            self._input_buffer = ""

        if got_input and text:
            log.info("STT input received: '%s'", text)
            return text

        log.info("STT: no input received (timeout)")
        return None

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    import whisper
                    log.info("Loading Whisper model '%s'...", self.config.whisper_model)
                    self._model = whisper.load_model(self.config.whisper_model)
                    log.info("Whisper model loaded")
        return self._model

    def _record_until_silence(self, timeout: float) -> Optional[np.ndarray]:
        import sounddevice as sd

        try:
            sd.check_input_settings()
        except Exception:
            return None

        sr = self.config.sample_rate
        chunk_dur = 0.5
        chunk_samples = int(sr * chunk_dur)
        silence_limit = self.config.silence_threshold
        energy_threshold = 0.01

        chunks = []
        speaking = False
        silence_start = None
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            with self._input_lock:
                if self._keyboard_override:
                    self._keyboard_override = False
                    return None
            audio = sd.rec(chunk_samples, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            chunk = audio.flatten()
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > energy_threshold:
                speaking = True
                silence_start = None
                chunks.append(chunk)
            elif speaking:
                chunks.append(chunk)
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start > silence_limit:
                    break

        if not chunks:
            return None
        return np.concatenate(chunks)

    def _transcribe(self, audio: np.ndarray) -> str:
        model = self._get_model()
        result = model.transcribe(audio, language="en", fp16=False)
        text = result["text"].strip()
        log.info("STT transcribed: '%s'", text)
        if text and self.transcription_callback:
            self.transcription_callback("You", text)
        return text
