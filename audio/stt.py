import threading
import time
import os
from collections import deque
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
    except Exception as e:
        log.warning("STT mic check failed: %s: %s", type(e).__name__, e)
        return False


def _find_input_device():
    """Pick a stable input-device index. Prefers any device whose name
    contains 'yeti', 'microphone', or 'mic' (in that order). Returns None if
    nothing matches — in which case we fall back to sounddevice's default."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception:
        return None
    candidates = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = dev.get("name", "").lower()
        score = None
        if "yeti" in name:
            score = 0
        elif "microphone" in name:
            score = 1
        elif "mic" in name:
            score = 2
        if score is not None:
            candidates.append((score, idx, name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _reset_portaudio():
    """PortAudio sometimes wedges its host-API state after another process
    grabs Core Audio (e.g. afplay finishing playback). Terminate + reinit to
    get a fresh session."""
    try:
        import sounddevice as sd
        sd._terminate()
        sd._initialize()
        return True
    except Exception as e:
        log.error("PortAudio reset failed: %s", e)
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

        # Pin a specific input device by name. PortAudio's idea of "default"
        # can shift when other apps grab Core Audio; an explicit index sticks.
        self._input_device = _find_input_device() if self._has_mic else None

        if self._has_mic:
            try:
                import sounddevice as sd
                dev_info = sd.query_devices(self._input_device) if self._input_device is not None else None
                dev_name = dev_info["name"] if dev_info else "default"
                log.info("STT: microphone detected (using '%s'), Whisper enabled", dev_name)
            except Exception:
                log.info("STT: microphone detected, using Whisper")
            # Preload Whisper in the background so the first listen() doesn't
            # eat 1-2 seconds of model-load latency right when the user speaks.
            threading.Thread(target=self._get_model, daemon=True).start()
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
        """Capture from the mic in short chunks until the user stops talking.

        Improvements over the previous version (which felt sluggish):
        - 100ms chunks instead of 500ms → speech-start latency drops 5×.
        - 300ms pre-roll buffer keeps the audio just *before* voicing was
          detected, so we never clip the first phoneme of a word.
        - Energy threshold is auto-calibrated from the first 400ms of room
          tone, scaled 4× above noise floor, with a floor and ceiling.
        - Shorter silence cutoff (0.8s) so we transcribe faster after the
          user finishes a sentence.
        """
        import sounddevice as sd

        device = self._input_device

        def _rec(n):
            return sd.rec(
                n, samplerate=sr, channels=1, dtype="float32",
                device=device,
            )

        # Verify the device works right now. If PortAudio's host state has
        # wedged (paInternalError after a prior afplay session, etc.), reset
        # and try once more.
        for attempt in (1, 2):
            try:
                sd.check_input_settings(device=device) if device is not None else sd.check_input_settings()
                break
            except Exception as e:
                log.warning("STT mic unavailable (attempt %d): %s", attempt, e)
                if attempt == 1 and _reset_portaudio():
                    continue
                return None

        sr = self.config.sample_rate
        chunk_dur = 0.10
        chunk_samples = int(sr * chunk_dur)
        silence_limit = 0.8

        # Calibrate noise floor from the first few chunks of room tone.
        calibration_chunks = 4  # 400ms
        cal_rms = []
        for _ in range(calibration_chunks):
            try:
                a = _rec(chunk_samples)
                sd.wait()
            except Exception as e:
                log.error("STT calibration error: %s: %s", type(e).__name__, e)
                return None
            cal_rms.append(float(np.sqrt(np.mean(a.flatten() ** 2))))
        noise = float(np.median(cal_rms))
        # Speech needs to be clearly above noise — 4× with floor/ceiling.
        energy_threshold = max(0.003, min(0.05, noise * 4.0))
        log.info("STT: noise=%.4f threshold=%.4f", noise, energy_threshold)

        pre_roll = deque(maxlen=3)  # 300ms of audio just before speech started
        chunks = []
        speaking = False
        silence_start: Optional[float] = None
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            with self._input_lock:
                if self._keyboard_override:
                    self._keyboard_override = False
                    return None
            try:
                audio = _rec(chunk_samples)
                sd.wait()
            except Exception as e:
                log.error("STT recording error: %s: %s", type(e).__name__, e)
                return None
            chunk = audio.flatten()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if not speaking:
                pre_roll.append(chunk)
                if rms > energy_threshold:
                    speaking = True
                    chunks.extend(pre_roll)
                    silence_start = None
                    log.info("STT: speech detected (rms=%.4f)", rms)
            else:
                chunks.append(chunk)
                if rms > energy_threshold:
                    silence_start = None
                else:
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
