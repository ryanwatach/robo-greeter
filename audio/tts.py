import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

from config import AudioConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


class TTSEngine:
    """Text-to-speech with three tiers:

    1. ElevenLabs (preferred) — generates MP3 via API, plays through `afplay`
       subprocess so the audio device is fully released afterwards.
    2. macOS `say` — local high-quality fallback if ElevenLabs is unavailable.
    3. pyttsx3 — last-resort for non-macOS hosts. NOTE: on macOS pyttsx3 holds
       a Core Audio session that can SIGKILL the process when `sounddevice.rec`
       tries to claim the mic immediately afterwards — so we avoid it on Darwin.
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self._speak_lock = threading.Lock()
        self._muted: bool = False
        self._use_say = sys.platform == "darwin" and shutil.which("say") is not None
        self._afplay_available = shutil.which("afplay") is not None
        self._eleven_key = config.elevenlabs_api_key
        self._eleven_voice_id = config.elevenlabs_voice_id
        self._eleven_client = None
        self._eleven_disabled = False
        self._pyttsx_engine = None

        if self._eleven_key and self._eleven_voice_id and self._afplay_available:
            try:
                from elevenlabs.client import ElevenLabs
                self._eleven_client = ElevenLabs(api_key=self._eleven_key)
                log.info("TTS: ElevenLabs ready (voice_id=%s...)",
                         self._eleven_voice_id[:6])
            except Exception as e:
                log.warning("TTS: ElevenLabs init failed: %s", e)
                self._eleven_disabled = True

        if not self._eleven_client and self._use_say:
            log.info("TTS: using macOS 'say' command")
        elif not self._eleven_client and not self._use_say:
            import pyttsx3
            self._pyttsx_engine = pyttsx3.init()
            self._pyttsx_engine.setProperty("rate", 150)
            self._pyttsx_engine.setProperty("volume", 0.9)
            log.info("TTS: using pyttsx3 (local text-to-speech)")

    def speak(self, text: str, blocking: bool = True):
        if blocking:
            self._do_speak(text)
        else:
            t = threading.Thread(target=self._do_speak, args=(text,), daemon=True)
            t.start()

    def _do_speak(self, text: str):
        with self._speak_lock:
            if self._muted:
                return
            if self._eleven_client and not self._eleven_disabled:
                if self._speak_eleven(text):
                    return
                # fall through on failure
            if self._use_say:
                try:
                    subprocess.run(["say", "-r", "200", text], check=False)
                    time.sleep(0.1)
                    return
                except Exception as e:
                    log.error("TTS 'say' error: %s", e)
            if self._pyttsx_engine is not None:
                try:
                    self._pyttsx_engine.say(text)
                    self._pyttsx_engine.runAndWait()
                except Exception as e:
                    log.error("TTS pyttsx3 error: %s", e)

    def _speak_eleven(self, text: str) -> bool:
        """Synthesize via ElevenLabs and play through `afplay`. Returns True
        on success; on any failure logs and returns False so the caller can
        fall back."""
        try:
            audio_iter = self._eleven_client.text_to_speech.convert(
                voice_id=self._eleven_voice_id,
                text=text,
                model_id="eleven_flash_v2_5",  # fastest tier for conversation
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio_iter)
            if not audio_bytes:
                log.warning("TTS: ElevenLabs returned empty audio")
                return False

            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            try:
                subprocess.run(["afplay", tmp.name], check=False)
                time.sleep(0.1)
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            return True
        except Exception as e:
            log.warning("TTS: ElevenLabs failed (%s) — falling back", e)
            # Don't permanently disable on first failure; auth errors will keep
            # failing and we'll keep logging, but transient hiccups recover.
            return False

    def stop(self):
        if self._pyttsx_engine is not None:
            try:
                self._pyttsx_engine.stop()
            except Exception:
                pass

    def toggle_mute(self):
        self._muted = not self._muted
        log.info("TTS: mute %s", "ON" if self._muted else "OFF")

    def is_muted(self) -> bool:
        return self._muted

    def is_speaking(self) -> bool:
        return self._speak_lock.locked()
