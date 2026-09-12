from typing import Optional

from audio.tts import TTSEngine
from audio.stt import STTEngine
from config import AudioConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")


class AudioManager:
    def __init__(self, tts: TTSEngine, stt: STTEngine, config: AudioConfig, speak_callback=None):
        self.tts = tts
        self.stt = stt
        self.config = config
        self.speak_callback = speak_callback

    def say(self, text: str):
        log.info("Speaking: %s", text)
        if self.speak_callback:
            self.speak_callback("Jarvis", text)
        self.tts.speak(text, blocking=True)

    def ask(self, prompt: str, timeout: Optional[float] = None) -> Optional[str]:
        self.say(prompt)
        response = self.stt.listen(timeout=timeout)
        return response

    def ask_yes_no(self, prompt: str) -> Optional[bool]:
        response = self.ask(prompt)
        if response is None:
            return None
        lower = response.lower().strip()
        yes_words = {"yes", "yeah", "yep", "sure", "correct", "yup", "uh huh", "right"}
        no_words = {"no", "nah", "nope", "wrong", "incorrect", "not"}
        for w in yes_words:
            if w in lower:
                return True
        for w in no_words:
            if w in lower:
                return False
        return None

    def interrupt(self):
        self.tts.stop()
