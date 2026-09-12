import os
import json
import datetime
import random
from typing import List, Optional
from collections import deque

import google.genai as genai
from google.genai import types as genai_types

from interaction.jarvis_context import build_system_instruction

from audio.audio_manager import AudioManager
from utils.logger import setup_logger

log = setup_logger("robo-greeter")

# Goodbye keywords
GOODBYE_KEYWORDS = {
    "bye", "goodbye", "see you", "later", "gotta go", "talk soon",
    "take care", "catch you", "until later", "farewell", "cya", "see ya",
    "have to go", "need to go", "outta here", "heading out"
}


class ConversationManager:
    """Real-time conversational AI powered by Google Gemini."""

    FACE_ABSENT_GOODBYE_SECONDS = 2.0  # how long face must be gone before we end the conversation

    def __init__(self, audio: AudioManager, api_key: str, face_absent_seconds=None, tools=None):
        self.audio = audio
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"  # 1.5 was retired
        self.conversation_history = deque(maxlen=6)  # Keep last 3 exchanges
        self.api_call_count = 0
        self.max_free_tier_calls = 3  # Rough limit for free tier per conversation
        self._face_absent_seconds = face_absent_seconds or (lambda: 999.0)
        # Plain Python callables exposed to Gemini for automatic function
        # calling. The SDK introspects type hints + docstrings to generate
        # the tool schema and invokes them when the model decides to call.
        self.tools = list(tools or [])
        # Persistent identity/location/role context for Jarvis. Sent as the
        # `system_instruction` so it stays in effect across all turns.
        self.system_instruction = build_system_instruction()

    def _listen_with_face_check(self, timeout: float = 8.0) -> Optional[str]:
        """Listen for STT input. If STT times out but the user is still
        visible (face seen within the last FACE_ABSENT_GOODBYE_SECONDS), keep
        listening. Returns the user input, or None only if the user has truly
        left (face absent past the threshold)."""
        while True:
            user_input = self.audio.stt.listen(timeout=timeout)
            if user_input:
                return user_input
            if self._face_absent_seconds() >= self.FACE_ABSENT_GOODBYE_SECONDS:
                return None

    def start_conversation(self, names: List[str]):
        """Begin a natural conversation after sign-in."""
        name_str = self._join_names(names)
        time_period = self._get_time_period()

        # Initial greeting + question
        initial_prompt = self._build_initial_prompt(name_str, time_period)
        initial_response = self._call_gemini(initial_prompt, kind="initial")

        self.audio.say(initial_response)
        self.conversation_history.append({"role": "assistant", "content": initial_response})

        # Listen to response
        user_input = self._listen_with_face_check(timeout=8.0)
        if not user_input:
            self.audio.say("No worries. Have a great day!")
            return

        # Check for goodbye
        if self._is_goodbye(user_input):
            goodbye = self._generate_goodbye(name_str)
            self.audio.say(goodbye)
            return

        # Continue conversation
        self._continue_conversation(user_input, name_str)

    def _continue_conversation(self, user_input: str, name_str: str, turns: int = 0):
        """Keep conversation going with context."""
        if turns >= 2:  # Limit to 2 follow-ups
            goodbye = self._generate_goodbye(name_str)
            self.audio.say(goodbye)
            return

        # Add to history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Check for goodbye in user input
        if self._is_goodbye(user_input):
            goodbye = self._generate_goodbye(name_str)
            self.audio.say(goodbye)
            return

        # Generate contextual response
        follow_up_prompt = self._build_follow_up_prompt(user_input, name_str)
        response = self._call_gemini(follow_up_prompt, kind="followup")

        self.audio.say(response)
        self.conversation_history.append({"role": "assistant", "content": response})

        # Listen for next input
        next_input = self._listen_with_face_check(timeout=8.0)
        if not next_input:
            goodbye = self._generate_goodbye(name_str)
            self.audio.say(goodbye)
            return

        # Recursive continue
        self._continue_conversation(next_input, name_str, turns + 1)

    def _build_initial_prompt(self, name_str: str, time_period: str) -> str:
        """Build system prompt for initial greeting + question."""
        return f"""You are Jarvis, a warm and friendly office greeter AI. You're having a natural conversation.

Context:
- Name(s): {name_str}
- Time: {time_period}
- Your greeting just said they're signed in

Generate a natural follow-up question asking how they're doing.
Be conversational, brief (1-2 sentences), and genuine.
No greetings like "Good morning" again - just ask how they are.

Example: "How are you doing today?" or "How's everything going with you?"

Respond naturally as Jarvis would."""

    def _build_follow_up_prompt(self, user_input: str, name_str: str) -> str:
        """Build prompt for follow-up response with history."""
        history_text = "\n".join([
            f"{msg['role'].title()}: {msg['content']}"
            for msg in self.conversation_history
        ])

        return f"""You are Jarvis, a friendly office greeter. Continue the conversation naturally.

Context:
- Name(s): {name_str}
- Time: {self._get_time_period()}

Conversation so far:
{history_text}
User: {user_input}

Respond naturally (1-2 sentences max). Be warm, genuine, and conversational.
Ask a follow-up question to keep them engaged, OR wrap up the conversation gracefully.
Keep it brief and natural."""

    def _call_gemini(self, prompt: str, kind: str = "general") -> str:
        """Call Gemini API and return response. Falls back to a context-aware
        local response if Gemini is unavailable (quota, model 404, etc.).
        `kind` is "initial" (Jarvis asks how the user is), "followup"
        (Jarvis responds to user input), "goodbye", or "general"."""
        try:
            log.info(f"Calling Gemini API with model: {self.model_name}")
            # Build the request config: always include system_instruction so
            # Jarvis stays in character + grounded in lab/location. Only
            # attach tools on follow-up turns — initial/goodbye prompts are
            # one-shot generations that don't need to call back into the DB.
            cfg_kwargs = {"system_instruction": self.system_instruction}
            if self.tools and kind == "followup":
                cfg_kwargs["tools"] = self.tools
            cfg = genai_types.GenerateContentConfig(**cfg_kwargs)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg,
            )
            text = (response.text or "").strip()
            log.info(f"Gemini response: {text}")
            self.api_call_count += 1
            return text
        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower() or
                "404" in error_str or "NOT_FOUND" in error_str):
                log.warning(f"Gemini API unavailable ({error_str[:50]}...), using local response fallback")
                return self._get_local_response(prompt, kind)
            else:
                log.error(f"Gemini API error: {e}")
                return "Sorry, let me try that again."

    def _get_local_response(self, prompt: str, kind: str = "general") -> str:
        """Context-aware local fallback. The previous version inspected the
        prompt text and matched 'how are you' in the META instructions,
        which made Jarvis answer "I'm doing well..." even though Jarvis was
        the one supposed to be asking. `kind` disambiguates."""
        if kind == "initial":
            # Jarvis is supposed to ASK how the user is doing.
            return random.choice([
                "How are you doing today?",
                "How's everything going with you?",
                "How are you feeling so far today?",
                "How's your day been?",
            ])

        if kind == "goodbye":
            return "Take care, have a wonderful day!"

        if kind == "followup":
            lower = prompt.lower()
            # Pull just the user's last line out of the history-rich prompt
            user_tail = ""
            for line in reversed(prompt.splitlines()):
                if line.lower().startswith("user:"):
                    user_tail = line.split(":", 1)[1].strip().lower()
                    break
            text = user_tail or lower
            if any(w in text for w in ("good", "great", "excellent", "fine", "well", "alright", "okay", "ok")):
                return "Glad to hear that! Anything I can help you with before you head in?"
            if any(w in text for w in ("bad", "tired", "rough", "stressed", "not great")):
                return "Sorry to hear that. I hope your day gets better from here."
            return "Got it. Anything I can help you with before you head in?"

        return "I'm here to help with your check-in."

    def _generate_goodbye(self, name_str: str) -> str:
        """Generate personalized goodbye."""
        prompt = f"""Generate a warm, brief goodbye for {name_str}.
Keep it to 1 sentence. Be genuine and friendly.
Examples: "Great talking with you! Have a wonderful day!"
or "Take care, and see you soon!"

Respond ONLY with the goodbye message, nothing else."""
        return self._call_gemini(prompt, kind="goodbye")

    def _is_goodbye(self, text: str) -> bool:
        """Detect goodbye keywords."""
        lower = text.lower().strip()
        for keyword in GOODBYE_KEYWORDS:
            if keyword in lower:
                return True
        return False

    def _get_time_period(self) -> str:
        """Get time of day."""
        hour = datetime.datetime.now().hour
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        else:
            return "Evening"

    @staticmethod
    def _join_names(names: List[str]) -> str:
        """Join names naturally."""
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return ", ".join(names[:-1]) + f", and {names[-1]}"
