import os
import json
import datetime
from typing import List, Optional
from collections import deque

import google.genai as genai

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

    def __init__(self, audio: AudioManager, api_key: str):
        self.audio = audio
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-1.5-flash"  # Better free tier limits than 2.0
        self.conversation_history = deque(maxlen=6)  # Keep last 3 exchanges
        self.api_call_count = 0
        self.max_free_tier_calls = 3  # Rough limit for free tier per conversation

    def start_conversation(self, names: List[str]):
        """Begin a natural conversation after sign-in."""
        name_str = self._join_names(names)
        time_period = self._get_time_period()

        # Initial greeting + question
        initial_prompt = self._build_initial_prompt(name_str, time_period)
        initial_response = self._call_gemini(initial_prompt)

        self.audio.say(initial_response)
        self.conversation_history.append({"role": "assistant", "content": initial_response})

        # Listen to response
        user_input = self.audio.stt.listen(timeout=8.0)
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
        response = self._call_gemini(follow_up_prompt)

        self.audio.say(response)
        self.conversation_history.append({"role": "assistant", "content": response})

        # Listen for next input
        next_input = self.audio.stt.listen(timeout=8.0)
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

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API and return response. Falls back to local responses if quota exceeded."""
        try:
            log.info(f"Calling Gemini API with model: {self.model_name}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            text = response.text.strip()
            log.info(f"Gemini response: {text}")
            self.api_call_count += 1
            return text
        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower() or
                "404" in error_str or "NOT_FOUND" in error_str):
                log.warning(f"Gemini API unavailable ({error_str[:50]}...), using local response fallback")
                return self._get_local_response(prompt)
            else:
                log.error(f"Gemini API error: {e}")
                return "Sorry, let me try that again."

    def _get_local_response(self, prompt: str) -> str:
        """Generate a simple local response when API quota is exceeded."""
        # Simple rule-based conversation fallback
        if "how are you" in prompt.lower() or "how are you doing" in prompt.lower():
            return "I'm doing well, thank you for asking! How can I help you today?"
        elif "what" in prompt.lower() and ("name" in prompt.lower() or "you" in prompt.lower()):
            return "I'm Jarvis, your office greeter. Nice to meet you!"
        elif "how" in prompt.lower():
            return "That's an interesting question. I'm here to help with your check-in process."
        elif any(word in prompt.lower() for word in ["good", "great", "excellent", "thanks", "thank"]):
            return "That's wonderful! I'm glad to hear that. Is there anything else I can help you with?"
        else:
            return "That sounds interesting. Tell me more about that."

    def _generate_goodbye(self, name_str: str) -> str:
        """Generate personalized goodbye."""
        prompt = f"""Generate a warm, brief goodbye for {name_str}.
Keep it to 1 sentence. Be genuine and friendly.
Examples: "Great talking with you! Have a wonderful day!"
or "Take care, and see you soon!"

Respond ONLY with the goodbye message, nothing else."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            log.error(f"Goodbye generation error: {e}")
            return f"Take care, {name_str}!"

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
