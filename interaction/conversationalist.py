import datetime
import random
from typing import List, Optional

from audio.audio_manager import AudioManager
from utils.logger import setup_logger

log = setup_logger("robo-greeter")

# Goodbye keywords
GOODBYE_KEYWORDS = {
    "bye", "goodbye", "see you", "later", "gotta go", "talk soon",
    "take care", "catch you", "until later", "farewell", "cya"
}


class Conversationalist:
    """Handles post-sign-in conversations with natural dialogue."""

    def __init__(self, audio: AudioManager):
        self.audio = audio

    def start_conversation(self, names: List[str]):
        """Begin a natural conversation after sign-in."""
        greeting = self._get_time_greeting()

        if len(names) == 1:
            self.audio.say(f"{greeting}, {names[0]}. How are you doing today?")
        else:
            name_list = self._join_names(names)
            self.audio.say(f"{greeting}, {name_list}. How are you all doing today?")

        # Listen for response
        response = self.audio.stt.listen(timeout=8.0)
        if not response:
            self.audio.say("No worries. Have a great day!")
            return

        # Detect if they said goodbye
        if self._is_goodbye(response):
            self.audio.say(self._get_goodbye_phrase(names))
            return

        # Acknowledge their response and continue
        self._continue_conversation(response, names)

    def _continue_conversation(self, initial_response: str, names: List[str]):
        """Keep the conversation going with follow-ups."""
        # Acknowledge their answer
        acknowledgment = self._get_acknowledgment(initial_response)
        self.audio.say(acknowledgment)

        # Ask a follow-up question
        follow_up = self._get_follow_up(initial_response)
        response = self.audio.ask(follow_up, timeout=8.0)

        if not response:
            self.audio.say(self._get_goodbye_phrase(names))
            return

        # Check for goodbye
        if self._is_goodbye(response):
            self.audio.say(self._get_goodbye_phrase(names))
            return

        # One more round if they keep engaging
        self.audio.say(random.choice([
            "That sounds nice!",
            "Got it!",
            "I appreciate you sharing that.",
        ]))

    def _get_time_greeting(self) -> str:
        """Return greeting based on time of day."""
        hour = datetime.datetime.now().hour

        if 6 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 18:
            return "Good afternoon"
        else:
            return "Good evening"

    def _get_acknowledgment(self, response: str) -> str:
        """Generate warm acknowledgment of their response."""
        response_lower = response.lower()

        # Detect sentiment
        positive_words = {"good", "great", "excellent", "wonderful", "amazing", "fantastic", "awesome"}
        negative_words = {"bad", "terrible", "awful", "horrible", "sick", "tired"}

        has_positive = any(word in response_lower for word in positive_words)
        has_negative = any(word in response_lower for word in negative_words)

        if has_positive:
            return random.choice([
                "That's great to hear!",
                "Wonderful!",
                "I'm glad to hear that!",
                "That's fantastic!",
            ])
        elif has_negative:
            return random.choice([
                "I hope things improve for you.",
                "Hang in there!",
                "Sorry to hear that.",
                "I hope tomorrow is better.",
            ])
        else:
            return random.choice([
                "Got it!",
                "I see.",
                "That's interesting.",
                "Thanks for sharing.",
            ])

    def _get_follow_up(self, response: str) -> str:
        """Generate a follow-up question based on their response."""
        response_lower = response.lower()

        positive_words = {"good", "great", "excellent", "wonderful", "amazing"}
        has_positive = any(word in response_lower for word in positive_words)

        if has_positive:
            return random.choice([
                "Anything interesting planned for today?",
                "Got any fun plans?",
                "Anything exciting happening?",
            ])
        else:
            return random.choice([
                "Is there anything I can help with?",
                "Anything I can do to make your day better?",
                "Let me know if you need anything.",
            ])

    def _is_goodbye(self, text: str) -> bool:
        """Check if the user is saying goodbye."""
        lower = text.lower().strip()
        for keyword in GOODBYE_KEYWORDS:
            if keyword in lower:
                return True
        return False

    def _get_goodbye_phrase(self, names: List[str]) -> str:
        """Generate a friendly goodbye."""
        if len(names) == 1:
            return random.choice([
                f"Great talking with you, {names[0]}. Have a wonderful day!",
                f"It was good seeing you, {names[0]}. Take care!",
                f"See you later, {names[0]}!",
                f"Have a great day, {names[0]}!",
            ])
        else:
            name_list = self._join_names(names)
            return random.choice([
                f"Great talking with you all. Have a wonderful day!",
                f"It was good seeing you. Take care!",
                f"See you all later!",
                f"Have a great day, everyone!",
            ])

    @staticmethod
    def _join_names(names: List[str]) -> str:
        """Join names naturally: 'Alice and Bob' or 'Alice, Bob, and Charlie'."""
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return ", ".join(names[:-1]) + f", and {names[-1]}"
