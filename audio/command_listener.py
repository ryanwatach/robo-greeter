import os
import sys
import threading
import time
from typing import Optional, Callable
from audio.stt import STTEngine
from config import AudioConfig
from utils.logger import setup_logger

log = setup_logger("robo-greeter")

# Command keywords and handlers
COMMANDS = {
    "kill all": "kill_all",
    "kill instances": "kill_all",
    "stop all": "kill_all",
    "restart": "restart",
    "status": "status",
    "quit": "quit",
    "exit": "quit",
    "hello jarvis": "hello",
    "jarvis": "hello",
}


class CommandListener:
    """
    Background thread that listens for voice commands while Jarvis is idle.
    Allows system-level control via natural speech.
    """

    def __init__(self, stt: STTEngine):
        self.stt = stt
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handlers = {}

    def register_handler(self, command: str, handler: Callable):
        """Register a handler for a specific command."""
        self._handlers[command] = handler

    def start(self):
        """Start the command listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        log.info("Command listener started")

    def stop(self):
        """Stop the command listener thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        log.info("Command listener stopped")

    def _listen_loop(self):
        """Main listening loop - runs in background thread."""
        while self._running:
            try:
                # Short timeout to remain responsive
                text = self.stt.listen(timeout=5.0)
                if text:
                    self._process_command(text)
            except Exception as e:
                log.debug(f"Command listener error: {e}")
                continue

    def _process_command(self, text: str) -> bool:
        """
        Process voice command.
        Returns True if command was handled, False otherwise.
        """
        text_lower = text.lower().strip()
        log.info(f"Command detected: '{text}'")

        # Check for matching commands
        for keyword, command in COMMANDS.items():
            if keyword in text_lower:
                if command in self._handlers:
                    log.info(f"Executing command: {command}")
                    self._handlers[command]()
                    return True
                else:
                    log.warning(f"No handler for command: {command}")
                    return True

        log.debug(f"Unrecognized command: {text}")
        return False


class SystemCommandHandler:
    """Handles system-level commands triggered by voice."""

    def __init__(self, on_kill_all: Optional[Callable] = None,
                 on_restart: Optional[Callable] = None):
        self.on_kill_all = on_kill_all
        self.on_restart = on_restart

    def kill_all(self):
        """Kill all instances."""
        log.warning("KILL COMMAND RECEIVED - Shutting down all instances")
        if self.on_kill_all:
            self.on_kill_all()
        # Force exit
        os._exit(0)

    def restart(self):
        """Restart the system."""
        log.warning("RESTART COMMAND RECEIVED - Restarting system")
        if self.on_restart:
            self.on_restart()
        # Kill and restart - will be picked up by systemd/nohup
        os._exit(1)

    def status(self):
        """Report system status."""
        log.info("STATUS COMMAND - System is online")

    def hello(self):
        """Respond to greeting."""
        log.info("HELLO COMMAND - Acknowledged")
