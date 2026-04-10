import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)


@dataclass
class CameraConfig:
    # Camera source type: "rtsp" for network camera, "webcam" for local device
    source_type: str = "rtsp"  # "rtsp" or "webcam"

    # For RTSP cameras (Amcrest, etc.)
    host: str = "192.168.1.108"
    port: int = 80
    user: str = "admin"
    password: str = "admin"
    rtsp_url: str = ""

    # For local webcam
    webcam_device: int = 0  # Device index (0 = default/built-in camera)

    processing_scale: float = 0.25
    target_fps: float = 4.0
    buffer_size: int = 1

    def __post_init__(self):
        if self.source_type == "rtsp" and not self.rtsp_url:
            self.rtsp_url = (
                f"rtsp://{self.user}:{self.password}@{self.host}:554"
                f"/cam/realmonitor?channel=1&subtype=0"
            )


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_disappeared: int = 40
    center_weight: float = 0.3
    size_weight: float = 0.5
    persistence_weight: float = 0.2
    hysteresis_bonus: float = 0.15


@dataclass
class IdentityConfig:
    distance_threshold: float = 0.6  # More forgiving distance (was 0.45)
    confirmation_frames: int = 8     # More frames for stability (was 5)
    confirmation_ratio: float = 0.6  # Lower threshold for consensus (was 0.7) - need 5/8 not 4/5
    embedding_buffer_size: int = 8
    unknown_threshold_frames: int = 8


@dataclass
class PTZConfig:
    dead_zone_x: float = 0.70
    dead_zone_y: float = 0.70
    horizontal_speed: int = 1
    vertical_speed: int = 1
    update_interval: float = 1.2
    smoothing_alpha: float = 0.10
    move_duration: float = 0.06


@dataclass
class AudioConfig:
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    whisper_model: str = "base"
    gemini_api_key: str = ""
    listen_timeout: float = 30.0
    silence_threshold: float = 1.5
    max_retries: int = 2
    sample_rate: int = 16000

    def __post_init__(self):
        if not self.elevenlabs_api_key:
            self.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not self.elevenlabs_voice_id:
            self.elevenlabs_voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")


@dataclass
class StateConfig:
    acquire_timeout: float = 4.0  # Wait for face matcher to confirm identity (5 frames @ 4fps = 1.25s + buffer)
    identify_timeout: float = 15.0
    interaction_timeout: float = 60.0
    disengage_cooldown: float = 3.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    ptz: PTZConfig = field(default_factory=PTZConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    state: StateConfig = field(default_factory=StateConfig)
    db_path: str = "data/faces.db"
    log_level: str = "DEBUG"  # Show detailed matcher logs
