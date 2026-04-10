# Jarvis: Intelligent Robotic Greeter

An intelligent doorway greeter system that recognizes faces, greets people by name, and engages in natural conversation. Built with Python, OpenCV, and cloud APIs for a truly interactive experience.

**Status:** Fully functional prototype running on Mac Mini with support for both PTZ network cameras and local webcams.

---

## Features

✅ **Real-time Face Recognition** — Detects and recognizes individuals using facial embeddings
✅ **Natural Greetings** — Greets known people by name with synthesized speech
✅ **Name Enrollment** — Automatically asks and learns names of new visitors
✅ **Conversational AI** — Holds natural conversations using Google Gemini LLM
✅ **Caption System** — Always-on captions show all interactions with timestamps
✅ **Interactive Chat** — Type responses during conversations (even when muted)
✅ **Camera Tracking** — Smoothly follows subjects with PTZ camera control
✅ **Persistent Database** — Stores learned identities for future recognition
✅ **Multi-Camera Support** — Works with Amcrest PTZ cameras or any local webcam
✅ **Graceful Degradation** — Handles failures with natural fallback responses

---

## Quick Start

### Installation

```bash
git clone https://github.com/ryanwatach/robo-greeter.git
cd robo-greeter

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional, for API keys)
cp .env.example .env
# Edit .env with your API keys (ElevenLabs, Gemini)
```

### Run with Default Camera (Amcrest PTZ)

```bash
python main.py
```

### Run with Laptop Webcam

Edit `config.py` and change the default camera config:

```python
# In AppConfig, change camera field:
camera: CameraConfig = field(default_factory=lambda: CameraConfig(source_type="webcam"))
```

Or use environment variable:
```bash
export CAMERA_SOURCE_TYPE=webcam
python main.py
```

### Controls

While running:
- **Q** — Quit
- **M** — Toggle mute (audio on/off)
- **B** — Build/enroll a new face in the database
- **W/A/S/D** or **Arrow Keys** — Manual PTZ camera control (if using PTZ camera)
- **Type any letter during conversation** — Switch to keyboard input mode for typing responses
- **Enter** — Submit typed response
- **Escape** — Cancel typing

---

## Configuration

Edit `config.py` to customize behavior:

### Camera

```python
CameraConfig:
    source_type: str = "rtsp"  # or "webcam" for local device

    # For RTSP (Amcrest, etc.)
    host: str = "192.168.1.108"
    user: str = "admin"
    password: str = "admin"

    # For webcam
    webcam_device: int = 0  # 0=built-in, 1=USB, etc.

    processing_scale: float = 0.25  # Downscale for speed
    target_fps: float = 4.0  # Processing frame rate
```

### Face Recognition

```python
IdentityConfig:
    distance_threshold: float = 0.6  # Lower = stricter matching
    confirmation_frames: int = 8  # Frames needed to confirm identity
    confirmation_ratio: float = 0.6  # % of frames that must match
```

### Audio & Conversation

```python
AudioConfig:
    elevenlabs_api_key: str = ""  # Get from elevenlabs.io
    gemini_api_key: str = ""  # Get from makersuite.google.com
    whisper_model: str = "base"  # "tiny", "small", "base", "medium"
    listen_timeout: float = 30.0  # Max seconds to listen for input
```

### State Timeouts

```python
StateConfig:
    acquire_timeout: float = 4.0  # Time to confirm identity before greeting
    identify_timeout: float = 15.0  # Max time to wait for identification
    interaction_timeout: float = 60.0  # Max conversation time
```

---

## Architecture

```
robo-greeter/
├── main.py                          # Main loop & UI dashboard
├── config.py                        # Configuration (camera, thresholds, etc.)
├── state_machine.py                 # State machine (IDLE → SCANNING → etc.)
├── video/
│   ├── capture.py                   # Camera capture (RTSP or webcam)
│   └── processor.py                 # Frame preprocessing & scaling
├── tracking/
│   ├── tracker.py                   # Multi-object tracking (MOT)
│   └── subject_selector.py          # Select primary subject to interact with
├── identity/
│   ├── database.py                  # SQLite face database
│   ├── matcher.py                   # Face embedding matching & voting
├── audio/
│   ├── tts.py                       # Text-to-speech (ElevenLabs or pyttsx3)
│   ├── stt.py                       # Speech-to-text (Whisper or keyboard)
│   ├── audio_manager.py             # Audio orchestration
│   └── command_listener.py          # Mute/volume control
├── interaction/
│   ├── greeter.py                   # Greeting logic
│   ├── conversationalist.py         # LLM conversation management
│   └── conversation_manager.py      # Google Gemini API integration
├── camera_control/
│   └── ptz.py                       # PTZ camera control (pan/tilt/zoom)
└── utils/
    └── logger.py                    # Logging setup
```

### Core Flow

1. **Capture** → Video frames from camera
2. **Track** → Detect faces and assign IDs across frames
3. **Identify** → Match faces against database using embeddings
4. **Greet** → Synthesize and play greeting message
5. **Interact** → Capture user response, optionally via LLM
6. **Persist** → Store new identities in database
7. **Follow** → Move PTZ camera to track subject

### Key Design Decisions

- **Multi-frame confirmation**: Never trust a single frame. Require N consistent matches before confirming identity.
- **Separate threads**: Video capture, audio, and PTZ control run independently to prevent one from blocking another.
- **Graceful fallback**: If face recognition fails, ask the person's name. If LLM unavailable, use rule-based responses.
- **Real-time dashboard**: OpenCV overlay shows video, detected faces, tracked IDs, and chat captions.

---

## How This Was Built

This system was architected and implemented with [Claude AI](https://claude.ai) using Claude Code. Below is the master prompt that guided development:

---

### Master Prompt

> You are a senior systems architect and robotics/AI engineer. Your task is to design and implement a production-grade prototype of an intelligent robotic greeter system running on a constrained edge device (Mac Mini) using Python.
>
> This system integrates computer vision, identity recognition, conversational AI, and camera motion control.
>
> You must design for robustness, low latency, and realistic real-world failure modes. Avoid naive implementations.

#### System Overview

The system:
- Observes a doorway using an Amcrest PTZ camera (or local webcam)
- Detects and tracks a single active person
- Recognizes known individuals using facial embeddings
- Greets recognized individuals naturally by name
- Handles unknown individuals by asking their name via voice
- Converts spoken name to text and binds it to their facial encoding
- Stores identity persistently for future recognition
- Encourages a natural "sign-in" interaction
- Moves the camera smoothly to follow the active subject
- Displays real-time captions with timestamps
- Allows interactive typing responses during conversations

#### Core Design Principles

1. **PRIORITIZE CORRECTNESS OVER SPEED** — False positives (wrong identity) are worse than slow recognition.
2. **MULTI-FRAME CONFIRMATION** — Never identify from a single frame. Require consistent matches across N frames.
3. **ACTIVE SUBJECT LOCK** — Only one person tracked and interacted with at a time. Define clear acquisition and release conditions.
4. **ASYNCHRONOUS PIPELINE** — Separate vision loop, recognition pipeline, audio I/O, and camera control.
5. **FAIL GRACEFULLY** — Degrade to "I don't recognize you yet", "Could you repeat that?", "Please step closer".

---

## Troubleshooting

### Camera Won't Connect

- **RTSP**: Verify IP, port, username, password in `config.py`. Try accessing the RTSP URL directly in VLC.
- **Webcam**: Check device index (`webcam_device`). List cameras: `python -c "import cv2; print(cv2.CAP_PROP_SUPPORTED" or use `ffmpeg -f avfoundation -list_devices true -i ""`

### Face Not Being Recognized

- Lower `distance_threshold` in `IdentityConfig` to be more strict, or raise it to be more forgiving
- Increase `confirmation_frames` if system needs longer to be sure
- Check database has the person enrolled: Press **B** to build and enroll

### Audio Not Working

- **TTS**: Ensure `elevenlabs_api_key` is set, or system will fall back to local `pyttsx3`
- **STT**: System prefers Whisper if microphone is detected. Ensure audio input device exists
- **Muted**: Press **M** to toggle audio on

### LLM Returning Generic Responses

- Check `gemini_api_key` is set correctly
- Verify Google Gemini API has quota available
- System falls back to rule-based responses if API fails (this is intentional)

---

## Requirements

- Python 3.9+
- macOS (or Linux with minor adjustments)
- Webcam or IP camera (Amcrest PTZ recommended for full features)
- Microphone for voice input
- Internet connection for API calls (optional—system degrades gracefully offline)

### Dependencies

See `requirements.txt`:
- OpenCV (`cv2`)
- face_recognition (dlib-based embeddings)
- Google Generative AI (`google-genai`)
- Whisper (speech-to-text)
- ElevenLabs API (`elevenlabs`)
- pyttsx3 (local TTS fallback)

---

## Performance

- **Video Processing**: ~4 FPS on Mac Mini (downscaled to 25% resolution)
- **Face Recognition**: ~100ms per frame (dlib CPU-based)
- **Latency**: ~2 second end-to-end (capture → identify → greet)
- **Memory**: ~150–200 MB during operation
- **Database Size**: Tested with 5–100 identities

---

## API Keys Setup

### Google Gemini (Conversation)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy and add to `.env`:
   ```
   GEMINI_API_KEY=your-key-here
   ```

### ElevenLabs (Text-to-Speech)

1. Sign up at [elevenlabs.io](https://elevenlabs.io)
2. Copy API key from account settings
3. Choose a voice ID (default: "21m00Tcm4TlvDq8ikWAM")
4. Add to `.env`:
   ```
   ELEVENLABS_API_KEY=your-key-here
   ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
   ```

---

## Future Improvements

- [ ] Multi-subject support (greet multiple people simultaneously)
- [ ] Emotion detection (adjust greeting tone based on mood)
- [ ] Visitor logging dashboard (who visited and when)
- [ ] Mobile app for remote monitoring
- [ ] Integration with door unlock/entry system
- [ ] Custom wake words for always-listening mode
- [ ] Privacy mode (face blur, no recording)

---

## License

MIT

---

## Contributing

Issues and pull requests welcome. Please test on both RTSP cameras and local webcams before submitting.

---

**Built with Claude AI** — An experiment in AI-assisted robotics engineering.
