Master Prompt:
You are a senior systems architect and robotics/AI engineer. Your task is to design and implement a production-grade prototype of an intelligent robotic greeter system running on a constrained edge device (Mac Mini) using Python.

This system integrates computer vision, identity recognition, conversational AI, and camera motion control.

You must design for robustness, low latency, and realistic real-world failure modes. Avoid naive implementations.

---

## SYSTEM OVERVIEW

The system:

* Observes a doorway using an Amcrest PTZ camera
* Detects and tracks a single active person
* Recognizes known individuals using facial embeddings
* Greets recognized individuals naturally by name
* Handles unknown individuals by asking their name via voice
* Converts spoken name to text and binds it to their facial encoding
* Stores identity persistently for future recognition
* Encourages a natural “sign-in” interaction
* Moves the camera smoothly to follow the active subject

Constraints:

* Max 10 people in room, but only 1 active interaction at a time
* Database size <100 identities
* Real-time preferred but not at cost of stability
* Edge compute limited (Mac Mini, no heavy GPU assumptions)

---

## CORE DESIGN PRINCIPLES

1. PRIORITIZE CORRECTNESS OVER SPEED
   False positives (wrong identity) are worse than slow recognition.

2. MULTI-FRAME CONFIRMATION
   Never identify a person from a single frame.
   Require consistent matches across N frames.

3. ACTIVE SUBJECT LOCK
   Only one person is tracked and interacted with at a time.
   Define clear acquisition and release conditions.

4. ASYNCHRONOUS PIPELINE
   Separate:

* Vision loop
* Recognition pipeline
* Audio I/O
* Camera control

5. FAIL GRACEFULLY
   System must degrade to:

* “I don’t recognize you yet”
* “Could you repeat that?”
* “Please step closer”

---

## SYSTEM COMPONENTS

1. VIDEO INGESTION

* Stream from Amcrest camera
* Downscale frames for processing
* Process at ~3–5 FPS max

2. FACE DETECTION + ENCODING
   Use face_recognition library:

* Detect faces
* Generate embeddings
* Maintain temporal buffer per face

3. FACE TRACKING

* Assign IDs to detected faces across frames
* Select primary subject based on:

  * proximity to center
  * largest bounding box
  * temporal persistence

4. IDENTITY MATCHING

* Compare embeddings to stored database
* Use cosine distance threshold
* Require consistent match across multiple frames

5. UNKNOWN USER FLOW

* Trigger voice prompt via ElevenLabs:
  “Hey, I don’t think we’ve met yet. What’s your name?”

* Capture audio

* Transcribe to text (use Whisper or equivalent)

* Clean and normalize name

* Confirm:
  “Did you say [name]?”

* Store:

  * Name
  * Averaged embedding across multiple frames

6. KNOWN USER FLOW

* Greet naturally:
  “Hey [name], good to see you.”
* Prompt for sign-in:
  Avoid robotic phrasing

7. CAMERA CONTROL
   Using python-amcrest:

* Track subject position relative to frame center
* Move camera slowly to re-center
* Implement dead zone to prevent jitter
* Limit update frequency

8. DATA STORAGE

* Local database (SQLite or JSON)

* Store:

  * name
  * embedding vector
  * timestamp metadata

* Support updating embeddings over time

9. AUDIO SYSTEM

* TTS: ElevenLabs
* STT: Whisper or similar
* Must support:

  * interruption handling
  * timeout fallback

---

## STATE MACHINE

Define explicit states:

IDLE:

* No active subject

ACQUIRE_TARGET:

* Detect candidate face
* Stabilize tracking

IDENTIFYING:

* Attempt recognition across multiple frames

KNOWN_INTERACTION:

* Greet and sign-in

UNKNOWN_INTERACTION:

* Ask name, confirm, store identity

TRACKING:

* Maintain camera lock

DISENGAGE:

* Subject leaves frame or timeout

---

## FAILURE MODES TO HANDLE

* Multiple faces detected
* Face partially visible
* Lighting changes
* Motion blur
* Misrecognition
* Audio transcription errors
* Network lag (camera or API)

---

## PERFORMANCE OPTIMIZATIONS

* Frame skipping
* Embedding caching
* Batch comparisons
* Limit camera movement frequency
* Use threading or asyncio

---

## DELIVERABLES

1. Modular Python architecture
2. Clear separation of components
3. Configurable thresholds
4. Logging system for debugging
5. Simple CLI or dashboard for monitoring

---

## GOAL

Build a system that feels natural, not robotic, and avoids obvious failure cases.

Do not optimize prematurely for scale. Optimize for correctness, stability, and user experience.
