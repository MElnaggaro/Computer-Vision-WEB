<div align="center">

# Smart Classroom Assistant

**Real-time classroom intelligence powered by computer vision, speech recognition, and NLP.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-NLP-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="Vanilla JavaScript" src="https://img.shields.io/badge/Vanilla_JS-Frontend-F7DF1E?style=for-the-badge&logo=javascript&logoColor=111111">
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MElnaggaro/Computer-Vision-PROJECT/main.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI">
  <img alt="License" src="https://img.shields.io/badge/License-Not%20specified-lightgrey?style=for-the-badge">
</p>

<p>
  <a href="#project-overview">Overview</a> •
  <a href="#key-features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#api-documentation">API</a> •
  <a href="#ai--ml-details">AI/ML</a>
</p>

</div>

---

## Project Overview

Smart Classroom Assistant is a full-stack AI system for running a live classroom dashboard from a local webcam and microphone. It recognizes registered students, marks attendance, analyzes post-attendance emotion, records student questions, classifies those questions by computer-science topic, and persists everything into a unified classroom event log.

The project solves a practical classroom workflow problem: instructors need attendance, participation, and engagement signals without manually interrupting teaching. This repository implements that workflow as a local-first FastAPI service plus a cinematic browser dashboard served from static HTML, CSS, and JavaScript.

The system is intentionally grounded in local files and hardware:

| Capability | What the repository actually implements |
| --- | --- |
| Attendance | Face recognition against `BackEnd/data/students_faces/` with cached dlib embeddings |
| Engagement | FER-based emotion inference with per-track temporal smoothing |
| Questions | Browser Web Speech API plus backend audio upload fallback |
| Topic classification | Sentence-transformer embeddings, calibrated logistic regression, and keyword routing |
| Persistence | Append-only JSON event log at `BackEnd/logs/classroom_log.json` |
| UI | Static dashboard with live camera feed, event feed, question history, summaries, CSV export, and registration flow |

---

## Key Features

| Icon | Feature | Description |
| --- | --- | --- |
| 🎥 | Computer vision attendance | Detects faces from browser-uploaded webcam frames and marks recognized students present once per session. |
| 🧬 | Incremental face encoding cache | Uses `manifest.json` fingerprints and per-student pickle files so unchanged student folders load quickly instead of rebuilding every startup. |
| 🛰️ | Temporal identity stabilization | Tracks faces across frames with IoU matching, majority voting, and identity locks to reduce flicker and avoid repeated expensive encodings. |
| 🧠 | Async emotion overlay | Runs emotion analysis only after attendance is recorded, throttles inference every `N` frames, and commits a final mood after a stable sample window. |
| 🎙️ | Speech-to-question flow | Captures browser speech, falls back to uploaded audio transcription when browser speech fails, and attaches questions to the active student. |
| 📚 | NLP topic classification | Classifies questions into academic CS topics with semantic embeddings, calibrated probabilities, keyword boosts, and uncertainty handling. |
| 🪪 | Guest and registration modes | Supports unknown visitors as guests or as pending students requiring admin approval before entering the recognition dataset. |
| 📡 | Event-sourced dashboard | Replays and polls the JSON event log to populate live attendance, emotion, question, registration, and CSV export views. |
| 🚀 | One-command local launcher | `BackEnd/run.py` starts FastAPI, serves the frontend, writes frontend API config, opens the browser, and tears down child processes cleanly. |
| ✅ | Testable service design | The backend includes unit and integration tests for routes, vision, registration, speech, NLP, event logging, and full pipeline flows. |

---

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Frontend | HTML5, CSS3, Vanilla JavaScript, Three.js, GSAP, ScrollTrigger, Lenis |
| Backend | FastAPI, Uvicorn, Pydantic, pydantic-settings |
| Computer Vision | OpenCV, `face-recognition`, dlib, NumPy |
| Emotion Analysis | FER-compatible emotion model wrapper, TensorFlow/Keras ecosystem dependencies, per-track smoothing |
| Speech | Web Speech API in browser, SpeechRecognition, PyAudio, ffmpeg conversion path for uploaded audio |
| NLP | sentence-transformers, scikit-learn, LogisticRegression, CalibratedClassifierCV, joblib, pandas |
| Storage | JSON event log, pickle face-encoding caches, joblib NLP model artifact, CSV export endpoint |
| Infrastructure | Python static server, FastAPI static mount at `/ui`, GitHub Actions workflow |
| Testing | pytest, pytest-asyncio, FastAPI TestClient, httpx, mocked hardware-heavy dependencies |
| Deployment | Local Python/Uvicorn workflow. No Dockerfile or production deployment manifest is currently committed. |

---

## Architecture

```text
User / Classroom
      |
      | webcam + microphone
      v
Static Frontend Dashboard
FrontEnd/index.html, JS/dashboard.js, Three.js, GSAP
      |
      | REST calls, frame uploads, event polling
      v
FastAPI Backend
BackEnd/app/main.py
      |
      +-- /api/v1/vision
      |       |
      |       +-- OpenCV frame decode/downscale
      |       +-- face_recognition HOG detector
      |       +-- dlib 128-d embeddings
      |       +-- EncodingManager cache
      |       +-- FaceTracker identity lock
      |       +-- AttendanceService
      |
      +-- /api/v1/speech
      |       |
      |       +-- browser audio upload
      |       +-- ffmpeg to 16 kHz mono WAV when needed
      |       +-- Google speech recognition via SpeechRecognition
      |
      +-- /api/v1/nlp and /api/v1/interaction
      |       |
      |       +-- text preprocessing
      |       +-- sentence-transformer embeddings
      |       +-- calibrated logistic regression
      |       +-- keyword probability boosts
      |
      +-- /api/v1/registration
      |       |
      |       +-- pending face captures
      |       +-- server-side admin codeword validation
      |       +-- move into students_faces
      |       +-- rebuild face encodings
      |
      v
Local Persistence
BackEnd/data/students_faces/       registered face images
BackEnd/data/encodings/            per-student face caches
BackEnd/data/nlp/raw/dataset.csv   NLP training dataset
BackEnd/data/nlp/trained/models/   trained NLP pipeline
BackEnd/logs/classroom_log.json    append-only classroom event log
```

### Runtime Data Flow

| Flow | Steps |
| --- | --- |
| Attendance | Browser captures webcam frame -> frontend sends base64 JPEG -> backend detects faces -> matches embeddings -> tracker stabilizes identity -> attendance event is logged. |
| Emotion | Recognized student is already marked -> face crop enters emotion tracker -> predictions are sampled every configured interval -> majority vote produces stable mood -> emotion event is logged. |
| Question | Browser speech transcript or backend speech transcription -> `/interaction/ask-question` resolves student ownership -> NLP pipeline classifies topic -> question event is logged. |
| Registration | Unknown visitor starts registration -> captures 5 to 10 face images -> submits `Firstname_Lastname` -> admin codeword is checked server-side -> images move to `students_faces` -> encodings rebuild. |

---

## Repository Structure

```text
Computer-Vision-PROJECT/
├── BackEnd/
│   ├── app/
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── routes/
│   │   │   ├── events_routes.py
│   │   │   ├── health_routes.py
│   │   │   ├── interaction_routes.py
│   │   │   ├── nlp_routes.py
│   │   │   ├── registration_routes.py
│   │   │   ├── speech_routes.py
│   │   │   └── vision_routes.py
│   │   ├── services/
│   │   │   ├── logging/
│   │   │   ├── nlp/
│   │   │   ├── orchestrator/
│   │   │   ├── registration/
│   │   │   ├── speech/
│   │   │   └── vision/
│   │   └── main.py
│   ├── data/
│   │   ├── encodings/
│   │   ├── nlp/
│   │   ├── pending_students/
│   │   └── students_faces/
│   ├── logs/
│   │   └── classroom_log.json
│   ├── tests/
│   ├── _uvicorn_bootstrap.py
│   ├── requirements.txt
│   └── run.py
├── FrontEnd/
│   ├── CSS/
│   │   ├── dashboard.css
│   │   └── styles.css
│   ├── JS/
│   │   ├── config.js
│   │   ├── dashboard.js
│   │   ├── intro-cinematic.js
│   │   ├── script.js
│   │   └── three-scene.js
│   ├── model/
│   │   ├── 1930s_movie_camera.glb
│   │   └── aiu 3d.glb
│   └── index.html
└── .github/
    └── workflows/
        ├── main.yml
        └── pipeline.py
```

| Path | Purpose |
| --- | --- |
| `BackEnd/app/main.py` | FastAPI application factory, route registration, CORS, frontend mount at `/ui`. |
| `BackEnd/app/core/config.py` | Central settings loaded from environment variables or `.env`. |
| `BackEnd/app/routes/` | Versioned API routers under `/api/v1` plus root aliases for health and logs. |
| `BackEnd/app/services/vision/` | Face detection, recognition, tracking, emotion, attendance, webcam runner, encoding cache. |
| `BackEnd/app/services/nlp/` | Text preprocessing, sentence-transformer feature extraction, training, inference, keyword routing. |
| `BackEnd/app/services/speech/` | Microphone transcription and browser audio upload decoding. |
| `BackEnd/app/services/registration/` | Pending student registration and admin approval workflow. |
| `BackEnd/app/services/logging/` | Thread-safe append-only JSON event logger. |
| `BackEnd/data/students_faces/` | Registered student image folders. Current repository snapshot contains 6 student folders and 69 face images. |
| `BackEnd/data/encodings/` | Face encoding manifest and per-student pickle caches. Current manifest tracks 62 accepted cached encodings. |
| `BackEnd/data/nlp/raw/dataset.csv` | Balanced NLP dataset with 2,400 rows across 8 topics. |
| `FrontEnd/` | Static premium UI with cinematic intro, 3D model layer, and live dashboard. |
| `BackEnd/tests/` | 19 backend test files covering routes, services, and integration flows. |

---

## Installation

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python | Python 3.10+ recommended. The GitHub workflow config uses 3.10, and the local cache shows Python 3.11 compatibility. |
| Webcam and microphone | Required for live dashboard recognition and speech workflows. |
| CMake and C++ build tools | Often required by `dlib` and `face-recognition` if prebuilt wheels are unavailable. |
| ffmpeg | Required for backend transcription of browser `webm`, `ogg`, `mp3`, or `m4a` audio uploads. Native WAV, AIFF, and FLAC can decode without ffmpeg. |
| Node.js | Not required. The frontend uses committed static JavaScript assets. |
| Docker | Not configured in this repository. |

### Windows PowerShell

```powershell
git clone https://github.com/MElnaggaro/Computer-Vision-PROJECT.git
cd Computer-Vision-PROJECT

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r BackEnd\requirements.txt

# Runtime imports used by the current NLP and emotion modules.
pip install sentence-transformers fer
```

If `PyAudio`, `dlib`, or `face-recognition` fails to build, install Microsoft C++ Build Tools and CMake, then retry the pip install.

### macOS / Linux

```bash
git clone https://github.com/MElnaggaro/Computer-Vision-PROJECT.git
cd Computer-Vision-PROJECT

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r BackEnd/requirements.txt

# Runtime imports used by the current NLP and emotion modules.
pip install sentence-transformers fer
```

Useful system packages:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake ffmpeg portaudio19-dev

# macOS with Homebrew
brew install cmake ffmpeg portaudio
```

---

## Configuration

The backend uses `pydantic-settings`. Values can be overridden through environment variables or a `.env` file. When using `BackEnd/run.py`, place `.env` in `BackEnd/` because the launcher starts the backend with `BackEnd` as the working directory.

### Core Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROJECT_NAME` | `Smart Classroom Assistant API` | API title exposed in OpenAPI. |
| `VERSION` | `1.0.0` | API version metadata. |
| `API_V1_STR` | `/api/v1` | Versioned API prefix. |
| `BACKEND_CORS_ORIGINS` | `['*']` | CORS origins accepted by FastAPI. |
| `ADMIN_CODEWORD` | `aiu` | Server-side approval code for pending registrations. Override this for any shared environment. |

### Data Paths

| Variable | Default Path |
| --- | --- |
| `DATA_DIR` | `BackEnd/data` |
| `STUDENTS_FACES_DIR` | `BackEnd/data/students_faces` |
| `PENDING_STUDENTS_DIR` | `BackEnd/data/pending_students` |
| `ENCODINGS_DIR` | `BackEnd/data/encodings` |
| `ATTENDANCE_LOG_FILE` | `BackEnd/logs/classroom_log.json` |
| `NLP_MODEL_PATH` | `BackEnd/data/nlp/trained/models/nlp_pipeline.joblib` |
| `NLP_DATASET_PATH` | `BackEnd/data/nlp/raw/dataset.csv` |

### Vision and Emotion Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `FACE_RECOGNITION_MODEL` | `hog` | CPU-friendly `face_recognition` detector. `cnn` can be used when dlib GPU support is available. |
| `FACE_RECOGNITION_TOLERANCE` | `0.45` | Strict face-distance threshold. Lower means stricter matching. |
| `MIN_FACE_SIZE` | `40` | Minimum face width and height in pixels for training images. |
| `MIN_FACE_SHARPNESS` | `30.0` | Laplacian variance threshold used to skip blurry training images. |
| `NUM_JITTERS` | `3` | Number of dlib encoding jitters for training image embeddings. |
| `TRACK_HISTORY_SIZE` | `10` | Per-track recognition history window. |
| `TRACK_STABILITY_THRESHOLD` | `1` | Votes required for stable identity. |
| `ATTENDANCE_STABLE_FRAMES` | `1` | Stable frames required before marking attendance. |
| `EMOTION_DETECTION_INTERVAL` | `3` | Run emotion inference every N recognition frames. |
| `EMOTION_BUFFER_SIZE` | `10` | Emotion majority-vote smoothing window. |
| `EMOTION_MIN_STABLE_SAMPLES` | `5` | Samples required before final emotion is logged. |

### Frontend Runtime Config

`BackEnd/run.py` rewrites `FrontEnd/JS/config.js` before launching the UI.

| Frontend Setting | Default | Purpose |
| --- | --- | --- |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Backend API base URL. |
| `HEALTH_PATH` | `/health` | Root liveness endpoint. |
| `RECOGNIZE_INTERVAL_MS` | `200` | Minimum delay between self-paced recognition requests. |
| `EVENT_POLL_MS` | `1500` | Poll interval for new event log records. |
| `HEALTH_INTERVAL_MS` | `30000` | Health polling interval while online. |
| `HEALTH_OFFLINE_INTERVAL_MS` | `2000` | Health retry interval while offline. |

### External Services

| Service | Configuration |
| --- | --- |
| Google speech recognition | The code uses `SpeechRecognition.recognize_google(...)`. No API key is stored in the repo, but the machine needs internet access for this speech path. |
| Browser Web Speech API | Availability depends on the user's browser and network. The dashboard falls back to backend audio upload when browser speech reports a network-style failure. |

---

## Running the Project

### One-Command Full Stack

```bash
python BackEnd/run.py
```

The launcher performs five steps:

| Step | Action |
| --- | --- |
| 1 | Starts the FastAPI backend at `http://127.0.0.1:8000`. |
| 2 | Starts a static frontend server at `http://127.0.0.1:5500`. |
| 3 | Polls `GET /health` until the backend reports online. |
| 4 | Opens the default browser. |
| 5 | Supervises both child processes until `Ctrl+C`. |

Optional launcher flags:

```bash
python BackEnd/run.py --backend-port 8001 --frontend-port 5501 --no-browser
```

### Backend Only

```bash
cd BackEnd
python -u _uvicorn_bootstrap.py 127.0.0.1 8000 info
```

Open API docs at:

```text
http://127.0.0.1:8000/docs
```

The backend also mounts the static frontend at:

```text
http://127.0.0.1:8000/ui
```

### Frontend Only

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory FrontEnd
```

Open:

```text
http://127.0.0.1:5500
```

### Rebuild Face Encodings

```bash
curl -X POST http://127.0.0.1:8000/api/v1/vision/rebuild-encodings
```

### Train or Rebuild the NLP Model

```bash
cd BackEnd
python -m app.services.nlp.Question_Classification
```

### Production-Style Backend Process

No production deployment manifest is committed, but the ASGI app can be served directly by Uvicorn:

```bash
cd BackEnd
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Use a single worker unless attendance state, camera state, and active-student session state are moved to shared external storage. The current `VisionSession` is process-local by design.

### Run Tests

```bash
cd BackEnd
python -m pytest tests -q
```

---

## API Documentation

FastAPI serves interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### Health and Events

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Root metadata with project name, version, docs path, and UI path. |
| `GET` | `/health` | Root liveness endpoint used by the dashboard. |
| `GET` | `/api/v1/health` | Versioned liveness endpoint. |
| `GET` | `/api/v1/events` | Return event log records, optionally with `?since=<index>`. |
| `GET` | `/api/v1/logs/events` | Versioned alias for event polling. |
| `GET` | `/api/v1/logs/attendance-csv` | Versioned CSV download endpoint. |
| `GET` | `/events` | Root alias for event log records. |
| `GET` | `/logs/events` | Root alias used by the frontend for event polling. |
| `GET` | `/logs/attendance-csv` | Download attendance rows as CSV. |

### Vision

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/api/v1/vision/` | Vision router health. |
| `POST` | `/api/v1/vision/start-camera` | Open server-side webcam for MJPEG streaming. |
| `POST` | `/api/v1/vision/stop-camera` | Release server-side webcam. |
| `GET` | `/api/v1/vision/stream` | MJPEG stream of annotated server-side frames. |
| `POST` | `/api/v1/vision/recognize-frame` | Process one browser-uploaded base64 frame. |
| `POST` | `/api/v1/vision/reset-attendance` | Clear session attendance, trackers, and active student. |
| `POST` | `/api/v1/vision/rebuild-encodings` | Rebuild face encoding cache from `students_faces`. |
| `GET` | `/api/v1/vision/state` | Return student summaries, marked count, and active student. |
| `POST` | `/api/v1/vision/build-encodings` | Legacy alias for `rebuild-encodings`. |
| `POST` | `/api/v1/vision/start-attendance` | Legacy alias for `recognize-frame`. |
| `POST` | `/api/v1/vision/reset-session` | Legacy alias for `reset-attendance`. |

### Speech

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/api/v1/speech/` | Speech router health. |
| `GET` | `/api/v1/speech/status` | Check whether the server microphone can be opened. |
| `POST` | `/api/v1/speech/transcribe` | Capture one phrase from the server microphone and transcribe it. |
| `POST` | `/api/v1/speech/transcribe-audio` | Transcribe uploaded browser audio using native decode or ffmpeg conversion. |
| `POST` | `/api/v1/speech/debug` | Return audio decode diagnostics and speech request status. |

### NLP and Interaction

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/api/v1/nlp/` | NLP router health. |
| `POST` | `/api/v1/nlp/classify` | Classify a text-only question into a topic. |
| `GET` | `/api/v1/interaction/` | Interaction router health. |
| `POST` | `/api/v1/interaction/guest-session` | Allocate a `Guest_NNN` identity and log guest attendance. |
| `POST` | `/api/v1/interaction/ask-question` | Resolve speaker identity, classify the question, and log the result. |

### Registration

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/api/v1/registration/` | Registration router health. |
| `POST` | `/api/v1/registration/start` | Start a pending registration session. |
| `POST` | `/api/v1/registration/capture` | Save one base64 face image to the pending session. |
| `POST` | `/api/v1/registration/submit` | Attach a `Firstname_Lastname` identity after enough captures. |
| `POST` | `/api/v1/registration/approve` | Validate admin codeword, move images into `students_faces`, rebuild encodings. |
| `POST` | `/api/v1/registration/reject` | Reject a pending session and optionally delete captured files. |
| `GET` | `/api/v1/registration/sessions` | List active in-memory registration sessions. |

### Example Requests

Classify a question:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/nlp/classify \
  -H "Content-Type: application/json" \
  -d '{"question":"What is a semaphore?"}'
```

Ask a question for a known or active student:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/interaction/ask-question \
  -H "Content-Type: application/json" \
  -d '{"student":"Mohammed_Ayman","text":"What is TCP handshake?"}'
```

Start a guest session:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/interaction/guest-session
```

Read new log records after index 10:

```bash
curl http://127.0.0.1:8000/logs/events?since=10
```

<details>
<summary><strong>Core request and response shapes</strong></summary>

| Endpoint | Request Body | Response Highlights |
| --- | --- | --- |
| `POST /api/v1/vision/recognize-frame` | `image_base64`, `mark_attendance` | `faces_detected`, `results`, `active_student` |
| `POST /api/v1/nlp/classify` | `question` | `question`, `topic`, `topic_confidence`, `timestamp` |
| `POST /api/v1/interaction/ask-question` | `student`, `text` | `student`, `question`, `topic`, `registered`, `is_guest`, `resolved_from_active` |
| `POST /api/v1/registration/start` | Empty JSON body or no body | `session_id`, `image_count`, `created_at`, `temp_dir` |
| `POST /api/v1/registration/capture` | `session_id`, `image_base64` | `image_count`, `min_required`, `max_allowed`, `ready_for_submit` |
| `POST /api/v1/registration/approve` | `session_id`, `codeword` | `student`, `approved`, `encoding_summary` |
| `GET /logs/events` | Query `since` | `count`, `total`, `events` |

</details>

---

## AI / ML Details

### Computer Vision Pipeline

| Stage | Implementation |
| --- | --- |
| Input | Browser captures webcam frames, mirrors the video naturally, downsizes uploads to width `480`, encodes JPEG at quality `0.65`, and posts a data URL to `/vision/recognize-frame`. |
| Decode | `decode_base64_frame()` accepts raw base64 or `data:image/...;base64,...` and decodes to an OpenCV BGR frame. |
| Detection | `FaceDetector` wraps `face_recognition.face_locations` with default `hog` model. Each frame is downscaled by `0.5` before detection, then bounding boxes are scaled back to full resolution. |
| Recognition | `FaceRecognizer` computes dlib face encodings, compares against one representative mean vector per student, then optionally verifies against all detailed encodings for the best student. |
| Thresholding | `FACE_RECOGNITION_TOLERANCE=0.45` is used as a strict maximum distance for accepting a known identity. |
| Similarity | Distance is converted into a nonlinear similarity score so strong matches display more intuitive percentages. |
| Tracking | `FaceTracker` associates detections frame-to-frame using IoU and maintains recognition history for stable identity output. |
| Identity lock | Stable tracks lock a recognized identity for up to `30` frames when IoU is at least `0.35`, avoiding repeated embedding computation for the same face. |
| Attendance | `AttendanceService` marks a recognized student once per session and prevents duplicate registered attendance events. |
| Active speaker | `VisionSession` keeps the most recent registered or guest identity active for about `8` seconds so questions can be attributed without resending the name. |

### Face Encoding Dataset

The current repository snapshot contains:

| Dataset Item | Current Value |
| --- | ---: |
| Registered student folders | 6 |
| Face image files in `students_faces` | 69 |
| Accepted cached encodings in `manifest.json` | 62 |
| Encoding cache format | Per-student `.pkl` files plus `manifest.json` |
| Supported image extensions | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |

Encoding quality controls:

| Control | Default | Purpose |
| --- | ---: | --- |
| Minimum face size | `40px` | Skip faces too small to encode reliably. |
| Minimum sharpness | `30.0` | Skip blurry images using Laplacian variance. |
| Number of jitters | `3` | Improve encoding robustness when building the cache. |
| Single-face validation | Required | Images with zero or multiple faces are skipped. |

### Emotion Analysis

The emotion pipeline is deliberately decoupled from first attendance marking. Attendance is recorded as soon as identity is stable, while emotion is updated asynchronously afterward.

| Component | Detail |
| --- | --- |
| Detector | `EmotionDetector` imports `FER` from `fer.fer` or `fer` and initializes `FER(mtcnn=False)`. |
| Input | Cropped BGR face region from the recognized tracked face. |
| Primary path | Runs FER detection on a padded crop to help Haar-based internal detection. |
| Fallback path | Converts the crop to grayscale, resizes to `48x48`, and feeds the emotion classifier directly when Haar detection misses. |
| Label mapping | Raw labels are converted to classroom-friendly labels such as `Anxious`, `Surprised`, and `Uncomfortable`. |
| Smoothing | `EmotionTracker` stores per-track rolling buffers and uses majority vote plus averaged confidence. |
| Finalization | A stable emotion is logged after `EMOTION_MIN_STABLE_SAMPLES=5`. |

### Speech Pipeline

| Mode | Implementation |
| --- | --- |
| Browser-first | The dashboard uses `window.SpeechRecognition` or `webkitSpeechRecognition` with `en-US`. |
| Upload fallback | If browser speech fails with a network-style error, `MediaRecorder` uploads audio to `/speech/transcribe-audio`. |
| Native decode | WAV, FLAC, and AIFF are decoded directly with `speech_recognition.AudioFile`. |
| ffmpeg decode | WebM, OGG, MP3, M4A, and unknown formats are transcoded to 16-bit PCM WAV at 16 kHz mono. |
| Transcription | Backend transcription uses `SpeechRecognition` and Google's speech recognition request path. |
| Error handling | Routes return distinct status codes for timeout, unclear audio, speech API failures, microphone errors, and decode failures. |

### NLP Question Classification

The NLP pipeline is implemented in `BackEnd/app/services/nlp/Question_Classification.py`.

| Stage | Detail |
| --- | --- |
| Dataset | `BackEnd/data/nlp/raw/dataset.csv` with columns `topic` and `question`. |
| Size | 2,400 questions. |
| Topic balance | 8 classes with 300 examples each. |
| Preprocessing | Lowercasing, contraction expansion, punctuation removal, stopword removal, lemmatization fallback, speech correction rules. |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers`. |
| Classifier | `LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")`. |
| Calibration | `CalibratedClassifierCV(method="sigmoid", cv=5)`. |
| Split | `train_test_split(test_size=0.2, random_state=42, stratify=y)`. |
| Inference enhancement | Keyword boosts add topic-specific probability mass, then probabilities are renormalized. |
| Uncertainty | Predictions below `0.30` top probability return `Uncertain`. |
| Artifact | `BackEnd/data/nlp/trained/models/nlp_pipeline.joblib`. |

Current dataset topics:

| Topic | Rows |
| --- | ---: |
| Computer Networks | 300 |
| Computer Organization and Architecture | 300 |
| Digital Logic | 300 |
| General Aptitude | 300 |
| Mathematics | 300 |
| Operating System | 300 |
| Programming and Data Structure | 300 |
| Theory of Computation | 300 |

---

## Performance / Runtime Characteristics

This repository contains runtime safeguards and instrumentation, but no committed benchmark report. The defensible metrics below come directly from configuration and source code.

| Area | Implemented Behavior |
| --- | --- |
| Recognition cadence | Frontend self-paces recognition requests with a `200ms` minimum floor, so it avoids overlapping requests and caps the fast path near 5 requests per second. |
| Event refresh | Event log polling runs every `1500ms`, with immediate one-shot polling when new attendance is marked. |
| Health polling | Online backend heartbeat is `30000ms`; offline retry is `2000ms`. |
| Face detection cost control | Frames are downscaled to `0.5x` for detection before boxes are mapped back to full resolution. |
| Recognition cost control | Stable identity locks skip repeated `face_encodings()` work for up to `30` frames. |
| Emotion cost control | Emotion is skipped until attendance is already recorded and then runs every `3` recognition frames. |
| Emotion stability | Final mood is not committed until `5` emotion samples are collected. |
| Duplicate attendance | Registered students are tracked in an in-memory set and logged once per session. |
| Persistence | Events are appended immediately to JSON, so successful writes are durable without a background flush. |

For real benchmarking, capture latency around `POST /api/v1/vision/recognize-frame`, browser FPS, CPU usage, and memory usage on the target machine and camera resolution.

---

## Screenshots / Demo

No screenshot or GIF assets are currently committed. The application does include a visual demo-ready UI in `FrontEnd/index.html`.

| Demo Area | What to Capture |
| --- | --- |
| Hero | Cinematic intro, particle field, AIU 3D model, gradient typography. |
| Live dashboard | Camera panel, identity overlay, event feed, question history, student summaries. |
| Registration | Unknown visitor actions, capture modal, inline admin approval panel. |
| CSV export | Attendance log export from `/logs/attendance-csv`. |

Suggested future asset locations:

```text
docs/assets/hero.png
docs/assets/dashboard.png
docs/assets/registration-flow.png
docs/assets/demo.gif
```

---

## Challenges & Engineering Decisions

| Challenge | Engineering Decision |
| --- | --- |
| Avoiding slow startup from face encoding rebuilds | Added a manifest-based cache that fingerprints each student folder and rebuilds only changed, added, or deleted students. |
| Keeping live recognition responsive | The frontend uses self-paced polling instead of `setInterval`, and the backend uses identity locks to avoid repeated face encoding for stable tracks. |
| Preventing duplicate attendance | Registered names are held in a session-level set, so repeated frames for the same student do not create duplicate attendance rows. |
| Reducing first-recognition latency | Attendance is not gated on emotion. Emotion analysis starts only after attendance is logged. |
| Handling unknown visitors | The UI exposes both `Continue as Guest` and `Register Stranger`; the backend assigns guests deterministic `Guest_NNN` identities. |
| Securing student registration flow | The frontend gates the admin panel for UX, but the backend performs the real codeword validation with constant-time comparison. |
| Making questions attributable | The backend resolves question ownership from explicit student, active vision identity, guest identity, or `Unknown` fallback. |
| Supporting browser audio formats | Uploaded audio uses native decode when possible and ffmpeg conversion for WebM, OGG, MP3, M4A, and other formats. |
| Preserving event history for the UI | Attendance, emotion, question, and registration events share one chronological JSON log that the dashboard can replay and poll. |
| Improving Windows reliability | The launcher uses a Windows-friendly Uvicorn bootstrap and job-object cleanup so child processes do not remain orphaned. |

---

## Testing

The backend includes tests for the API surface, attendance, face recognition, emotion detection, registration, speech, NLP, guest sessions, identity resolution, and full integration flows.

```bash
cd BackEnd
python -m pytest tests -q
```

Targeted examples:

```bash
cd BackEnd
python -m pytest tests/test_api_routes.py -q
python -m pytest tests/test_registration.py -q
python -m pytest tests/test_vision.py -q
python tests/test_nlp_accuracy.py
```

Hardware-heavy paths are mocked in the automated tests so CI and local development do not require a real webcam, real microphone, or live dlib/FER inference for every test.

---

## Future Improvements

| Priority | Improvement |
| --- | --- |
| High | Add a production-grade `Dockerfile` and `docker-compose.yml` for reproducible deployment. |
| High | Replace the demo admin codeword flow with authenticated users, roles, and password hashing. |
| High | Move classroom events from JSON to SQLite or PostgreSQL while keeping JSON export. |
| Medium | Add WebSocket or Server-Sent Events so the dashboard receives logs without polling. |
| Medium | Add benchmark scripts for recognition latency, throughput, CPU usage, and emotion inference cost. |
| Medium | Add a committed model card for face-recognition constraints, emotion limitations, and NLP evaluation results. |
| Medium | Add screenshot and GIF assets under `docs/assets/` for recruiter-friendly previews. |
| Low | Generate a typed API client from OpenAPI for frontend calls. |
| Low | Add privacy controls for face image retention, log retention, and student data export. |

---

## Contribution Guide

Contributions should preserve the local-first architecture and avoid introducing ungrounded claims or mock-only features into production paths.

```bash
git checkout -b feature/your-change
cd BackEnd
python -m pytest tests -q
```

Recommended contribution standards:

| Area | Expectation |
| --- | --- |
| Backend changes | Add or update pytest coverage under `BackEnd/tests/`. |
| API changes | Keep FastAPI response models accurate and update this README endpoint table. |
| Vision changes | Avoid opening webcams or loading heavy models at import time. |
| Frontend changes | Keep `FrontEnd/JS/config.js` compatible with `BackEnd/run.py`. |
| Data changes | Do not commit private student images without consent. |

---

## Data & Privacy Notes

This project processes face images, webcam frames, microphone audio, student names, attendance status, emotion labels, and question text. Treat the repository data folders and generated logs as sensitive.

| Data | Location | Note |
| --- | --- | --- |
| Registered face images | `BackEnd/data/students_faces/` | Used to build recognition encodings. |
| Pending registration images | `BackEnd/data/pending_students/` | Temporary captures until approval or rejection. |
| Face encodings | `BackEnd/data/encodings/` | Pickled embedding caches derived from face images. |
| Classroom events | `BackEnd/logs/classroom_log.json` | Attendance, emotion, question, guest, and registration events. |
| NLP model and dataset | `BackEnd/data/nlp/` | Academic question classifier data and artifact. |

For real classroom use, add consent, retention, access control, and audit policies before deployment.

---

## License

No license file is currently present in this repository. Without an explicit license, reuse rights are not granted by default. Add a `LICENSE` file if this project is intended to be open source.

---

<div align="center">

**Built as a local-first AI classroom system: vision for attendance, speech for interaction, NLP for insight, and a live dashboard for instructors.**

</div>
