"""
Application configuration.

All paths are resolved relative to the BackEnd root directory so
the project can be checked out anywhere without breaking.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Resolve BackEnd root (two levels up from this file) ──────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Central, immutable application settings."""

    # ── Project metadata ─────────────────────────────────────────────
    PROJECT_NAME: str = "Smart Classroom Assistant API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # ── CORS ─────────────────────────────────────────────────────────
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # ── Directory paths (all relative to BackEnd/) ───────────────────
    BASE_DIR: Path = _BACKEND_ROOT
    DATA_DIR: Path = _BACKEND_ROOT / "data"
    STUDENTS_FACES_DIR: Path = _BACKEND_ROOT / "data" / "students_faces"
    PENDING_STUDENTS_DIR: Path = _BACKEND_ROOT / "data" / "pending_students"
    ENCODINGS_DIR: Path = _BACKEND_ROOT / "data" / "encodings"
    ENCODINGS_FILE: Path = _BACKEND_ROOT / "data" / "encodings" / "face_encodings.pkl"
    LOGS_DIR: Path = _BACKEND_ROOT / "logs"
    ATTENDANCE_LOG_FILE: Path = _BACKEND_ROOT / "logs" / "classroom_log.json"

    # ── NLP Paths ────────────────────────────────────────────────────
    NLP_MODEL_DIR: Path = DATA_DIR / "nlp" / "trained" / "models"
    NLP_MODEL_PATH: Path = NLP_MODEL_DIR / "nlp_pipeline.joblib"
    NLP_DATASET_PATH: Path = DATA_DIR / "nlp" / "raw" / "dataset.csv"

    # ── Vision / face‑recognition settings ───────────────────────────
    FACE_RECOGNITION_MODEL: str = "hog"          # "hog" (CPU) or "cnn" (GPU)
    FACE_RECOGNITION_TOLERANCE: float = 0.45     # strict threshold for matching
    SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    )

    # ── Image quality validation ─────────────────────────────────────
    MIN_FACE_SIZE: int = 40                      # minimum face width/height (px)
    MIN_FACE_SHARPNESS: float = 30.0             # Laplacian variance threshold
    NUM_JITTERS: int = 3                         # jitters for encoding accuracy

    # ── Temporal stabilization ───────────────────────────────────────
    TRACK_HISTORY_SIZE: int = 10                 # frames of recognition history
    TRACK_STABILITY_THRESHOLD: int = 6           # votes needed for stable ID
    TRACK_MAX_MISSED_FRAMES: int = 10            # frames before dropping a track
    TRACK_IOU_THRESHOLD: float = 0.25            # IoU for cross‑frame matching
    ATTENDANCE_STABLE_FRAMES: int = 10           # stable frames before marking

    # ── Emotion detection ────────────────────────────────────────────
    EMOTION_DETECTION_INTERVAL: int = 5          # run detector every N recognition frames
    EMOTION_BUFFER_SIZE: int = 10                # smoothing window (majority-vote frames)
    EMOTION_MAX_STALE_FRAMES: int = 30           # drop buffer after N missed frames

    # ── Registration / Admin approval ───────────────────────────────
    # WARNING: For production deployments, override ADMIN_CODEWORD via
    # the ``.env`` file or an environment variable rather than committing
    # a real secret to source control.
    ADMIN_CODEWORD: str = "aiu"
    REGISTRATION_MIN_IMAGES: int = 5
    REGISTRATION_MAX_IMAGES: int = 10
    REGISTRATION_NAME_PATTERN: str = r"^[A-Za-z]+_[A-Za-z]+$"

    # ── Emotion stability gate ───────────────────────────────────────
    # Attendance is logged only after the emotion tracker has collected
    # at least this many samples for the face — preventing the "first
    # frame's emotion" from being committed before any averaging.
    EMOTION_MIN_STABLE_SAMPLES: int = 5

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
    )


settings = Settings()
