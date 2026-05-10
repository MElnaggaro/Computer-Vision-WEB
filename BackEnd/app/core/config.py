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
    LOGS_DIR: Path = _BACKEND_ROOT / "app" / "logs"
    ATTENDANCE_LOG_FILE: Path = _BACKEND_ROOT / "app" / "logs" / "classroom_log.json"

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

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
    )


settings = Settings()
