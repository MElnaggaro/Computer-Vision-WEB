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
    FACE_RECOGNITION_TOLERANCE: float = 0.6       # lower = stricter
    CONFIDENCE_THRESHOLD: float = 0.6             # minimum confidence to accept
    SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    )

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
    )


settings = Settings()
