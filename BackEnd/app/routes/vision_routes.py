"""
Vision API Routes
=================
Exposes the face‑recognition and attendance pipeline as RESTful
endpoints behind ``/api/v1/vision``.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.vision.attendance_service import AttendanceService
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognizer import FaceRecognizer

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Singletons (shared across requests within one process) ───────────
_encoding_manager = EncodingManager()
_face_detector = FaceDetector()
_face_recognizer = FaceRecognizer(encoding_manager=_encoding_manager)
_attendance_service = AttendanceService()


# ── Request / Response schemas ───────────────────────────────────────

class FramePayload(BaseModel):
    """Base64‑encoded BGR frame sent by the frontend / test client."""
    image_base64: str = Field(
        ..., description="Base64‑encoded JPEG or PNG image"
    )


class BuildEncodingsResponse(BaseModel):
    message: str
    summary: Dict[str, Any]


class RecognizeFrameResponse(BaseModel):
    faces_detected: int
    results: list


class AttendanceStatusResponse(BaseModel):
    message: str
    new_records: list
    total_marked: int


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/")
async def vision_root() -> Dict[str, str]:
    """Health‑check for the vision sub‑router."""
    return {"message": "Vision routes operational"}


@router.post("/build-encodings", response_model=BuildEncodingsResponse)
async def build_encodings() -> BuildEncodingsResponse:
    """Scan ``data/students_faces/`` and (re)build the encoding cache.

    Should be called once on setup, or whenever new student images are added.
    """
    try:
        summary = _encoding_manager.build_encodings()
        return BuildEncodingsResponse(
            message="Encodings built successfully",
            summary=summary,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Error building encodings")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/recognize-frame", response_model=RecognizeFrameResponse)
async def recognize_frame(payload: FramePayload) -> RecognizeFrameResponse:
    """Decode a base64 frame, detect faces, and recognise each one.

    Returns structured results (name / registered / similarity / location).
    """
    frame = _decode_frame(payload.image_base64)

    # Ensure encodings are in memory
    if not _encoding_manager.is_loaded:
        _encoding_manager.load_encodings()

    locations = _face_detector.detect_faces(frame)
    results = _face_recognizer.recognize_faces(frame, locations) if locations else []

    return RecognizeFrameResponse(
        faces_detected=len(locations),
        results=results,
    )


@router.post("/start-attendance", response_model=AttendanceStatusResponse)
async def start_attendance(payload: FramePayload) -> AttendanceStatusResponse:
    """Full pipeline: detect → recognise → mark attendance → persist.

    Idempotent for known students (duplicates are silently ignored).
    """
    frame = _decode_frame(payload.image_base64)

    if not _encoding_manager.is_loaded:
        _encoding_manager.load_encodings()

    locations = _face_detector.detect_faces(frame)
    results = _face_recognizer.recognize_faces(frame, locations) if locations else []

    new_records = []
    for result in results:
        record = _attendance_service.mark_attendance(
            name=result["name"],
            registered=result["registered"],
            similarity=result["similarity"],
        )
        if record is not None:
            new_records.append(record)

    # Auto‑persist after every attendance frame
    if new_records:
        _attendance_service.save_log()

    return AttendanceStatusResponse(
        message="Attendance processed",
        new_records=new_records,
        total_marked=len(_attendance_service.marked_students),
    )


@router.post("/reset-session")
async def reset_session() -> Dict[str, str]:
    """Clear the in‑memory attendance session (for a new class period)."""
    _attendance_service.reset_session()
    return {"message": "Attendance session reset successfully"}


# ── Helpers ──────────────────────────────────────────────────────────

def _decode_frame(image_base64: str) -> np.ndarray:
    """Decode a base64‑encoded image into a BGR ``np.ndarray``."""
    try:
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as exc:
        logger.error("Failed to decode frame: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")
