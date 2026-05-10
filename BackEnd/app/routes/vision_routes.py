"""
Vision API Routes
=================
Exposes the face-recognition + emotion + attendance pipeline as RESTful
endpoints behind ``/api/v1/vision``.

Endpoints
---------
* ``GET  /``                  — sub-router health
* ``POST /start-camera``      — open the server-side webcam (MJPEG mode)
* ``POST /stop-camera``       — release the server-side webcam
* ``GET  /stream``            — MJPEG stream of annotated frames
* ``POST /recognize-frame``   — process a single base64 frame (browser webcam mode)
* ``POST /reset-attendance``  — clear in-memory attendance + trackers
* ``POST /rebuild-encodings`` — rescan ``students_faces`` and rebuild cache
* ``GET  /state``             — student summaries for the frontend dashboard
* ``POST /build-encodings``   — alias for ``rebuild-encodings`` (legacy)
* ``POST /start-attendance``  — alias for ``recognize-frame`` (legacy)
* ``POST /reset-session``     — alias for ``reset-attendance`` (legacy)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.vision.vision_session import (
    VisionError,
    decode_base64_frame,
    get_vision_session,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ───────────────────────────────────────


class FramePayload(BaseModel):
    """Base64-encoded BGR frame sent by the frontend / test client."""

    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG image")
    mark_attendance: bool = Field(
        default=True,
        description="Whether to mark attendance for stable + recognised faces.",
    )


class StartCameraPayload(BaseModel):
    camera_index: int = Field(default=0, ge=0)


class CameraStatusResponse(BaseModel):
    running: bool
    message: str


class RecognizeFrameResponse(BaseModel):
    faces_detected: int
    results: List[Dict[str, Any]]
    active_student: str | None = None


class BuildEncodingsResponse(BaseModel):
    message: str
    summary: Dict[str, Any]


class AttendanceResetResponse(BaseModel):
    message: str


class StateResponse(BaseModel):
    students: List[Dict[str, Any]]
    marked_count: int
    active_student: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/")
async def vision_root() -> Dict[str, str]:
    """Sub-router health-check."""
    return {"message": "Vision routes operational"}


@router.post("/start-camera", response_model=CameraStatusResponse)
async def start_camera(payload: StartCameraPayload | None = None) -> CameraStatusResponse:
    """Open the server-side webcam to enable the MJPEG ``/stream`` endpoint."""
    session = get_vision_session()
    index = payload.camera_index if payload else 0
    ok = session.start_camera(camera_index=index)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail=f"Could not open server-side camera at index {index}",
        )
    return CameraStatusResponse(running=True, message="Camera started")


@router.post("/stop-camera", response_model=CameraStatusResponse)
async def stop_camera() -> CameraStatusResponse:
    """Release the server-side webcam (no-op if not running)."""
    session = get_vision_session()
    session.stop_camera()
    return CameraStatusResponse(running=False, message="Camera stopped")


@router.get("/stream")
async def stream() -> StreamingResponse:
    """MJPEG stream of annotated camera frames."""
    session = get_vision_session()
    if not session.is_camera_running():
        raise HTTPException(
            status_code=409,
            detail="Server-side camera is not running. POST /start-camera first.",
        )
    return StreamingResponse(
        session.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/recognize-frame", response_model=RecognizeFrameResponse)
async def recognize_frame(payload: FramePayload) -> RecognizeFrameResponse:
    """Process a single base64 frame uploaded from the browser webcam."""
    try:
        frame = decode_base64_frame(payload.image_base64)
    except VisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    session = get_vision_session()
    try:
        results = session.recognize_frame(
            frame, mark_attendance=payload.mark_attendance
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("recognize_frame failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return RecognizeFrameResponse(
        faces_detected=len(results),
        results=results,
        active_student=session.get_active_student(),
    )


@router.post("/reset-attendance", response_model=AttendanceResetResponse)
async def reset_attendance() -> AttendanceResetResponse:
    """Reset in-memory attendance + tracker state for a fresh class period."""
    session = get_vision_session()
    session.reset_attendance()
    return AttendanceResetResponse(message="Attendance session reset successfully")


@router.post("/rebuild-encodings", response_model=BuildEncodingsResponse)
async def rebuild_encodings() -> BuildEncodingsResponse:
    """Rescan ``data/students_faces/`` and rebuild the encoding cache."""
    session = get_vision_session()
    try:
        summary = session.rebuild_encodings()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("rebuild_encodings failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return BuildEncodingsResponse(
        message="Encodings rebuilt successfully", summary=summary
    )


@router.get("/state", response_model=StateResponse)
async def state() -> StateResponse:
    """Return current attendance + per-student summary for the dashboard."""
    session = get_vision_session()
    summaries = session.get_summaries()
    return StateResponse(
        students=summaries,
        marked_count=len(session.attendance_service.marked_students),
        active_student=session.get_active_student(),
    )


# ── Legacy aliases (kept for backwards-compatibility / older tests) ─


@router.post("/build-encodings", response_model=BuildEncodingsResponse)
async def build_encodings() -> BuildEncodingsResponse:
    """Legacy alias for :func:`rebuild_encodings`."""
    return await rebuild_encodings()


@router.post("/start-attendance", response_model=RecognizeFrameResponse)
async def start_attendance(payload: FramePayload) -> RecognizeFrameResponse:
    """Legacy alias for :func:`recognize_frame`."""
    return await recognize_frame(payload)


@router.post("/reset-session", response_model=AttendanceResetResponse)
async def reset_session() -> AttendanceResetResponse:
    """Legacy alias for :func:`reset_attendance`."""
    return await reset_attendance()
