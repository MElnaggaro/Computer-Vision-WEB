"""
Registration Routes
===================
HTTP endpoints behind ``/api/v1/registration``.

Workflow
--------
1. ``POST /start``    — open a new pending registration session.
2. ``POST /capture``  — append a single base64 frame (call 5–10 times).
3. ``POST /submit``   — bind a ``Firstname_Lastname`` to the session.
4. ``POST /approve``  — admin codeword verified server-side; the new
                        student is moved into ``data/students_faces/``
                        and the encoding cache is rebuilt.
   ``POST /reject``   — discard the session; emits a rejection event.

All admin codeword validation happens **server-side** via
``RegistrationService._verify_codeword`` (constant-time comparison).
The frontend never validates the codeword itself.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.registration.registration_service import (
    InvalidNameError,
    NotEnoughImagesError,
    RegistrationError,
    SessionNotFoundError,
    UnauthorizedError,
    get_registration_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ───────────────────────────────────────


class StartResponse(BaseModel):
    session_id: str
    image_count: int
    name: Optional[str] = None
    created_at: Optional[str] = None
    temp_dir: Optional[str] = None


class CaptureRequest(BaseModel):
    session_id: str = Field(..., description="Active registration session id")
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG frame")


class CaptureResponse(BaseModel):
    session_id: str
    image_count: int
    min_required: int
    max_allowed: int
    ready_for_submit: bool


class SubmitRequest(BaseModel):
    session_id: str
    name: str = Field(..., description="Firstname_Lastname")


class ApproveRequest(BaseModel):
    session_id: str
    codeword: str = Field(..., description="Admin codeword (server-validated)")


class RejectRequest(BaseModel):
    session_id: str
    delete_files: bool = Field(
        default=True,
        description="If true, remove the pending folder; if false, keep it.",
    )


class GenericResponse(BaseModel):
    student: str
    approved: bool
    encoding_summary: Optional[Dict[str, Any]] = None
    deleted_pending: Optional[bool] = None


class SessionInfo(BaseModel):
    session_id: str
    image_count: int
    name: Optional[str] = None
    created_at: Optional[str] = None
    temp_dir: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/")
async def registration_root() -> Dict[str, str]:
    return {"message": "Registration routes operational"}


@router.post("/start", response_model=StartResponse)
async def start() -> StartResponse:
    """Open a new pending registration session."""
    info = get_registration_service().start()
    return StartResponse(**info)


@router.post("/capture", response_model=CaptureResponse)
async def capture(payload: CaptureRequest) -> CaptureResponse:
    """Append a single base64 face image to the session's pending folder."""
    try:
        result = get_registration_service().capture(
            session_id=payload.session_id, image_base64=payload.image_base64
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CaptureResponse(**result)


@router.post("/submit", response_model=StartResponse)
async def submit(payload: SubmitRequest) -> StartResponse:
    """Bind a ``Firstname_Lastname`` name to the session."""
    try:
        info = get_registration_service().submit(
            session_id=payload.session_id, name=payload.name
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except NotEnoughImagesError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StartResponse(**info)


@router.post("/approve", response_model=GenericResponse)
async def approve(payload: ApproveRequest) -> GenericResponse:
    """Admin-approve the registration after server-side codeword validation."""
    try:
        result = get_registration_service().approve(
            session_id=payload.session_id, codeword=payload.codeword
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except UnauthorizedError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except RegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return GenericResponse(**result)


@router.post("/reject", response_model=GenericResponse)
async def reject(payload: RejectRequest) -> GenericResponse:
    """Discard the pending session and emit a rejection event."""
    try:
        result = get_registration_service().reject(
            session_id=payload.session_id, delete_files=payload.delete_files
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return GenericResponse(**result)


@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions() -> List[SessionInfo]:
    """List all currently active registration sessions."""
    sessions = get_registration_service().list_sessions()
    return [SessionInfo(**s) for s in sessions]
