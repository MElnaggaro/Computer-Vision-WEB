"""
Interaction Routes
==================
Implements the speech + NLP integration as a single coherent endpoint:
``POST /api/v1/interaction/ask-question``.

The endpoint can run in two modes:

1. **Microphone mode** (default) — records from the server's microphone,
   transcribes via Google Speech, classifies the topic via the NLP
   pipeline, then logs a ``question`` event tied to the supplied student.

2. **Text mode** — when a ``text`` field is provided, skips speech
   recognition and just classifies the supplied text.  Useful for the
   browser flow where speech-to-text happens client-side, or for tests.

Identity ownership
------------------
A question is always attached to a single student.  The route resolves
ownership in this order:

1. The ``student`` field of the request, if it names a known student
   or a guest (``Guest_NNN``).
2. The vision session's active student — i.e. the person whose face
   was last seen on camera within
   :attr:`VisionSession.ACTIVE_STUDENT_TTL_SECONDS`.
3. Falls back to ``"Unknown"`` (logged with ``registered=false``).

In addition, ``POST /interaction/guest-session`` allocates a new
``Guest_NNN`` identity and pins it as the active student so subsequent
mic presses are attributed to that guest.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.orchestrator.question_pipeline import QuestionPipeline
from app.services.speech.speech_to_text import SpeechError
from app.services.vision.vision_session import get_vision_session

logger = logging.getLogger(__name__)
router = APIRouter()

# Reuse a single pipeline instance — it lazy-loads the NLP model.
# We pass ``log_events=False`` because the route logs through the
# attendance service so the student attribution is correct.
_pipeline = QuestionPipeline(log_events=False)


# ── Schemas ──────────────────────────────────────────────────────────


class AskQuestionRequest(BaseModel):
    """Request body for ``/interaction/ask-question``.

    Fields:
        student: Student to attribute the question to.  Pass an empty
                 string or ``"Unknown"`` to let the backend resolve it
                 from the active vision-session identity.
        text:    Optional pre-transcribed text.  When provided the
                 microphone is *not* opened.
    """

    student: Optional[str] = Field(
        default=None,
        description="Student to attribute question to. Empty/Unknown → use active student.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Pre-transcribed question text. Skips microphone capture.",
    )


class AskQuestionResponse(BaseModel):
    student: str
    question: str
    topic: str
    topic_confidence: float
    registered: bool
    is_guest: bool
    timestamp: str
    resolved_from_active: bool


class GuestSessionResponse(BaseModel):
    """Response for ``POST /interaction/guest-session``."""

    student: str
    registered: bool
    is_guest: bool
    timestamp: str
    message: str


# ── Helpers ─────────────────────────────────────────────────────────


def _resolve_student(requested: Optional[str]) -> Dict[str, Any]:
    """Resolve who a question belongs to.

    Order of precedence:
        1. Explicit non-Unknown name from the request.  Registered /
           guest metadata is filled in from the attendance service so
           the response and log event stay consistent.
        2. The vision session's active student (last seen within TTL).
        3. ``"Unknown"`` fallback.

    Returns a dict with ``student``, ``registered``, ``is_guest``, and
    ``resolved_from_active`` flags.
    """
    session = get_vision_session()
    attendance = session.attendance_service
    name = (requested or "").strip()

    # Case 1 — explicit student id from the caller.  Trust it (the
    # frontend either passes a recognised face name or the freshly
    # allocated Guest_NNN).  We classify it against the attendance
    # service so registered/is_guest flags reflect the real state.
    if name and name.lower() != "unknown":
        is_guest = attendance.is_guest(name) or name.startswith("Guest_")
        if is_guest:
            # Keep the active-student window fresh for follow-up
            # mic presses that omit the student field.
            session.set_active_student(name, registered=False)
        return {
            "student": name,
            "registered": name in attendance.marked_students,
            "is_guest": is_guest,
            "resolved_from_active": False,
        }

    # Case 2 — pull from the live recognition session.
    info = session.get_active_student_info()
    if info["name"]:
        active_name = info["name"]
        return {
            "student": active_name,
            "registered": bool(info["registered"]),
            "is_guest": attendance.is_guest(active_name)
                or active_name.startswith("Guest_"),
            "resolved_from_active": True,
        }

    # Case 3 — Unknown fallback (still logged so the dashboard sees it).
    return {
        "student": "Unknown",
        "registered": False,
        "is_guest": False,
        "resolved_from_active": False,
    }


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/")
async def interaction_root() -> Dict[str, str]:
    """Sub-router health-check."""
    return {"message": "Interaction routes operational"}


@router.post("/guest-session", response_model=GuestSessionResponse)
async def create_guest_session() -> GuestSessionResponse:
    """Allocate a new ``Guest_NNN`` identity, log it, and pin it as active.

    Called by the dashboard when the user clicks "Continue as Guest".
    Subsequent mic presses without an explicit ``student`` field will
    be attributed to the freshly-created guest.
    """
    session = get_vision_session()
    record = session.register_guest()
    return GuestSessionResponse(
        student=record["student"],
        registered=False,
        is_guest=True,
        timestamp=record["timestamp"],
        message=f"Guest session created: {record['student']}",
    )


@router.post("/ask-question", response_model=AskQuestionResponse)
async def ask_question(payload: AskQuestionRequest) -> AskQuestionResponse:
    """Capture (or accept) a question and classify its topic for a given student."""
    resolution = _resolve_student(payload.student)
    student = resolution["student"]

    # 1. Resolve the question text (mic mode or text mode).
    if payload.text and payload.text.strip():
        question_text = payload.text.strip()
        result = _pipeline.process_text_question(question_text)
    else:
        try:
            voice_result = _pipeline.process_voice_question()
        except SpeechError as exc:
            import app.services.speech.speech_to_text as stt
            if isinstance(exc, stt.SpeechTimeoutError):
                raise HTTPException(
                    status_code=408,
                    detail="No speech detected within the time limit.",
                )
            elif isinstance(exc, stt.SpeechNotUnderstoodError):
                raise HTTPException(
                    status_code=422,
                    detail="Audio was not clear enough to understand.",
                )
            elif isinstance(exc, stt.SpeechAPIError):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Speech recognition service unavailable: "
                        f"{exc}"
                    ),
                )
            else:
                raise HTTPException(status_code=400, detail=f"Speech failed: {exc}")

        if voice_result is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No speech detected — please try again. If the server has "
                    "no microphone, supply a 'text' field instead."
                ),
            )
        result = voice_result

    # 2. Log the question event tied to the resolved owner.
    session = get_vision_session()
    persisted = session.attendance_service.add_question(
        student_name=student,
        question=result["question"],
        topic=result["topic"],
        topic_confidence=float(result.get("topic_confidence", 0.0)),
        source="interaction_ask_question",
    )

    return AskQuestionResponse(
        student=student,
        question=result["question"],
        topic=result["topic"],
        topic_confidence=float(result.get("topic_confidence", 0.0)),
        registered=bool(resolution["registered"]),
        is_guest=bool(resolution["is_guest"]),
        timestamp=result.get("timestamp")
        or (persisted.get("timestamp") if persisted else ""),
        resolved_from_active=bool(resolution["resolved_from_active"]),
    )
