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
   recognition and just classifies the supplied text. Useful for the
   browser flow where speech-to-text happens client-side, or for tests.

In both modes the question is attributed to the ``student`` from the
request body. If the student is ``"Unknown"``, the event is logged
with ``registered: false`` (the guest flow).
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
        student: Student name to attribute the question to.  Pass
                 ``"Unknown"`` for the guest flow.
        text:    Optional pre-transcribed text.  When provided the
                 microphone is *not* opened.
    """

    student: str = Field(default="Unknown", description="Student to attribute question to")
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
    timestamp: str


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/")
async def interaction_root() -> Dict[str, str]:
    """Sub-router health-check."""
    return {"message": "Interaction routes operational"}


@router.post("/ask-question", response_model=AskQuestionResponse)
async def ask_question(payload: AskQuestionRequest) -> AskQuestionResponse:
    """Capture (or accept) a question and classify its topic for a given student."""
    student = (payload.student or "Unknown").strip() or "Unknown"

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
                raise HTTPException(status_code=408, detail="No speech detected within the time limit.")
            elif isinstance(exc, stt.SpeechNotUnderstoodError):
                raise HTTPException(status_code=422, detail="Audio was not clear enough to understand.")
            elif isinstance(exc, stt.SpeechAPIError):
                raise HTTPException(status_code=502, detail="Network error: Could not reach speech recognition service.")
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

    # 2. Log the question event tied to the recognised student.
    session = get_vision_session()
    persisted = session.attendance_service.add_question(
        student_name=student,
        question=result["question"],
        topic=result["topic"],
        topic_confidence=float(result.get("topic_confidence", 0.0)),
        source="interaction_ask_question",
    )

    registered = bool(persisted.get("registered")) if persisted else (
        student != "Unknown" and student in session.attendance_service.marked_students
    )

    return AskQuestionResponse(
        student=student,
        question=result["question"],
        topic=result["topic"],
        topic_confidence=float(result.get("topic_confidence", 0.0)),
        registered=registered,
        timestamp=result.get("timestamp")
        or (persisted.get("timestamp") if persisted else ""),
    )
