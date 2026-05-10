"""
Speech Routes
=============
HTTP entry points for server-side microphone transcription.

Endpoints
---------
* ``GET  /``           — sub-router health-check
* ``GET  /status``     — report whether a microphone is reachable
* ``POST /transcribe`` — open the server mic, capture one phrase, transcribe via Google.

The browser dashboard primarily uses the in-page Web Speech API, but
falls back to ``POST /transcribe`` when the browser cannot reach
Google directly (the "Speech service unavailable" error).  Because the
project is intended to run on the same host as the user, the server
mic is the same physical microphone.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.speech.audio_decode import decode_audio
from app.services.speech.speech_to_text import (
    SpeechAPIError,
    SpeechError,
    SpeechNotUnderstoodError,
    SpeechRecognizer,
    SpeechTimeoutError,
    transcribe_audio_data,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────


class TranscribeRequest(BaseModel):
    """Optional knobs for ``POST /transcribe``."""

    language: str = Field(default="en-US", description="BCP-47 language tag")
    timeout: int = Field(default=5, ge=1, le=30, description="Seconds to wait for speech")
    phrase_time_limit: int = Field(
        default=10, ge=1, le=60, description="Max seconds per utterance"
    )


class TranscribeResponse(BaseModel):
    text: str
    language: str


class SpeechStatusResponse(BaseModel):
    available: bool
    message: str


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/")
async def speech_root() -> dict:
    return {"message": "Speech routes operational"}


@router.get("/status", response_model=SpeechStatusResponse)
async def speech_status() -> SpeechStatusResponse:
    """Report whether the server can open a microphone.

    The check is intentionally lightweight: we just try to
    instantiate ``sr.Microphone()`` and immediately release it.  Any
    exception (no audio device, PyAudio missing, OSError) is reported
    as ``available=false`` with the exception message so the frontend
    can show an actionable error instead of a generic popup.
    """
    try:
        import speech_recognition as sr
        mic = sr.Microphone()
        # Acquire / release the underlying PyAudio stream to confirm
        # the device is actually openable, not just declared.
        with mic as _source:
            pass
        return SpeechStatusResponse(available=True, message="Microphone reachable")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Speech status probe failed: %s", exc)
        return SpeechStatusResponse(
            available=False,
            message=f"Microphone unavailable: {exc}",
        )


@router.post("/transcribe", response_model=TranscribeResponse)
async def speech_transcribe(payload: Optional[TranscribeRequest] = None) -> TranscribeResponse:
    """Activate the microphone, capture audio, and transcribe it to text.

    Returns clear HTTP error codes the frontend can map to UX:

    * ``408`` — no speech detected within the timeout
    * ``422`` — speech captured but not intelligible
    * ``502`` — Google Speech API request failed (no internet, blocked, …)
    * ``503`` — server mic could not be opened
    """
    cfg = payload or TranscribeRequest()
    recognizer = SpeechRecognizer(
        language=cfg.language,
        timeout=cfg.timeout,
        phrase_time_limit=cfg.phrase_time_limit,
    )
    try:
        result = recognizer.listen_once()
    except SpeechTimeoutError as exc:
        raise HTTPException(status_code=408, detail=str(exc)) from exc
    except SpeechNotUnderstoodError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpeechAPIError as exc:
        # Distinguish between "no microphone" and "google unreachable":
        # the underlying class wraps both, but the message is descriptive.
        msg = str(exc)
        status = 503 if msg.lower().startswith("microphone") else 502
        raise HTTPException(status_code=status, detail=msg) from exc
    except SpeechError as exc:  # generic catch-all for new subclasses
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TranscribeResponse(text=result.text, language=result.language)


@router.post("/transcribe-audio", response_model=TranscribeResponse)
async def speech_transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("en-US")
) -> TranscribeResponse:
    """Transcribe an uploaded audio file from the browser's MediaRecorder."""
    payload = await audio.read()
    
    logger.info("Received audio upload for transcription: %d bytes", len(payload))
    
    # Decode audio to SpeechRecognition's AudioData
    audio_data, report = decode_audio(payload)
    if not report.success or not audio_data:
        logger.error("Audio decode failed: %s", report.error)
        raise HTTPException(status_code=400, detail=f"Audio decode failed: {report.error}")
        
    try:
        # Transcribe with Google
        result = transcribe_audio_data(audio_data, language=language)
        return TranscribeResponse(text=result.text, language=result.language)
    except SpeechNotUnderstoodError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpeechAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error during transcription: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/debug")
async def speech_debug(
    audio: UploadFile = File(...),
    language: str = Form("en-US")
) -> Dict[str, Any]:
    """Test endpoint that returns detailed diagnostic information."""
    payload = await audio.read()
    
    logger.info("--- DEBUG ENDPOINT ACCESSED ---")
    logger.info("1) frontend request received. file=%s", audio.filename)
    logger.info("2) audio bytes length: %d", len(payload))
    logger.info("3) speech service entered")
    
    audio_data, report = decode_audio(payload)
    
    logger.info("4) audio parsing success: %s", report.success)
    if report.error:
        logger.error("Decode error: %s", report.error)
        
    decode_success = report.success and audio_data is not None
    
    response = {
        "received_audio": True,
        "audio_size": len(payload),
        "decode_success": decode_success,
        "google_request_success": False,
        "error": None,
        "decode_report": {
            "format": report.detected_format,
            "duration_ms": report.duration_ms,
            "used_path": report.used_path
        }
    }
    
    if not decode_success:
        response["error"] = report.error or "Unknown decode failure"
        return response
        
    logger.info("5) exact recognizer being used: google")
    try:
        result = transcribe_audio_data(audio_data, language=language)
        response["google_request_success"] = True
        response["text"] = result.text
        logger.info("6) Google request success, text=%s", result.text)
    except Exception as exc:
        import traceback
        logger.error("7) exact exception traceback:\n%s", traceback.format_exc())
        response["google_request_success"] = False
        response["error"] = f"{type(exc).__name__}: {str(exc)}"
        
    return response
