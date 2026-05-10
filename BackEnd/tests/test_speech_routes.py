"""
Integration Tests — Speech Route
=================================
Validates that ``/api/v1/speech/transcribe`` returns clean HTTP error
codes the frontend can map to UX, and that ``/api/v1/speech/status``
reports microphone availability without raising.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import speech_recognition as sr
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.main import app

client = TestClient(app)


# ── /speech root ────────────────────────────────────────────────────


def test_speech_root_ok():
    r = client.get("/api/v1/speech/")
    assert r.status_code == 200
    assert "operational" in r.json()["message"]


# ── /speech/status ──────────────────────────────────────────────────


def test_status_does_not_raise():
    """Status probe must always return 200 with an `available` flag."""
    r = client.get("/api/v1/speech/status")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["available"], bool)
    assert isinstance(body["message"], str)


# ── /speech/transcribe ──────────────────────────────────────────────


@patch("app.services.speech.speech_to_text.sr.Microphone")
@patch.object(sr.Recognizer, "recognize_google", return_value="Can you explain convolution")
@patch.object(sr.Recognizer, "listen", return_value=MagicMock())
@patch.object(sr.Recognizer, "adjust_for_ambient_noise")
def test_transcribe_success(*_mocks):
    r = client.post("/api/v1/speech/transcribe")
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Can you explain convolution"
    assert body["language"] == "en-US"


@patch("app.services.speech.speech_to_text.sr.Microphone")
@patch.object(sr.Recognizer, "listen", side_effect=sr.WaitTimeoutError("timeout"))
@patch.object(sr.Recognizer, "adjust_for_ambient_noise")
def test_transcribe_timeout_returns_408(*_mocks):
    r = client.post("/api/v1/speech/transcribe")
    assert r.status_code == 408


@patch("app.services.speech.speech_to_text.sr.Microphone")
@patch.object(sr.Recognizer, "recognize_google", side_effect=sr.UnknownValueError())
@patch.object(sr.Recognizer, "listen", return_value=MagicMock())
@patch.object(sr.Recognizer, "adjust_for_ambient_noise")
def test_transcribe_unclear_returns_422(*_mocks):
    r = client.post("/api/v1/speech/transcribe")
    assert r.status_code == 422


@patch("app.services.speech.speech_to_text.sr.Microphone")
@patch.object(sr.Recognizer, "recognize_google", side_effect=sr.RequestError("api down"))
@patch.object(sr.Recognizer, "listen", return_value=MagicMock())
@patch.object(sr.Recognizer, "adjust_for_ambient_noise")
def test_transcribe_google_failure_returns_502(*_mocks):
    r = client.post("/api/v1/speech/transcribe")
    assert r.status_code == 502
    assert "google" in r.json()["detail"].lower()


def test_transcribe_accepts_optional_overrides():
    """Posting an override body must not break parsing (404/400/etc come from speech, not pydantic)."""
    with patch("app.services.speech.speech_to_text.sr.Microphone"), \
         patch.object(sr.Recognizer, "recognize_google", return_value="hello world"), \
         patch.object(sr.Recognizer, "listen", return_value=MagicMock()), \
         patch.object(sr.Recognizer, "adjust_for_ambient_noise"):
        r = client.post(
            "/api/v1/speech/transcribe",
            json={"language": "en-GB", "timeout": 3, "phrase_time_limit": 5},
        )
        assert r.status_code == 200
        assert r.json()["language"] == "en-GB"


# ── /speech/transcribe-audio ────────────────────────────────────────


@patch("app.routes.speech_routes.decode_audio")
@patch("app.routes.speech_routes.transcribe_audio_data")
def test_transcribe_audio_success(mock_transcribe, mock_decode):
    from app.services.speech.audio_decode import DecodeReport
    from app.services.speech.speech_to_text import SpeechResult
    
    mock_decode.return_value = (MagicMock(), DecodeReport(
        success=True, used_path="native", bytes_in=10, 
        sample_rate=None, sample_width=None, duration_ms=None, 
        detected_format="wav", error=None
    ))
    mock_transcribe.return_value = SpeechResult("hello audio", "en-US", True)
    
    r = client.post(
        "/api/v1/speech/transcribe-audio",
        files={"audio": ("test.wav", b"fake audio data", "audio/wav")}
    )
    assert r.status_code == 200
    assert r.json()["text"] == "hello audio"


@patch("app.routes.speech_routes.decode_audio")
def test_transcribe_audio_decode_failure(mock_decode):
    from app.services.speech.audio_decode import DecodeReport
    
    mock_decode.return_value = (None, DecodeReport(
        success=False, used_path="n/a", bytes_in=0, 
        sample_rate=None, sample_width=None, duration_ms=None, 
        detected_format="unknown", error="invalid format"
    ))
    
    r = client.post(
        "/api/v1/speech/transcribe-audio",
        files={"audio": ("test.webm", b"bad data", "audio/webm")}
    )
    assert r.status_code == 400
    assert "invalid format" in r.json()["detail"]


# ── /speech/debug ───────────────────────────────────────────────────


@patch("app.routes.speech_routes.decode_audio")
@patch("app.routes.speech_routes.transcribe_audio_data")
def test_debug_endpoint(mock_transcribe, mock_decode):
    from app.services.speech.audio_decode import DecodeReport
    from app.services.speech.speech_to_text import SpeechResult
    
    mock_decode.return_value = (MagicMock(), DecodeReport(
        success=True, used_path="native", bytes_in=10, 
        sample_rate=None, sample_width=None, duration_ms=None, 
        detected_format="wav", error=None
    ))
    mock_transcribe.return_value = SpeechResult("debug text", "en-US", True)
    
    r = client.post(
        "/api/v1/speech/debug",
        files={"audio": ("test.wav", b"fake audio data", "audio/wav")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["received_audio"] is True
    assert body["decode_success"] is True
    assert body["google_request_success"] is True
    assert body["text"] == "debug text"
