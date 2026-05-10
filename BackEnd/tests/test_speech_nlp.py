"""
Integration Tests — Speech + NLP Pipeline
==========================================
End-to-end flow that hits both the Speech route and the Interaction
route in the same request lifecycle.

Speech is mocked at the Google API boundary so no network or hardware
is required — but the NLP stage uses the real cached
``nlp_pipeline.joblib`` model so the topic predictions reflect what
production would emit.
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
from app.services.logging.log_service import LogService
from app.services.vision import vision_session
from app.services.vision.attendance_service import AttendanceService

client = TestClient(app)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    log_service = LogService(log_file=log_path)

    vision_session.reset_vision_session()
    sess = vision_session.VisionSession(enable_emotion=False)
    sess.attendance_service = AttendanceService(log_file=log_path)
    monkeypatch.setattr(vision_session, "get_vision_session", lambda: sess)
    monkeypatch.setattr(
        "app.routes.interaction_routes.get_vision_session", lambda: sess
    )
    yield {"log": log_service, "session": sess}
    vision_session.reset_vision_session()


# ── speech → ask-question (text path) ───────────────────────────────


def test_text_question_runs_nlp(isolated):
    """Sending text directly to ask-question still classifies + logs."""
    isolated["session"].attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    isolated["session"].set_active_student("Mohammed_Ayman", registered=True)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Mohammed_Ayman", "text": "What is TCP handshake?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["student"] == "Mohammed_Ayman"
    assert body["topic"] == "Computer Networks"
    assert 0.0 <= body["topic_confidence"] <= 1.0


# ── speech voice → NLP → log (mocked Google API) ─────────────────────


@patch("app.services.speech.speech_to_text.sr.Microphone")
@patch.object(
    sr.Recognizer,
    "recognize_google",
    return_value="Explain process synchronization with semaphores",
)
@patch.object(sr.Recognizer, "listen", return_value=MagicMock())
@patch.object(sr.Recognizer, "adjust_for_ambient_noise")
def test_voice_pipeline_attributes_and_logs(
    _ambient, _listen, _google, _mic, isolated
):
    """Voice → transcript → NLP → event log, all in one request."""
    sess = isolated["session"]
    sess.attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    sess.set_active_student("Mohammed_Ayman", registered=True)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Mohammed_Ayman"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["student"] == "Mohammed_Ayman"
    assert body["question"] == "Explain process synchronization with semaphores"
    assert body["topic"] == "Operating System"

    events = isolated["log"].load_logs()
    q = next(e for e in events if e["event"] == "question")
    assert q["student"] == "Mohammed_Ayman"
    assert q["topic"] == "Operating System"
    assert q["registered"] is True


@patch("app.services.speech.speech_to_text.sr.Microphone")
@patch.object(sr.Recognizer, "listen", side_effect=sr.WaitTimeoutError("timeout"))
@patch.object(sr.Recognizer, "adjust_for_ambient_noise")
def test_voice_pipeline_timeout_returns_408(_ambient, _listen, _mic, isolated):
    """Speech timeouts surface as a clean HTTP 408 from the integration route."""
    isolated["session"].attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    isolated["session"].set_active_student("Mohammed_Ayman", registered=True)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Mohammed_Ayman"},
    )
    assert r.status_code == 408


@patch("app.services.speech.speech_to_text.sr.Microphone")
@patch.object(sr.Recognizer, "recognize_google", side_effect=sr.RequestError("api down"))
@patch.object(sr.Recognizer, "listen", return_value=MagicMock())
@patch.object(sr.Recognizer, "adjust_for_ambient_noise")
def test_voice_pipeline_google_failure_returns_502(
    _ambient, _listen, _google, _mic, isolated
):
    """Google API failure on the speech leg surfaces as 502 (not a generic 500)."""
    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Unknown"},
    )
    assert r.status_code == 502
    assert "speech" in r.json()["detail"].lower()
