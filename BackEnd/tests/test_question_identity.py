"""
Integration Tests — Question / Identity Ownership
==================================================
Validates that ``POST /interaction/ask-question`` always attributes
the question to the correct person:

    1. Explicit registered student wins.
    2. Empty/Unknown caller → resolves from the live vision session
       active student (face seen within the TTL).
    3. Active student expires → falls back to ``Unknown``.
    4. A registered face on camera takes ownership away from a guest.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
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
    # Tighten the TTL so the expiry test runs in milliseconds.
    sess.ACTIVE_STUDENT_TTL_SECONDS = 0.5
    monkeypatch.setattr(vision_session, "get_vision_session", lambda: sess)
    monkeypatch.setattr(
        "app.routes.interaction_routes.get_vision_session", lambda: sess
    )

    yield {"log": log_service, "session": sess}

    vision_session.reset_vision_session()


# ── Active-student resolution ───────────────────────────────────────


def test_explicit_registered_student_wins(isolated):
    """Explicit student id from the request always takes priority."""
    sess = isolated["session"]
    sess.attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    # Pin a different person as active to ensure explicit name wins.
    sess.attendance_service.mark_attendance(
        "Other_Student", registered=True, similarity=0.9
    )
    sess.set_active_student("Other_Student", registered=True)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Mohammed_Ayman", "text": "What is TCP"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["student"] == "Mohammed_Ayman"
    assert body["registered"] is True
    assert body["resolved_from_active"] is False


def test_empty_student_resolves_to_active(isolated):
    """Empty caller falls back to the live active student."""
    sess = isolated["session"]
    sess.attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    sess.set_active_student("Mohammed_Ayman", registered=True)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "", "text": "What is TCP"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["student"] == "Mohammed_Ayman"
    assert body["registered"] is True
    assert body["resolved_from_active"] is True


def test_unknown_caller_with_active_attribution(isolated):
    """Caller passes 'Unknown' but session has an active student → use it."""
    sess = isolated["session"]
    sess.attendance_service.mark_attendance(
        "Catherine_Adel", registered=True, similarity=0.9
    )
    sess.set_active_student("Catherine_Adel", registered=True)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Unknown", "text": "What is paging"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["student"] == "Catherine_Adel"
    assert body["resolved_from_active"] is True


def test_active_student_expires_falls_back_to_unknown(isolated):
    """If TTL expires, an Unknown caller logs as Unknown / not resolved."""
    sess = isolated["session"]
    sess.attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    sess.set_active_student("Mohammed_Ayman", registered=True)
    # Wait past the 0.5 s TTL configured by the fixture.
    time.sleep(0.7)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "", "text": "Explain semaphore"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["student"] == "Unknown"
    assert body["registered"] is False
    assert body["resolved_from_active"] is False


def test_registered_face_supersedes_guest(isolated):
    """A guest is active, but `set_active_student` for a registered student wins."""
    sess = isolated["session"]
    sess.register_guest()  # active = Guest_001
    assert sess.get_active_student() == "Guest_001"

    sess.attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    sess.set_active_student("Mohammed_Ayman", registered=True)

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "", "text": "What is TCP"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["student"] == "Mohammed_Ayman"
    assert body["is_guest"] is False
    assert body["resolved_from_active"] is True


def test_question_belongs_to_recognised_student_persisted(isolated):
    """Persisted log event reflects the resolved student."""
    sess = isolated["session"]
    sess.attendance_service.mark_attendance(
        "Mohammed_Ayman", registered=True, similarity=0.9
    )
    sess.set_active_student("Mohammed_Ayman", registered=True)

    client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Mohammed_Ayman", "text": "What is TCP handshake?"},
    )

    events = isolated["log"].load_logs()
    q_events = [e for e in events if e["event"] == "question"]
    assert len(q_events) == 1
    assert q_events[0]["student"] == "Mohammed_Ayman"
    assert q_events[0]["registered"] is True
    assert q_events[0]["topic"] == "Computer Networks"
