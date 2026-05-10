"""
Integration Tests — Guest Flow + Question Logging
==================================================
Validates the "Continue as Guest" pipeline end-to-end:

    1. POST /interaction/guest-session       → allocates Guest_NNN, logs attendance
    2. POST /interaction/ask-question        → routes question to that guest
    3. /logs/events                          → both events present, registered=False

Also covers monotonically-increasing guest ids and isolation between
sequential guest sessions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ── Ensure BackEnd/ is on sys.path ───────────────────────────────────
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
    """Fresh log file + tmpdir-backed VisionSession per test."""
    log_path = tmp_path / "log.json"
    log_service = LogService(log_file=log_path)

    vision_session.reset_vision_session()
    sess = vision_session.VisionSession(enable_emotion=False)
    sess.attendance_service = AttendanceService(log_file=log_path)
    monkeypatch.setattr(vision_session, "get_vision_session", lambda: sess)
    monkeypatch.setattr(
        "app.routes.vision_routes.get_vision_session", lambda: sess
    )
    monkeypatch.setattr(
        "app.routes.interaction_routes.get_vision_session", lambda: sess
    )

    yield {"log_path": log_path, "log_service": log_service, "session": sess}

    vision_session.reset_vision_session()


# ── Guest session creation ──────────────────────────────────────────


def test_guest_session_allocates_and_logs(isolated):
    """`POST /guest-session` returns a Guest_NNN id and persists an attendance event."""
    r = client.post("/api/v1/interaction/guest-session")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["student"].startswith("Guest_")
    assert body["registered"] is False
    assert body["is_guest"] is True

    events = isolated["log_service"].load_logs()
    assert len(events) == 1
    e = events[0]
    assert e["event"] == "attendance"
    assert e["student"] == body["student"]
    assert e["registered"] is False
    assert e["attendance"] == "Present"


def test_guest_ids_increment(isolated):
    """Sequential guests should receive ascending Guest_001, Guest_002, …"""
    a = client.post("/api/v1/interaction/guest-session").json()["student"]
    b = client.post("/api/v1/interaction/guest-session").json()["student"]
    c = client.post("/api/v1/interaction/guest-session").json()["student"]
    assert a == "Guest_001"
    assert b == "Guest_002"
    assert c == "Guest_003"


# ── Guest + question integration ────────────────────────────────────


def test_guest_question_attribution(isolated):
    """A question after `guest-session` is attributed to the guest id."""
    guest = client.post("/api/v1/interaction/guest-session").json()["student"]

    # Caller passes the guest id explicitly.
    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": guest, "text": "Explain semaphore"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["student"] == guest
    assert body["registered"] is False
    assert body["is_guest"] is True
    assert body["topic"] == "Operating System"

    events = isolated["log_service"].load_logs()
    types = [e["event"] for e in events]
    assert types == ["attendance", "question"]
    q_event = events[1]
    assert q_event["student"] == guest
    assert q_event["registered"] is False
    assert q_event["topic"] == "Operating System"


def test_guest_question_without_explicit_id_uses_active_session(isolated):
    """A blank-student request after `guest-session` falls back to the active guest."""
    guest = client.post("/api/v1/interaction/guest-session").json()["student"]

    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "", "text": "What is paging"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["student"] == guest
    assert body["is_guest"] is True
    assert body["resolved_from_active"] is True


def test_unknown_student_with_no_active_session_logs_unknown(isolated):
    """No active student + Unknown caller → still logged, registered=False."""
    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Unknown", "text": "What is convolution"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["student"] == "Unknown"
    assert body["registered"] is False
    assert body["is_guest"] is False

    events = isolated["log_service"].load_logs()
    assert any(e["event"] == "question" and e["student"] == "Unknown" for e in events)


def test_guest_summary_contains_question(isolated):
    """The guest's per-student summary should reflect the asked question."""
    guest = client.post("/api/v1/interaction/guest-session").json()["student"]
    client.post(
        "/api/v1/interaction/ask-question",
        json={"student": guest, "text": "What is TCP handshake"},
    )
    summary = isolated["session"].attendance_service.get_student_summary()
    assert any(s["student"] == guest for s in summary)
    guest_record = next(s for s in summary if s["student"] == guest)
    assert len(guest_record["questions"]) == 1
    assert guest_record["questions"][0]["topic"] == "Computer Networks"
    assert guest_record["registered"] is False
