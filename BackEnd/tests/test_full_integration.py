"""
Full Integration Tests
=======================
End-to-end happy-path tests that exercise multiple modules at once,
mirroring the user-facing flows described in the project requirements:

    • Known student flow  — recognise + attendance + question
    • Unknown guest flow  — guest question event
    • Registration approved flow
    • Registration rejected flow
    • Backend-offline state via /health gating
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Ensure BackEnd/ is on sys.path ───────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.main import app
from app.services.logging.log_service import LogService
from app.services.registration import registration_service
from app.services.registration.registration_service import RegistrationService
from app.services.vision import vision_session
from app.services.vision.attendance_service import AttendanceService

client = TestClient(app)


def _fake_b64() -> str:
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Run each test against fresh, tmpdir-backed services."""
    log_path = tmp_path / "log.json"
    log_service = LogService(log_file=log_path)

    # Patch the module-level singleton accessor everywhere it's imported
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

    # Replace registration service with a tmpdir-backed instance
    reg_svc = RegistrationService(
        pending_root=tmp_path / "pending",
        students_root=tmp_path / "students",
        admin_codeword="classroom-admin-2026",
        log_service=log_service,
    )
    monkeypatch.setattr(
        "app.routes.registration_routes.get_registration_service", lambda: reg_svc
    )

    yield {"log_path": log_path, "log_service": log_service, "session": sess, "reg": reg_svc}

    vision_session.reset_vision_session()
    registration_service.reset_registration_service()


# ── Known student happy path ─────────────────────────────────────────


def test_known_student_flow(isolated, monkeypatch):
    """A known student is recognised, marked present, and asks a question."""
    from app.services.vision.face_detection import FaceDetector
    from app.services.vision.face_recognizer import FaceRecognizer

    monkeypatch.setattr(
        FaceDetector, "detect_faces", lambda *a, **k: [(10, 90, 90, 10)]
    )
    monkeypatch.setattr(
        FaceRecognizer,
        "recognize_faces",
        lambda *a, **k: [
            {
                "name": "Mohammed_Ayman",
                "registered": True,
                "similarity": 0.95,
                "distance": 0.05,
                "location": (10, 90, 90, 10),
            }
        ],
    )

    # Push enough frames for the FaceTracker to mark "stable + attendance_ready"
    # (needs TRACK_STABILITY_THRESHOLD + ATTENDANCE_STABLE_FRAMES = 6 + 10 = 16+)
    for _ in range(20):
        r = client.post(
            "/api/v1/vision/recognize-frame",
            json={"image_base64": _fake_b64()},
        )
        assert r.status_code == 200

    # Ask a question — must attribute to the recognised student
    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Mohammed_Ayman", "text": "What is TCP handshake?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["student"] == "Mohammed_Ayman"
    assert body["topic"] == "Computer Networks"

    # Verify both events landed in the unified log
    events = isolated["log_service"].load_logs()
    types = [e["event"] for e in events]
    assert "attendance" in types
    assert "question" in types
    q_event = next(e for e in events if e["event"] == "question")
    assert q_event["student"] == "Mohammed_Ayman"
    assert q_event["topic"] == "Computer Networks"


# ── Unknown guest flow ───────────────────────────────────────────────


def test_unknown_guest_question(isolated):
    """A guest (Unknown) can still ask a question; event is logged with registered=False."""
    r = client.post(
        "/api/v1/interaction/ask-question",
        json={"student": "Unknown", "text": "Explain semaphore"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["student"] == "Unknown"
    assert body["registered"] is False
    assert body["topic"] == "Operating System"

    events = isolated["log_service"].load_logs()
    q_event = next(e for e in events if e["event"] == "question")
    assert q_event["student"] == "Unknown"
    assert q_event.get("registered") is False


# ── Registration approved ────────────────────────────────────────────


def test_registration_approved_flow(isolated, monkeypatch):
    """Capture → submit → approve creates the student folder + event."""
    sid = client.post("/api/v1/registration/start").json()["session_id"]
    for _ in range(5):
        client.post(
            "/api/v1/registration/capture",
            json={"session_id": sid, "image_base64": _fake_b64()},
        )
    r = client.post(
        "/api/v1/registration/submit",
        json={"session_id": sid, "name": "Ahmed_Ali"},
    )
    assert r.status_code == 200

    with patch(
        "app.services.registration.registration_service.get_vision_session"
    ) as mock_vs:
        mock_vs.return_value.rebuild_encodings.return_value = {"total_encodings": 5}
        r = client.post(
            "/api/v1/registration/approve",
            json={"session_id": sid, "codeword": "classroom-admin-2026"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["approved"] is True
    assert body["student"] == "Ahmed_Ali"

    events = isolated["log_service"].load_logs()
    assert any(
        e["event"] == "registration_approved" and e["student"] == "Ahmed_Ali"
        for e in events
    )


# ── Registration rejected ────────────────────────────────────────────


def test_registration_rejected_flow(isolated):
    """Reject path emits a registration_rejected event and removes the pending folder."""
    sid = client.post("/api/v1/registration/start").json()["session_id"]
    for _ in range(5):
        client.post(
            "/api/v1/registration/capture",
            json={"session_id": sid, "image_base64": _fake_b64()},
        )
    client.post(
        "/api/v1/registration/submit",
        json={"session_id": sid, "name": "Reject_Me"},
    )
    r = client.post(
        "/api/v1/registration/reject",
        json={"session_id": sid, "delete_files": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] is False

    events = isolated["log_service"].load_logs()
    assert any(
        e["event"] == "registration_rejected" and e["student"] == "Reject_Me"
        for e in events
    )


# ── Health endpoint sanity ──────────────────────────────────────────


def test_health_reports_online():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "online"
