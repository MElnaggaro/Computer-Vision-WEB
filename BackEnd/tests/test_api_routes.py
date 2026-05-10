"""
API Routes Integration Tests
=============================
End-to-end FastAPI tests via ``TestClient``. Heavy components (dlib,
DeepFace, real microphone, encoding rebuild) are stubbed so the suite
runs without hardware.
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

# ── Ensure BackEnd/ is on sys.path ────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.main import app
from app.services.vision import vision_session
from app.services.registration import registration_service

client = TestClient(app)


# ── Helpers ──────────────────────────────────────────────────────────


def _fake_b64() -> str:
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


@pytest.fixture(autouse=True)
def _isolate_singletons():
    """Reset session/registration singletons before each test."""
    vision_session.reset_vision_session()
    registration_service.reset_registration_service()
    yield
    vision_session.reset_vision_session()
    registration_service.reset_registration_service()


# ── Health ──────────────────────────────────────────────────────────


class TestHealth:
    def test_health_root(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "online"

    def test_health_v1(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "online"


# ── Speech ──────────────────────────────────────────────────────────


class TestSpeech:
    def test_speech_root(self):
        r = client.get("/api/v1/speech/")
        assert r.status_code == 200

    def test_speech_transcribe_mocked(self, monkeypatch):
        from app.services.speech.speech_to_text import SpeechRecognizer, SpeechResult

        monkeypatch.setattr(
            SpeechRecognizer, "listen_once",
            lambda self: SpeechResult(text="what is tcp", language="en-US"),
        )
        r = client.post("/api/v1/speech/transcribe")
        assert r.status_code == 200
        data = r.json()
        assert data["text"] == "what is tcp"


# ── NLP ─────────────────────────────────────────────────────────────


class TestNLP:
    def test_nlp_root(self):
        r = client.get("/api/v1/nlp/")
        assert r.status_code == 200

    def test_nlp_classify(self):
        r = client.post(
            "/api/v1/nlp/classify",
            json={"question": "How does the sliding window protocol work?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["topic"] == "Computer Networks"


# ── Vision ──────────────────────────────────────────────────────────


class TestVision:
    def test_vision_root(self):
        r = client.get("/api/v1/vision/")
        assert r.status_code == 200

    def test_recognize_frame(self, monkeypatch):
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
        r = client.post(
            "/api/v1/vision/recognize-frame",
            json={"image_base64": _fake_b64(), "mark_attendance": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["faces_detected"] == 1
        assert data["results"][0]["name"] == "Mohammed_Ayman"

    def test_recognize_frame_invalid_image(self):
        r = client.post(
            "/api/v1/vision/recognize-frame",
            json={"image_base64": "###bad###"},
        )
        assert r.status_code == 400

    def test_reset_attendance(self):
        r = client.post("/api/v1/vision/reset-attendance")
        assert r.status_code == 200
        assert "reset" in r.json()["message"].lower()

    def test_legacy_reset_session_alias(self):
        r = client.post("/api/v1/vision/reset-session")
        assert r.status_code == 200

    def test_state(self):
        r = client.get("/api/v1/vision/state")
        assert r.status_code == 200
        data = r.json()
        assert "students" in data
        assert "marked_count" in data


# ── Interaction (ask-question) ──────────────────────────────────────


class TestInteraction:
    def test_ask_question_text_mode(self):
        r = client.post(
            "/api/v1/interaction/ask-question",
            json={"student": "Mohammed_Ayman", "text": "What is TCP handshake?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["student"] == "Mohammed_Ayman"
        assert data["topic"] == "Computer Networks"
        assert data["question"] == "What is TCP handshake?"

    def test_ask_question_unknown_student(self):
        r = client.post(
            "/api/v1/interaction/ask-question",
            json={"student": "Unknown", "text": "Explain semaphore"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["student"] == "Unknown"
        assert data["registered"] is False

    def test_ask_question_voice_mode_mocked(self, monkeypatch):
        from app.services.speech.speech_to_text import SpeechRecognizer, SpeechResult

        monkeypatch.setattr(
            SpeechRecognizer,
            "listen_once",
            lambda self: SpeechResult(text="What is a semaphore?", language="en-US"),
        )
        r = client.post(
            "/api/v1/interaction/ask-question",
            json={"student": "Mohammed_Ayman"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["topic"] == "Operating System"


# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def _start(self):
        r = client.post("/api/v1/registration/start")
        assert r.status_code == 200
        return r.json()["session_id"]

    def _capture(self, sid, n=5):
        for _ in range(n):
            r = client.post(
                "/api/v1/registration/capture",
                json={"session_id": sid, "image_base64": _fake_b64()},
            )
            assert r.status_code == 200, r.text

    def test_full_flow_approved(self, monkeypatch, tmp_path):
        # Re-route registration service to a tmpdir so we don't pollute repo data
        from app.services.registration.registration_service import RegistrationService

        svc_instance = RegistrationService(
            pending_root=tmp_path / "pending",
            students_root=tmp_path / "students",
            admin_codeword="classroom-admin-2026",
        )
        monkeypatch.setattr(
            "app.routes.registration_routes.get_registration_service",
            lambda: svc_instance,
        )

        sid = self._start()
        self._capture(sid, n=5)

        r = client.post(
            "/api/v1/registration/submit",
            json={"session_id": sid, "name": "Ahmed_Ali"},
        )
        assert r.status_code == 200

        # Patch encoding rebuild
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
        assert (tmp_path / "students" / "Ahmed_Ali").is_dir()

    def test_reject_flow(self, monkeypatch, tmp_path):
        from app.services.registration.registration_service import RegistrationService

        svc_instance = RegistrationService(
            pending_root=tmp_path / "pending",
            students_root=tmp_path / "students",
            admin_codeword="classroom-admin-2026",
        )
        monkeypatch.setattr(
            "app.routes.registration_routes.get_registration_service",
            lambda: svc_instance,
        )

        sid = self._start()
        self._capture(sid, n=5)
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

    def test_invalid_admin_codeword(self, monkeypatch, tmp_path):
        from app.services.registration.registration_service import RegistrationService

        svc_instance = RegistrationService(
            pending_root=tmp_path / "pending",
            students_root=tmp_path / "students",
            admin_codeword="classroom-admin-2026",
        )
        monkeypatch.setattr(
            "app.routes.registration_routes.get_registration_service",
            lambda: svc_instance,
        )

        sid = self._start()
        self._capture(sid, n=5)
        client.post(
            "/api/v1/registration/submit",
            json={"session_id": sid, "name": "Try_Hacker"},
        )
        r = client.post(
            "/api/v1/registration/approve",
            json={"session_id": sid, "codeword": "WRONG"},
        )
        assert r.status_code == 401

    def test_invalid_name_format(self, monkeypatch, tmp_path):
        from app.services.registration.registration_service import RegistrationService

        svc_instance = RegistrationService(
            pending_root=tmp_path / "pending",
            students_root=tmp_path / "students",
            admin_codeword="classroom-admin-2026",
        )
        monkeypatch.setattr(
            "app.routes.registration_routes.get_registration_service",
            lambda: svc_instance,
        )

        sid = self._start()
        self._capture(sid, n=5)
        r = client.post(
            "/api/v1/registration/submit",
            json={"session_id": sid, "name": "ahmed ali"},
        )
        assert r.status_code == 422


# ── Events ──────────────────────────────────────────────────────────


class TestEvents:
    def test_events_returns_list(self):
        r = client.get("/api/v1/events")
        assert r.status_code == 200
        data = r.json()
        assert "events" in data and isinstance(data["events"], list)
