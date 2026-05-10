"""
Tests — Vision Module
======================
Unit-level tests for the vision pipeline components:
    • EncodingManager (cache I/O)
    • FaceDetector / FaceRecognizer (mocked)
    • AttendanceService (event logging)
    • VisionSession (HTTP-friendly wrapper) with patched detectors

No real webcam, dlib model, or DeepFace inference is required — all
heavy dependencies are mocked so the suite runs on CI.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# ── Ensure BackEnd/ is on sys.path ───────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.vision.attendance_service import AttendanceService
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.face_recognizer import FaceRecognizer
from app.services.vision.vision_session import (
    VisionError,
    VisionSession,
    decode_base64_frame,
    reset_vision_session,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _fake_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return a tiny encoded JPEG (BGR all-zero) for round-trip tests."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _fake_jpeg_b64() -> str:
    return base64.b64encode(_fake_jpeg_bytes()).decode("ascii")


# ── decode_base64_frame ──────────────────────────────────────────────


class TestDecodeBase64Frame:
    def test_decodes_plain_base64(self):
        frame = decode_base64_frame(_fake_jpeg_b64())
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[2] == 3

    def test_strips_data_url_prefix(self):
        data_url = "data:image/jpeg;base64," + _fake_jpeg_b64()
        frame = decode_base64_frame(data_url)
        assert isinstance(frame, np.ndarray)

    def test_empty_payload_raises(self):
        with pytest.raises(VisionError):
            decode_base64_frame("")

    def test_invalid_data_raises(self):
        with pytest.raises(VisionError):
            decode_base64_frame("not-base64-and-not-an-image")


# ── EncodingManager (cache round-trip) ───────────────────────────────


class TestEncodingManager:
    def test_save_and_load_roundtrip(self, tmp_path):
        cache = tmp_path / "encodings.pkl"
        students_dir = tmp_path / "students"
        students_dir.mkdir()

        mgr = EncodingManager(students_dir=students_dir, encodings_file=cache)
        # Inject fake encodings directly to bypass dlib
        mgr._names = ["Alice_Smith", "Bob_Jones"]
        mgr._encodings = [np.zeros(128, dtype=np.float32), np.ones(128, dtype=np.float32)]
        mgr._save_cache()

        # Fresh instance — should load from disk
        mgr2 = EncodingManager(students_dir=students_dir, encodings_file=cache)
        loaded = mgr2.load_encodings()
        assert loaded is True
        assert mgr2.is_loaded
        assert mgr2.names == ["Alice_Smith", "Bob_Jones"]
        assert len(mgr2.encodings) == 2

    def test_load_missing_returns_false(self, tmp_path):
        mgr = EncodingManager(
            students_dir=tmp_path / "absent",
            encodings_file=tmp_path / "missing.pkl",
        )
        assert mgr.load_encodings() is False
        assert not mgr.is_loaded


# ── FaceRecognizer matching logic ────────────────────────────────────


class TestFaceRecognizer:
    def test_unknown_when_no_encodings(self):
        mgr = EncodingManager()
        mgr._names = []
        mgr._encodings = []
        rec = FaceRecognizer(encoding_manager=mgr, tolerance=0.6)
        result = rec._match_encoding(np.zeros(128, dtype=np.float32), (0, 10, 10, 0))
        assert result["registered"] is False
        assert result["name"] == "Unknown"

    def test_known_when_within_tolerance(self):
        mgr = EncodingManager()
        target = np.zeros(128, dtype=np.float32)
        mgr._names = ["Alice_Smith"]
        mgr._encodings = [target]
        rec = FaceRecognizer(encoding_manager=mgr, tolerance=0.6)

        with patch(
            "app.services.vision.face_recognizer.fr_lib.face_distance",
            return_value=np.array([0.1]),
        ):
            res = rec._match_encoding(target, (0, 50, 50, 0))
        assert res["name"] == "Alice_Smith"
        assert res["registered"] is True
        assert 0.0 <= res["similarity"] <= 1.0

    def test_unknown_when_distance_above_tolerance(self):
        mgr = EncodingManager()
        mgr._names = ["Alice_Smith"]
        mgr._encodings = [np.zeros(128, dtype=np.float32)]
        rec = FaceRecognizer(encoding_manager=mgr, tolerance=0.45)

        with patch(
            "app.services.vision.face_recognizer.fr_lib.face_distance",
            return_value=np.array([0.9]),
        ):
            res = rec._match_encoding(np.zeros(128), (0, 10, 10, 0))
        assert res["name"] == "Unknown"
        assert res["registered"] is False


# ── AttendanceService event logging ──────────────────────────────────


class TestAttendanceService:
    def test_mark_attendance_writes_event(self, tmp_path):
        log = tmp_path / "log.json"
        svc = AttendanceService(log_file=log)
        rec = svc.mark_attendance(
            name="Alice_Smith",
            registered=True,
            similarity=0.92,
            emotion="Happy",
            emotion_confidence=0.9,
        )
        assert rec is not None
        assert rec["student"] == "Alice_Smith"
        assert "Alice_Smith" in svc.marked_students
        events = svc._log_service.load_logs()
        assert any(e["event"] == "attendance" and e["student"] == "Alice_Smith" for e in events)

    def test_duplicate_mark_is_idempotent(self, tmp_path):
        svc = AttendanceService(log_file=tmp_path / "log.json")
        svc.mark_attendance(name="Alice_Smith", registered=True, similarity=0.9)
        again = svc.mark_attendance(name="Alice_Smith", registered=True, similarity=0.9)
        assert again is None  # second call is a no-op for already-marked student

    def test_unknown_recorded_with_not_registered(self, tmp_path):
        svc = AttendanceService(log_file=tmp_path / "log.json")
        rec = svc.mark_attendance(name="Unknown", registered=False, similarity=0.0)
        assert rec is not None
        assert rec["attendance"] == "Not Registered"
        assert rec["registered"] is False

    def test_add_question_emits_question_event(self, tmp_path):
        svc = AttendanceService(log_file=tmp_path / "log.json")
        svc.mark_attendance(name="Alice_Smith", registered=True, similarity=0.9)
        evt = svc.add_question(
            student_name="Alice_Smith",
            question="What is TCP handshake?",
            topic="Computer Networks",
            topic_confidence=0.9,
        )
        assert evt is not None
        assert evt["event"] == "question"
        assert evt["topic"] == "Computer Networks"
        assert evt["registered"] is True

    def test_get_active_student_picks_latest_registered(self, tmp_path):
        svc = AttendanceService(log_file=tmp_path / "log.json")
        svc.mark_attendance(name="Alice_Smith", registered=True, similarity=0.9)
        svc.mark_attendance(name="Bob_Jones", registered=True, similarity=0.85)
        assert svc.get_active_student() == "Bob_Jones"


# ── VisionSession with patched components ────────────────────────────


class TestVisionSession:
    def setup_method(self):
        reset_vision_session()

    def teardown_method(self):
        reset_vision_session()

    def test_recognize_frame_with_no_faces(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.services.vision.vision_session.AttendanceService",
            lambda *a, **kw: AttendanceService(log_file=tmp_path / "log.json"),
        )
        sess = VisionSession(enable_emotion=False)
        with patch.object(sess.face_detector, "detect_faces", return_value=[]):
            results = sess.recognize_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert results == []

    def test_recognize_frame_returns_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.services.vision.vision_session.AttendanceService",
            lambda *a, **kw: AttendanceService(log_file=tmp_path / "log.json"),
        )
        sess = VisionSession(enable_emotion=False)
        with patch.object(sess.face_detector, "detect_faces", return_value=[(10, 90, 90, 10)]), \
             patch.object(
                 sess.face_recognizer,
                 "recognize_faces",
                 return_value=[
                     {
                         "name": "Alice_Smith",
                         "registered": True,
                         "similarity": 0.9,
                         "distance": 0.1,
                         "location": (10, 90, 90, 10),
                     }
                 ],
             ):
            results = sess.recognize_frame(np.zeros((200, 200, 3), dtype=np.uint8))
        assert len(results) == 1
        r = results[0]
        assert r["name"] == "Alice_Smith"
        assert "emotion" in r

    def test_reset_attendance_clears_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.services.vision.vision_session.AttendanceService",
            lambda *a, **kw: AttendanceService(log_file=tmp_path / "log.json"),
        )
        sess = VisionSession(enable_emotion=False)
        sess.attendance_service.mark_attendance(
            name="Alice_Smith", registered=True, similarity=0.9
        )
        assert "Alice_Smith" in sess.attendance_service.marked_students
        sess.reset_attendance()
        assert sess.attendance_service.marked_students == set()
