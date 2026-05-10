"""
Tests — Vision + Emotion Integration
======================================
End-to-end integration tests that exercise the full pipeline:

    face detection → recognition → emotion → attendance log

No real webcam or GPU required — all heavy calls are mocked.

Run:
    cd BackEnd
    pytest tests/test_integration_vision_emotion.py -v
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Ensure BackEnd/ is on sys.path ────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.vision.attendance_service import AttendanceService
from app.services.vision.emotion_detection import EmotionDetector
from app.services.vision.emotion_tracker import EmotionTracker
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.webcam_runner import ClassroomCamera


# ══════════════════════════════════════════════════════════════════════
# Shared fixtures / helpers
# ══════════════════════════════════════════════════════════════════════

def _make_fake_frame(height: int = 480, width: int = 640) -> np.ndarray:
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def _make_fake_encoding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(128).astype(np.float64)


def _make_mock_emotion_tracker(label: str = "Happy", confidence: float = 0.87) -> EmotionTracker:
    """Return an EmotionTracker that always returns a fixed emotion."""
    mock_detector = MagicMock(spec=EmotionDetector)
    mock_detector.predict.return_value = {
        "label": label,
        "confidence": confidence,
        "raw_scores": {},
    }
    return EmotionTracker(
        emotion_interval=1,
        buffer_size=5,
        max_stale_frames=30,
        detector=mock_detector,
    )


def _build_camera(
    tmp_path: Path,
    names: List[str],
    seeds: List[int],
    emotion_label: str = "Happy",
    emotion_confidence: float = 0.87,
    enable_emotion: bool = True,
) -> ClassroomCamera:
    """Wire a ClassroomCamera with fake encodings and a controllable emotion tracker."""
    cache_file = tmp_path / "encodings.pkl"
    data = {
        "names": names,
        "encodings": [_make_fake_encoding(seed=s) for s in seeds],
    }
    with open(cache_file, "wb") as fh:
        pickle.dump(data, fh)

    manager = EncodingManager(
        students_dir=tmp_path / "students",
        encodings_file=cache_file,
    )
    manager.load_encodings()

    attendance = AttendanceService(log_file=tmp_path / "test_log.json")
    emotion_tracker = _make_mock_emotion_tracker(emotion_label, emotion_confidence) if enable_emotion else None

    return ClassroomCamera(
        encoding_manager=manager,
        attendance_service=attendance,
        emotion_tracker=emotion_tracker,
        enable_emotion=enable_emotion,
    )


# ══════════════════════════════════════════════════════════════════════
# Integration tests
# ══════════════════════════════════════════════════════════════════════

class TestKnownStudentWithEmotion:
    """Known student: recognition + emotion + log."""

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_known_student_has_emotion_in_result(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """stable_results for a known student should contain emotion fields."""
        camera = _build_camera(
            tmp_path, names=["Mohammed_Ayman"], seeds=[1],
            emotion_label="Happy", emotion_confidence=0.92,
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=1)]
        mock_recognition_fr.face_distance.return_value = np.array([0.1])

        # Run enough frames for stable identity
        for _ in range(20):
            _, results = camera.process_frame(frame)

        assert len(results) == 1
        result = results[0]
        assert result["name"] == "Mohammed_Ayman"
        assert result["registered"] is True
        assert "emotion" in result
        assert "emotion_confidence" in result
        assert isinstance(result["emotion"], str)
        assert 0.0 <= result["emotion_confidence"] <= 1.0

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_known_student_emotion_in_attendance_log(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Attendance log for known student should include emotion fields."""
        camera = _build_camera(
            tmp_path, names=["Catherine_Adel"], seeds=[3],
            emotion_label="Neutral", emotion_confidence=0.74,
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(10, 100, 100, 10)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=3)]
        mock_recognition_fr.face_distance.return_value = np.array([0.15])

        for _ in range(20):
            camera.process_frame(frame)

        log_path = camera.attendance_service.save_log()
        assert log_path.exists()

        with open(log_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert len(data) == 1
        record = data[0]
        assert record["student"] == "Catherine_Adel"
        assert record["attendance"] == "Present"
        assert record["registered"] is True
        assert "emotion" in record
        assert "emotion_confidence" not in record
        assert "timestamp" in record

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_known_student_marked_once_despite_many_frames(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Even with emotion enabled, attendance is only recorded once per student."""
        camera = _build_camera(
            tmp_path, names=["Menna_Abdo"], seeds=[8],
            emotion_label="Happy", emotion_confidence=0.88,
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=8)]
        mock_recognition_fr.face_distance.return_value = np.array([0.08])

        for _ in range(30):
            camera.process_frame(frame)

        records = camera.attendance_service.records
        registered_records = [r for r in records if r.get("registered") is True]
        assert len(registered_records) == 1
        assert registered_records[0]["student"] == "Menna_Abdo"


class TestUnknownFaceWithEmotion:
    """Unknown face: emotion still detected and logged."""

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_unknown_face_has_emotion_in_result(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """stable_results for an unknown face should still contain emotion fields."""
        camera = _build_camera(
            tmp_path, names=["Mohammed_Ayman"], seeds=[1],
            emotion_label="Neutral", emotion_confidence=0.55,
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=99)]
        mock_recognition_fr.face_distance.return_value = np.array([0.85])

        for _ in range(20):
            _, results = camera.process_frame(frame)

        assert len(results) == 1
        result = results[0]
        assert result["name"] == "Unknown"
        assert result["registered"] is False
        assert "emotion" in result
        assert isinstance(result["emotion"], str)

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_unknown_face_not_marked_in_log(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Unknown faces should NOT be added to the marked set."""
        camera = _build_camera(
            tmp_path, names=["Mohammed_Ayman"], seeds=[1],
            emotion_label="Neutral", emotion_confidence=0.55,
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=99)]
        mock_recognition_fr.face_distance.return_value = np.array([0.85])

        for _ in range(20):
            camera.process_frame(frame)

        # Unknown faces are never added to the attendance marked set
        assert not camera.attendance_service.already_marked("Unknown")


class TestEmotionDisabledMode:
    """When enable_emotion=False, pipeline still works with no emotion fields."""

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_pipeline_works_without_emotion(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Disabling emotion should still allow recognition and attendance."""
        camera = _build_camera(
            tmp_path, names=["Mohammed_Ayman"], seeds=[1],
            enable_emotion=False,
        )
        assert camera.emotion_tracker is None

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=1)]
        mock_recognition_fr.face_distance.return_value = np.array([0.1])

        for _ in range(20):
            annotated, results = camera.process_frame(frame)

        assert len(results) == 1
        assert results[0]["name"] == "Mohammed_Ayman"
        # Emotion should default to Neutral with 0.0 confidence
        assert results[0].get("emotion") == "Neutral"
        assert results[0].get("emotion_confidence") == 0.0
        assert annotated.shape == frame.shape


class TestLogSchemaWithEmotion:
    """Verify the exact JSON log schema after emotion integration."""

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_log_schema_known_student(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Log record for a known student must have all required fields."""
        camera = _build_camera(
            tmp_path, names=["Test_Student"], seeds=[7],
            emotion_label="Happy", emotion_confidence=0.92,
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=7)]
        mock_recognition_fr.face_distance.return_value = np.array([0.1])

        for _ in range(20):
            camera.process_frame(frame)
        camera.attendance_service.save_log()

        with open(tmp_path / "test_log.json", "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert len(data) == 1
        record = data[0]

        # Required fields (emotion_confidence removed)
        required_fields = {"student", "attendance", "timestamp", "registered", "emotion"}
        for field in required_fields:
            assert field in record, f"Missing field: {field}"

        assert "emotion_confidence" not in record
        assert record["student"] == "Test_Student"
        assert record["attendance"] == "Present"
        assert record["registered"] is True
        assert record["emotion"] == "Happy"

    def test_attendance_service_mark_with_emotion(self, tmp_path: Path) -> None:
        """AttendanceService.mark_attendance should store emotion fields."""
        svc = AttendanceService(log_file=tmp_path / "log.json")
        record = svc.mark_attendance(
            name="Mohammed_Ayman",
            registered=True,
            similarity=0.91,
            emotion="Happy",
            emotion_confidence=0.92,
        )

        assert record is not None
        assert record["emotion"] == "Happy"
        assert "emotion_confidence" not in record

    def test_attendance_service_mark_without_emotion(self, tmp_path: Path) -> None:
        """AttendanceService should work fine without emotion args (backward compat)."""
        svc = AttendanceService(log_file=tmp_path / "log.json")
        record = svc.mark_attendance(
            name="Catherine_Adel",
            registered=True,
            similarity=0.88,
        )

        assert record is not None
        assert "emotion" not in record
        assert "emotion_confidence" not in record

    def test_attendance_service_unknown_face_log(self, tmp_path: Path) -> None:
        """Unknown face log should follow the required schema."""
        svc = AttendanceService(log_file=tmp_path / "log.json")
        record = svc.mark_attendance(
            name="Unknown",
            registered=False,
            similarity=0.18,
            emotion="Neutral",
            emotion_confidence=0.74,
        )

        assert record is not None
        assert record["student"] == "Unknown"
        assert record["attendance"] == "Not Registered"
        assert record["registered"] is False
        assert record["emotion"] == "Neutral"
        assert "emotion_confidence" not in record
