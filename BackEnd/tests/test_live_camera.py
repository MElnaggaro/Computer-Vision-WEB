"""
Tests — Live Camera Integration
================================
This module provides two things:

1. **Automated tests** (run via ``pytest``) that verify the webcam‑runner
   pipeline processes frames correctly using mocked inputs — no real
   camera required.

2. **A manual test helper** (run directly as a script) that opens the
   real webcam so you can verify recognition in person.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     MANUAL TESTING INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prerequisites:
    1. You have student images in  data/students_faces/<StudentName>/
    2. You have installed all requirements:
           pip install -r requirements.txt
    3. You are inside the BackEnd/ directory.

Step 1 — Build encodings (one time):
    python -m app.services.vision.webcam_runner --rebuild
       OR
    curl -X POST http://localhost:8000/api/v1/vision/build-encodings

Step 2 — Run the live webcam test:
    python app/services/vision/webcam_runner.py

Step 3 — Verification checklist:
    ✅  Your name appears with a GREEN bounding box
    ✅  Confidence is displayed (e.g. "Mohammed_Ayman (91%)")
    ✅  A stranger shows a RED box with "Unknown (18%)"
    ✅  FPS counter is visible in the top‑left corner
    ✅  Press R → session resets, you can be re‑marked
    ✅  Press Q → camera closes, logs are saved
    ✅  Check  app/logs/classroom_log.json  for attendance records

Step 4 — Verify attendance log:
    type app\\logs\\classroom_log.json      (Windows)
    cat  app/logs/classroom_log.json        (macOS / Linux)

    Expected known student entry:
    {
      "student": "Mohammed_Ayman",
      "attendance": "Present",
      "known": true,
      "confidence": 0.91,
      "timestamp": "2026-05-10T12:30:22+00:00"
    }

    Expected unknown entry:
    {
      "student": "Unknown",
      "attendance": "Not Registered",
      "known": false,
      "confidence": 0.18,
      "timestamp": "2026-05-10T12:31:10+00:00"
    }
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure BackEnd/ is on sys.path when run directly ─────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import json
import pickle
from typing import List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.services.vision.attendance_service import AttendanceService
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.webcam_runner import ClassroomCamera


# ── Helpers ──────────────────────────────────────────────────────────

def _make_fake_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def _make_fake_encoding(seed: int = 42) -> np.ndarray:
    """Deterministic 128‑d face encoding."""
    rng = np.random.RandomState(seed)
    return rng.randn(128).astype(np.float64)


def _build_camera_with_mocked_encodings(
    tmp_path: Path,
    names: List[str],
    seeds: List[int],
) -> ClassroomCamera:
    """Create a ClassroomCamera wired to a fake encoding cache."""
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

    camera = ClassroomCamera(
        encoding_manager=manager,
        attendance_service=attendance,
    )
    return camera


# ── Automated Tests ──────────────────────────────────────────────────

class TestClassroomCamera:
    """Unit tests for the ClassroomCamera pipeline (no real webcam)."""

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_process_frame_known_student(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Processing a frame with a known face should mark attendance."""
        camera = _build_camera_with_mocked_encodings(
            tmp_path,
            names=["Mohammed_Ayman"],
            seeds=[1],
        )

        frame = _make_fake_frame()
        face_location = (50, 200, 200, 50)

        mock_detection_fr.face_locations.return_value = [face_location]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=1)]
        mock_recognition_fr.face_distance.return_value = np.array([0.1])

        annotated, results = camera.process_frame(frame)

        # Recognition worked
        assert len(results) == 1
        assert results[0]["name"] == "Mohammed_Ayman"
        assert results[0]["known"] is True

        # Attendance was marked
        assert camera.attendance_service.already_marked("Mohammed_Ayman")

        # Annotated frame is a valid image
        assert annotated.shape == frame.shape

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_process_frame_unknown_face(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Processing a frame with an unknown face returns Unknown."""
        camera = _build_camera_with_mocked_encodings(
            tmp_path,
            names=["Mohammed_Ayman"],
            seeds=[1],
        )

        frame = _make_fake_frame()
        face_location = (50, 200, 200, 50)

        mock_detection_fr.face_locations.return_value = [face_location]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=99)]
        mock_recognition_fr.face_distance.return_value = np.array([0.85])

        annotated, results = camera.process_frame(frame)

        assert len(results) == 1
        assert results[0]["name"] == "Unknown"
        assert results[0]["known"] is False

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_process_frame_no_faces(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Processing a frame with no faces returns an empty list."""
        camera = _build_camera_with_mocked_encodings(
            tmp_path, names=[], seeds=[],
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = []

        annotated, results = camera.process_frame(frame)

        assert results == []
        assert annotated.shape == frame.shape

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_duplicate_frames_mark_once(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Processing the same student across many frames marks only once."""
        camera = _build_camera_with_mocked_encodings(
            tmp_path,
            names=["Noreen_Osama"],
            seeds=[5],
        )

        frame = _make_fake_frame()
        face_location = (50, 200, 200, 50)
        mock_detection_fr.face_locations.return_value = [face_location]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=5)]
        mock_recognition_fr.face_distance.return_value = np.array([0.05])

        # Simulate 30 consecutive frames (like 1 second of video)
        for _ in range(30):
            camera.process_frame(frame)

        records = camera.attendance_service.records
        known_records = [r for r in records if r["known"] is True]
        assert len(known_records) == 1
        assert known_records[0]["student"] == "Noreen_Osama"

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_attendance_log_saved(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """After processing, the attendance log should be persistable."""
        camera = _build_camera_with_mocked_encodings(
            tmp_path,
            names=["Catherine_Adel"],
            seeds=[3],
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(10, 100, 100, 10)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=3)]
        mock_recognition_fr.face_distance.return_value = np.array([0.15])

        camera.process_frame(frame)
        log_path = camera.attendance_service.save_log()

        assert log_path.exists()
        with open(log_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert len(data) == 1
        assert data[0]["student"] == "Catherine_Adel"
        assert data[0]["attendance"] == "Present"
        assert data[0]["known"] is True
        assert "timestamp" in data[0]
        assert "confidence" in data[0]

    @patch("app.services.vision.face_recognizer.fr_lib")
    @patch("app.services.vision.face_detection.fr_lib")
    def test_session_reset_allows_remarking(
        self,
        mock_detection_fr: MagicMock,
        mock_recognition_fr: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Resetting the session should allow re‑marking."""
        camera = _build_camera_with_mocked_encodings(
            tmp_path,
            names=["Menna_Abdo"],
            seeds=[8],
        )

        frame = _make_fake_frame()
        mock_detection_fr.face_locations.return_value = [(50, 200, 200, 50)]
        mock_recognition_fr.face_encodings.return_value = [_make_fake_encoding(seed=8)]
        mock_recognition_fr.face_distance.return_value = np.array([0.08])

        # First marking
        camera.process_frame(frame)
        assert camera.attendance_service.already_marked("Menna_Abdo")

        # Reset
        camera.attendance_service.reset_session()
        assert not camera.attendance_service.already_marked("Menna_Abdo")

        # Can be marked again
        camera.process_frame(frame)
        assert camera.attendance_service.already_marked("Menna_Abdo")

    def test_draw_annotations_known(self) -> None:
        """Drawing annotations for a known face should add green elements."""
        frame = _make_fake_frame(height=300, width=400)
        results = [{
            "name": "Test_Student",
            "known": True,
            "confidence": 0.92,
            "location": (50, 200, 200, 50),
        }]

        annotated = ClassroomCamera._draw_annotations(frame.copy(), results)

        # The frame shape should be unchanged
        assert annotated.shape == frame.shape
        # Pixel data should differ (annotations were drawn)
        assert not np.array_equal(annotated, frame)

    def test_draw_annotations_unknown(self) -> None:
        """Drawing annotations for an unknown face should add red elements."""
        frame = _make_fake_frame(height=300, width=400)
        results = [{
            "name": "Unknown",
            "known": False,
            "confidence": 0.18,
            "location": (50, 200, 200, 50),
        }]

        annotated = ClassroomCamera._draw_annotations(frame.copy(), results)

        assert annotated.shape == frame.shape
        assert not np.array_equal(annotated, frame)

    def test_draw_annotations_empty(self) -> None:
        """Drawing with no results should return the frame unchanged."""
        frame = _make_fake_frame(height=300, width=400)
        original = frame.copy()
        annotated = ClassroomCamera._draw_annotations(frame, [])

        assert np.array_equal(annotated, original)


# ── Manual test entry point ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MANUAL LIVE CAMERA TEST")
    print("=" * 60)
    print()
    print("This will open your webcam and run face recognition.")
    print("Make sure encodings are built first:")
    print("  python -m app.services.vision.webcam_runner --rebuild")
    print()
    print("Controls:")
    print("  Q — Quit")
    print("  R — Reset session")
    print("  B — Rebuild encodings")
    print()
    print("Starting in 3 seconds …")

    import time
    time.sleep(3)

    from app.services.vision.webcam_runner import ClassroomCamera as _Camera
    cam = _Camera()
    cam.run()
