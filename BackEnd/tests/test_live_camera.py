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
      "registered": true,
      "emotion": "Happy",
      "timestamp": "2026-05-10T12:30:22+00:00"
    }

    Expected unknown entry:
    {
      "student": "Unknown",
      "attendance": "Not Registered",
      "registered": false,
      "emotion": "Neutral",
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
    students_dir = tmp_path / "students"
    students_dir.mkdir(parents=True, exist_ok=True)
    
    manager = EncodingManager(
        students_dir=students_dir,
        encodings_dir=tmp_path / "encodings",
    )
    
    # Manually inject mock encodings into the manager's memory
    for name, seed in zip(names, seeds):
        enc = _make_fake_encoding(seed)
        manager._mean_names.append(name)
        manager._mean_encodings.append(enc)
        manager._detailed_cache[name] = [enc]

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

        # Simulate 12 consecutive frames to trigger tracking stability
        for _ in range(20):
            annotated, results = camera.process_frame(frame)

        # Recognition worked
        assert len(results) == 1
        assert results[0]["name"] == "Mohammed_Ayman"
        assert results[0]["registered"] is True
        assert results[0].get("stable") is True

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

        for _ in range(20):
            annotated, results = camera.process_frame(frame)

        assert len(results) == 1
        assert results[0]["name"] == "Unknown"
        assert results[0]["registered"] is False
        assert results[0].get("stable") is True

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
        registered_records = [r for r in records if r["registered"] is True]
        assert len(registered_records) == 1
        assert registered_records[0]["student"] == "Noreen_Osama"

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

        for _ in range(20):
            camera.process_frame(frame)
        log_path = camera.attendance_service.save_log()

        assert log_path.exists()
        with open(log_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert len(data) == 1
        assert data[0]["student"] == "Catherine_Adel"
        assert data[0]["attendance"] == "Present"
        assert data[0]["registered"] is True
        assert "timestamp" in data[0]
        assert "similarity" not in data[0]

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
        for _ in range(20):
            camera.process_frame(frame)
        assert camera.attendance_service.already_marked("Menna_Abdo")

        # Reset
        camera.attendance_service.reset_session()
        camera.face_tracker.reset()
        assert not camera.attendance_service.already_marked("Menna_Abdo")

        # Can be marked again
        for _ in range(20):
            camera.process_frame(frame)
        assert camera.attendance_service.already_marked("Menna_Abdo")

    def test_draw_annotations_known(self) -> None:
        """Drawing annotations for a known face should add green elements."""
        frame = _make_fake_frame(height=300, width=400)
        results = [{
            "name": "Test_Student",
            "registered": True,
            "similarity": 0.92,
            "location": (50, 200, 200, 50),
            "stable": True,
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
            "registered": False,
            "similarity": 0.18,
            "location": (50, 200, 200, 50),
            "stable": True,
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
#
# The manual test now launches the *fast* (pure vision performance) runner
# defined in ``app.services.vision.fast_camera``.  Emotion / NLP / speech
# / web API are intentionally not loaded here — this is the
# ``<100 ms recognition`` performance-mode entry point.

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Live camera test — pure vision performance mode."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Downscale factor for detection/encoding (0.25 = fastest).",
    )
    parser.add_argument(
        "--recog-every",
        type=int,
        default=10,
        help="Re-recognize every N frames; tracking fills the gap.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild encodings before starting.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="Run in camera-only diagnostic mode.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FAST CAMERA — PURE VISION PERFORMANCE MODE")
    print("=" * 60)
    print()
    print("  Disabled: emotion · speech · NLP · web API · attendance polling")
    print("  Enabled : KCF tracking · vectorized matching · per-track identity cache")
    print()
    print("  Controls inside the OpenCV window:")
    print("    Q / Esc — Quit")
    print("    R       — Reset all tracks (fresh recognition next frame)")
    print("    B       — Rebuild encoding cache from data/students_faces/")
    print()
    print(f"  scale={args.scale}  recog-every={args.recog_every}  "
          f"resolution={args.width}x{args.height}")
    print()
    print("  Starting in 2 seconds …")
    time.sleep(2)

    from app.services.vision.fast_camera import FastClassroomCamera

    runner = FastClassroomCamera(
        camera_index=args.camera,
        scale=args.scale,
        recog_every=args.recog_every,
        cam_width=args.width,
        cam_height=args.height,
    )

    if args.test_camera:
        print("\n=== STARTING CAMERA DIAGNOSTIC MODE ===")
        cap = runner._open_capture()
        if cap is None:
            print("Failed to open camera in diagnostic mode.")
            sys.exit(1)
            
        print("Camera opened successfully. Showing raw frames. Press Q to exit.")
        try:
            fps = 0.0
            last_t = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: ret=False")
                    continue
                if frame is None or frame.size == 0:
                    print("Warning: Invalid frame")
                    continue
                
                t_now = time.time()
                dt = t_now - last_t
                last_t = t_now
                if dt > 0:
                    if fps == 0.0:
                        fps = 1.0 / dt
                    else:
                        fps = fps * 0.9 + (1.0 / dt) * 0.1
                
                cv2.putText(frame, f"RAW FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera Diagnostic Mode", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        sys.exit(0)

    if args.rebuild:
        print("Rebuilding encodings from data/students_faces/ …")
        runner.rebuild_encodings()
    runner.run()
