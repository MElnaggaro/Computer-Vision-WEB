"""
Tests — Face Recognition Service
=================================
Covers:
    • Known face recognition
    • Unknown face handling
    • Invalid / corrupt image handling
    • Edge cases (no encodings loaded, empty frame)
    • EncodingManager build and load
"""

import sys
from pathlib import Path

# ── Ensure BackEnd/ is on sys.path when run directly ─────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pickle
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognizer import FaceRecognizer


# ── Helpers ──────────────────────────────────────────────────────────

def _make_fake_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a dummy BGR frame (random noise)."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def _make_fake_encoding(seed: int = 42) -> np.ndarray:
    """Deterministic 128‑d face encoding for reproducible tests."""
    rng = np.random.RandomState(seed)
    return rng.randn(128).astype(np.float64)


# ── Face Detector Tests ──────────────────────────────────────────────

class TestFaceDetector:
    """Unit tests for FaceDetector."""

    @patch("app.services.vision.face_detection.fr_lib")
    def test_detect_faces_returns_locations(self, mock_fr: MagicMock) -> None:
        """Detector should return the locations list from face_recognition."""
        fake_locations = [(50, 200, 200, 50)]
        mock_fr.face_locations.return_value = fake_locations

        detector = FaceDetector(model="hog")
        frame = _make_fake_frame()
        result = detector.detect_faces(frame)

        assert result == fake_locations
        mock_fr.face_locations.assert_called_once()

    def test_detect_faces_empty_frame(self) -> None:
        """Detector should handle an empty frame gracefully."""
        detector = FaceDetector(model="hog")
        empty_frame = np.array([], dtype=np.uint8)
        result = detector.detect_faces(empty_frame)
        assert result == []

    @patch("app.services.vision.face_detection.fr_lib")
    def test_detect_faces_no_faces_found(self, mock_fr: MagicMock) -> None:
        """Detector should return an empty list when no faces are found."""
        mock_fr.face_locations.return_value = []

        detector = FaceDetector(model="hog")
        frame = _make_fake_frame()
        result = detector.detect_faces(frame)

        assert result == []

    def test_detect_faces_none_frame(self) -> None:
        """Detector should return [] for a None frame without crashing."""
        detector = FaceDetector(model="hog")
        result = detector.detect_faces(None)  # type: ignore[arg-type]
        assert result == []


# ── Face Recognizer Tests ────────────────────────────────────────────

class TestFaceRecognizer:
    """Unit tests for FaceRecognizer."""

    def _build_recognizer(
        self,
        names: List[str],
        encodings: List[np.ndarray],
        tolerance: float = 0.5,
    ) -> FaceRecognizer:
        """Create a recognizer with a pre‑populated encoding manager."""
        manager = EncodingManager.__new__(EncodingManager)
        manager._names = names
        manager._encodings = encodings
        return FaceRecognizer(encoding_manager=manager, tolerance=tolerance)

    @patch("app.services.vision.face_recognizer.fr_lib")
    def test_known_face_recognition(self, mock_fr: MagicMock) -> None:
        """A known face should be recognised with ``known=True``."""
        known_encoding = _make_fake_encoding(seed=1)
        recognizer = self._build_recognizer(
            names=["Mohammed_Ayman"],
            encodings=[known_encoding],
            tolerance=0.5,
        )

        frame = _make_fake_frame()
        locations = [(50, 200, 200, 50)]

        mock_fr.face_encodings.return_value = [known_encoding]
        mock_fr.face_distance.return_value = np.array([0.0])

        results = recognizer.recognize_faces(frame, locations)

        assert len(results) == 1
        assert results[0]["name"] == "Mohammed_Ayman"
        assert results[0]["known"] is True
        assert results[0]["confidence"] == 1.0
        assert results[0]["location"] == (50, 200, 200, 50)

    @patch("app.services.vision.face_recognizer.fr_lib")
    def test_unknown_face_handling(self, mock_fr: MagicMock) -> None:
        """An unfamiliar face should return ``name='Unknown'``."""
        known_encoding = _make_fake_encoding(seed=1)
        recognizer = self._build_recognizer(
            names=["Mohammed_Ayman"],
            encodings=[known_encoding],
            tolerance=0.5,
        )

        frame = _make_fake_frame()
        locations = [(50, 200, 200, 50)]

        unknown_encoding = _make_fake_encoding(seed=99)
        mock_fr.face_encodings.return_value = [unknown_encoding]
        mock_fr.face_distance.return_value = np.array([0.85])

        results = recognizer.recognize_faces(frame, locations)

        assert len(results) == 1
        assert results[0]["name"] == "Unknown"
        assert results[0]["known"] is False
        assert results[0]["confidence"] < 0.5

    @patch("app.services.vision.face_recognizer.fr_lib")
    def test_multiple_faces_mixed(self, mock_fr: MagicMock) -> None:
        """Multiple faces: one known, one unknown."""
        enc_a = _make_fake_encoding(seed=1)
        enc_b = _make_fake_encoding(seed=2)
        recognizer = self._build_recognizer(
            names=["Student_A", "Student_B"],
            encodings=[enc_a, enc_b],
            tolerance=0.5,
        )

        frame = _make_fake_frame()
        locations = [(10, 100, 100, 10), (50, 200, 200, 50)]

        known_frame_enc = _make_fake_encoding(seed=1)
        unknown_frame_enc = _make_fake_encoding(seed=99)
        mock_fr.face_encodings.return_value = [known_frame_enc, unknown_frame_enc]

        mock_fr.face_distance.side_effect = [
            np.array([0.1, 0.7]),    # matches Student_A
            np.array([0.9, 0.85]),   # matches neither
        ]

        results = recognizer.recognize_faces(frame, locations)

        assert len(results) == 2
        assert results[0]["name"] == "Student_A"
        assert results[0]["known"] is True
        assert results[1]["name"] == "Unknown"
        assert results[1]["known"] is False

    @patch("app.services.vision.face_recognizer.fr_lib")
    def test_no_encodings_loaded(self, mock_fr: MagicMock) -> None:
        """If no encodings are loaded, all faces should be Unknown."""
        recognizer = self._build_recognizer(
            names=[],
            encodings=[],
        )

        frame = _make_fake_frame()
        locations = [(50, 200, 200, 50)]
        mock_fr.face_encodings.return_value = [_make_fake_encoding(seed=1)]

        results = recognizer.recognize_faces(frame, locations)

        assert len(results) == 1
        assert results[0]["name"] == "Unknown"
        assert results[0]["known"] is False

    def test_empty_locations(self) -> None:
        """No face locations should return an empty results list."""
        recognizer = self._build_recognizer(names=[], encodings=[])
        frame = _make_fake_frame()

        results = recognizer.recognize_faces(frame, [])

        assert results == []


# ── Invalid Input Tests ──────────────────────────────────────────────

class TestInvalidInput:
    """Tests for corrupt / invalid image data handling."""

    def test_encoding_manager_invalid_image_file(self, tmp_path: Path) -> None:
        """EncodingManager should skip a file that is not a valid image."""
        student_dir = tmp_path / "students" / "Fake_Student"
        student_dir.mkdir(parents=True)

        # Write garbage bytes pretending to be a JPEG
        corrupt_file = student_dir / "corrupt.jpg"
        corrupt_file.write_bytes(b"NOT_A_REAL_IMAGE_DATA_STREAM")

        manager = EncodingManager(
            students_dir=tmp_path / "students",
            encodings_file=tmp_path / "enc.pkl",
        )
        summary = manager.build_encodings()

        # The corrupt file should be in the skipped list
        assert len(summary["skipped"]) >= 1
        assert summary["total_encodings"] == 0

    def test_encoding_manager_missing_directory(self, tmp_path: Path) -> None:
        """EncodingManager should raise FileNotFoundError for missing dir."""
        manager = EncodingManager(
            students_dir=tmp_path / "does_not_exist",
            encodings_file=tmp_path / "enc.pkl",
        )
        with pytest.raises(FileNotFoundError):
            manager.build_encodings()

    def test_encoding_manager_empty_student_dir(self, tmp_path: Path) -> None:
        """EncodingManager should handle an empty students directory."""
        students_dir = tmp_path / "students"
        students_dir.mkdir()

        manager = EncodingManager(
            students_dir=students_dir,
            encodings_file=tmp_path / "enc.pkl",
        )
        summary = manager.build_encodings()

        assert summary["total_encodings"] == 0
        assert summary["students"] == {}

    def test_encoding_manager_corrupt_cache(self, tmp_path: Path) -> None:
        """EncodingManager should return False for a corrupt pickle cache."""
        cache_file = tmp_path / "bad_cache.pkl"
        cache_file.write_bytes(b"CORRUPT_PICKLE_DATA")

        manager = EncodingManager(
            students_dir=tmp_path / "students",
            encodings_file=cache_file,
        )
        loaded = manager.load_encodings()

        assert loaded is False
        assert manager.is_loaded is False

    def test_encoding_manager_load_valid_cache(self, tmp_path: Path) -> None:
        """EncodingManager should successfully load a valid pickle cache."""
        cache_file = tmp_path / "good_cache.pkl"
        data = {
            "names": ["Test_Student"],
            "encodings": [_make_fake_encoding(seed=7)],
        }
        with open(cache_file, "wb") as fh:
            pickle.dump(data, fh)

        manager = EncodingManager(
            students_dir=tmp_path / "students",
            encodings_file=cache_file,
        )
        loaded = manager.load_encodings()

        assert loaded is True
        assert manager.is_loaded is True
        assert manager.names == ["Test_Student"]
        assert len(manager.encodings) == 1
