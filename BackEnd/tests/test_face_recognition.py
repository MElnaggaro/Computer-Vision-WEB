"""
Tests — Face Recognition Service
=================================
Covers:
    • Known face recognition
    • Unknown face handling
    • Edge cases (no encodings loaded, empty frame)
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognition import FaceRecognizer


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

    @patch("app.services.vision.face_detection.face_recognition")
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

    @patch("app.services.vision.face_detection.face_recognition")
    def test_detect_faces_no_faces_found(self, mock_fr: MagicMock) -> None:
        """Detector should return an empty list when no faces are found."""
        mock_fr.face_locations.return_value = []

        detector = FaceDetector(model="hog")
        frame = _make_fake_frame()
        result = detector.detect_faces(frame)

        assert result == []


# ── Face Recognizer Tests ────────────────────────────────────────────

class TestFaceRecognizer:
    """Unit tests for FaceRecognizer."""

    def _build_recognizer(
        self,
        names: List[str],
        encodings: List[np.ndarray],
        tolerance: float = 0.6,
    ) -> FaceRecognizer:
        """Create a recognizer with a pre‑populated encoding manager."""
        manager = EncodingManager.__new__(EncodingManager)
        manager._names = names
        manager._encodings = encodings
        return FaceRecognizer(encoding_manager=manager, tolerance=tolerance)

    @patch("app.services.vision.face_recognition.fr_lib")
    def test_known_face_recognition(self, mock_fr: MagicMock) -> None:
        """A known face should be recognised with ``known=True``."""
        known_encoding = _make_fake_encoding(seed=1)
        recognizer = self._build_recognizer(
            names=["Mohammed_Ayman"],
            encodings=[known_encoding],
            tolerance=0.6,
        )

        frame = _make_fake_frame()
        locations = [(50, 200, 200, 50)]

        # Simulate face_encodings returning the SAME encoding (perfect match)
        mock_fr.face_encodings.return_value = [known_encoding]
        # Distance of 0.0 → perfect match
        mock_fr.face_distance.return_value = np.array([0.0])

        results = recognizer.recognize_faces(frame, locations)

        assert len(results) == 1
        assert results[0]["name"] == "Mohammed_Ayman"
        assert results[0]["known"] is True
        assert results[0]["confidence"] == 1.0
        assert results[0]["location"] == (50, 200, 200, 50)

    @patch("app.services.vision.face_recognition.fr_lib")
    def test_unknown_face_handling(self, mock_fr: MagicMock) -> None:
        """An unfamiliar face should return ``name='Unknown'``."""
        known_encoding = _make_fake_encoding(seed=1)
        recognizer = self._build_recognizer(
            names=["Mohammed_Ayman"],
            encodings=[known_encoding],
            tolerance=0.6,
        )

        frame = _make_fake_frame()
        locations = [(50, 200, 200, 50)]

        unknown_encoding = _make_fake_encoding(seed=99)
        mock_fr.face_encodings.return_value = [unknown_encoding]
        # Distance > tolerance → unknown
        mock_fr.face_distance.return_value = np.array([0.85])

        results = recognizer.recognize_faces(frame, locations)

        assert len(results) == 1
        assert results[0]["name"] == "Unknown"
        assert results[0]["known"] is False
        assert results[0]["confidence"] < 0.5

    @patch("app.services.vision.face_recognition.fr_lib")
    def test_multiple_faces_mixed(self, mock_fr: MagicMock) -> None:
        """Multiple faces: one known, one unknown."""
        enc_a = _make_fake_encoding(seed=1)
        enc_b = _make_fake_encoding(seed=2)
        recognizer = self._build_recognizer(
            names=["Student_A", "Student_B"],
            encodings=[enc_a, enc_b],
            tolerance=0.6,
        )

        frame = _make_fake_frame()
        locations = [(10, 100, 100, 10), (50, 200, 200, 50)]

        known_frame_enc = _make_fake_encoding(seed=1)
        unknown_frame_enc = _make_fake_encoding(seed=99)
        mock_fr.face_encodings.return_value = [known_frame_enc, unknown_frame_enc]

        # First call for known_frame_enc → close to enc_a
        # Second call for unknown_frame_enc → far from both
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

    @patch("app.services.vision.face_recognition.fr_lib")
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
