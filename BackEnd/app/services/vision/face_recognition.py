"""
Face Recognition Service
========================
Compares detected face encodings against the known‑student database
and returns structured recognition results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition as fr_lib
import numpy as np

from app.core.config import settings
from app.services.vision.encoding_manager import EncodingManager

logger = logging.getLogger(__name__)

# Type aliases
FaceLocation = Tuple[int, int, int, int]
RecognitionResult = Dict[str, Any]


class FaceRecognizer:
    """Identify detected faces against a database of known encodings."""

    def __init__(
        self,
        encoding_manager: Optional[EncodingManager] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        """
        Args:
            encoding_manager: Pre‑initialised encoding manager (DI‑friendly).
            tolerance: Face‑distance tolerance for matching (lower = stricter).
        """
        self.encoding_manager = encoding_manager or EncodingManager()
        self.tolerance = tolerance if tolerance is not None else settings.FACE_RECOGNITION_TOLERANCE

    # ── Public API ───────────────────────────────────────────────────

    def recognize_faces(
        self,
        frame: np.ndarray,
        face_locations: List[FaceLocation],
    ) -> List[RecognitionResult]:
        """Match each detected face to a known student.

        Args:
            frame: BGR image (from OpenCV).
            face_locations: Bounding boxes as returned by ``FaceDetector``.

        Returns:
            List of result dicts, one per face:
            ``{"name": str, "known": bool, "confidence": float, "location": tuple}``
        """
        if not self.encoding_manager.is_loaded:
            logger.warning("No encodings loaded – all faces will be Unknown.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Compute encodings for the faces found in this frame
        frame_encodings = fr_lib.face_encodings(rgb_frame, face_locations)

        results: List[RecognitionResult] = []
        for encoding, location in zip(frame_encodings, face_locations):
            result = self._match_encoding(encoding, location)
            results.append(result)

        return results

    # ── Internal ─────────────────────────────────────────────────────

    def _match_encoding(
        self,
        encoding: np.ndarray,
        location: FaceLocation,
    ) -> RecognitionResult:
        """Compare a single face encoding against all known encodings.

        Strategy:
            1. Compute face distances to every known encoding.
            2. Pick the minimum distance (best match).
            3. Convert distance → confidence (``1.0 - distance``).
            4. Accept if distance ≤ tolerance.
        """
        known_encodings = self.encoding_manager.encodings
        known_names = self.encoding_manager.names

        if not known_encodings:
            return self._unknown_result(location, confidence=0.0)

        distances: np.ndarray = fr_lib.face_distance(known_encodings, encoding)
        best_idx: int = int(np.argmin(distances))
        best_distance: float = float(distances[best_idx])
        confidence: float = round(1.0 - best_distance, 4)

        if best_distance <= self.tolerance:
            return {
                "name": known_names[best_idx],
                "known": True,
                "confidence": confidence,
                "location": location,
            }

        return self._unknown_result(location, confidence)

    @staticmethod
    def _unknown_result(
        location: FaceLocation,
        confidence: float,
    ) -> RecognitionResult:
        """Build a standard result dict for an unrecognised face."""
        return {
            "name": "Unknown",
            "known": False,
            "confidence": round(confidence, 4),
            "location": location,
        }
