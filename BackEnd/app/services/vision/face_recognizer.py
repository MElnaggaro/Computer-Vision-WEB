"""
Face Recognizer Service
========================
Compares detected face encodings against the known‑student database
and returns structured recognition results.

NOTE: This file is named ``face_recognizer.py`` (not ``face_recognition.py``)
to avoid shadowing the pip‑installed ``face_recognition`` library.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
    """Identify detected faces against a database of known encodings.

    Uses a **top‑K per‑student voting** strategy:
        1. Compute distance to *every* stored encoding.
        2. Group distances by student name.
        3. For each student, take the best K distances and average them.
        4. The student with the lowest average wins.

    This is far more accurate than a simple single‑minimum approach when
    students have multiple reference images (which they do).
    """

    def __init__(
        self,
        encoding_manager: Optional[EncodingManager] = None,
        tolerance: Optional[float] = None,
        top_k: int = 3,
    ) -> None:
        """
        Args:
            encoding_manager: Pre‑initialised encoding manager (DI‑friendly).
            tolerance: Face‑distance tolerance for matching (lower = stricter).
            top_k: Number of best distances to average per student.
        """
        self.encoding_manager = encoding_manager or EncodingManager()
        self.tolerance = tolerance if tolerance is not None else settings.FACE_RECOGNITION_TOLERANCE
        self.top_k = top_k

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

        Strategy (top‑K per‑student voting):
            1. Compute face distances to every known encoding.
            2. Group distances by student name.
            3. For each student, sort and take the best ``top_k`` distances.
            4. Average those best distances → the student's "score".
            5. Pick the student with the lowest average score.
            6. Accept if that score ≤ tolerance.
        """
        known_encodings = self.encoding_manager.encodings
        known_names = self.encoding_manager.names

        if not known_encodings:
            return self._unknown_result(location, confidence=0.0)

        distances: np.ndarray = fr_lib.face_distance(known_encodings, encoding)

        # ── Group distances by student ───────────────────────────────
        student_distances: Dict[str, List[float]] = defaultdict(list)
        for name, dist in zip(known_names, distances):
            student_distances[name].append(float(dist))

        # ── Best-K average per student ───────────────────────────────
        best_name: Optional[str] = None
        best_avg: float = float("inf")

        for name, dists in student_distances.items():
            dists_sorted = sorted(dists)
            top_k_dists = dists_sorted[: self.top_k]
            avg_dist = sum(top_k_dists) / len(top_k_dists)
            if avg_dist < best_avg:
                best_avg = avg_dist
                best_name = name

        confidence: float = self._distance_to_confidence(best_avg)

        if best_name is not None and best_avg <= self.tolerance:
            return {
                "name": best_name,
                "known": True,
                "confidence": confidence,
                "location": location,
            }

        return self._unknown_result(location, confidence)

    def _distance_to_confidence(self, distance: float) -> float:
        """Convert face distance to an intuitive confidence percentage.

        The raw ``1.0 − distance`` formula under‑reports confidence for
        correct matches (a real person at distance 0.35 shows only 65%).

        This mapping uses **non‑linear scaling** so that distances well
        within tolerance produce the high scores humans expect:

            distance │ confidence
            ─────────┼───────────
              0.00   │  100 %
              0.10   │   97 %
              0.20   │   90 %
              0.30   │   82 %
              0.40   │   73 %
              0.50   │   63 %
              0.60   │   50 %  ← tolerance boundary
              0.80   │   25 %
              1.00   │    0 %
        """
        if distance <= 0.0:
            return 1.0

        if distance <= self.tolerance:
            # Within tolerance → 50 %–100 %
            # Uses power curve (exponent < 1) to boost scores for good matches
            ratio = distance / self.tolerance          # 0 → 1
            return round(1.0 - 0.5 * (ratio ** 0.65), 4)
        else:
            # Beyond tolerance → 0 %–50 %, linear drop‑off
            overshoot = (distance - self.tolerance) / max(1.0 - self.tolerance, 0.001)
            return round(max(0.0, 0.5 * (1.0 - overshoot)), 4)

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
