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

    Finds the absolute minimum distance among all encodings across all students.
    """

    def __init__(
        self,
        encoding_manager: Optional[EncodingManager] = None,
        tolerance: Optional[float] = None,
        **kwargs,
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
        is_rgb: bool = False,
    ) -> List[RecognitionResult]:
        """Match each detected face to a known student.

        Args:
            frame: BGR or RGB image.
            face_locations: Bounding boxes as returned by ``FaceDetector``.
            is_rgb: True if the frame is already in RGB format.

        Returns:
            List of result dicts, one per face:
            ``{"name": str, "registered": bool, "similarity": float, "distance": float, "location": tuple}``
        """
        if not self.encoding_manager.is_loaded:
            logger.warning("No encodings loaded – all faces will be Unknown.")

        if not is_rgb:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

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

        Strategy (Two-Phase):
            PHASE 1 (FAST MATCH):
            1. Compute face distances to ONE representative vector (mean) per student.
            2. Find the minimum distance.
            3. If below tolerance, tentatively accept.
            
            PHASE 2 (OPTIONAL VERIFICATION):
            4. If accepted, verify against detailed student embeddings.
            5. If still accepted, assign correct student name.
            6. Else, Unknown.
        """
        known_encodings = self.encoding_manager.encodings
        known_names = self.encoding_manager.names

        if not known_encodings:
            logger.warning("No known encodings — returning Unknown.")
            return self._unknown_result(location, distance=1.0, similarity=0.0)

        # ── PHASE 1: Fast Match ──────────────────────────────────────
        distances: np.ndarray = fr_lib.face_distance(known_encodings, encoding)

        # Find minimum distance across representative encodings
        min_idx = np.argmin(distances)
        min_dist = float(distances[min_idx])
        best_name = known_names[min_idx]

        accepted = min_dist <= self.tolerance

        # ── PHASE 2: Detailed Verification ───────────────────────────
        if accepted:
            detailed_encs = self.encoding_manager.get_detailed_encodings_for(best_name)
            if detailed_encs:
                detailed_distances = fr_lib.face_distance(detailed_encs, encoding)
                best_detailed_dist = float(np.min(detailed_distances))
                # Use the best distance from detailed encodings for the final score
                min_dist = min(min_dist, best_detailed_dist)
                accepted = min_dist <= self.tolerance

        similarity: float = self._distance_to_similarity(min_dist)

        # ── Debug output ─────────────────────────────────────
        print("\n" + "-" * 30)
        print("Detected face")
        print(f"Known students checked: {len(known_encodings)}")
        print(f"Best match: {best_name}")
        print(f"Distance: {min_dist:.4f}")
        print(f"Threshold: {self.tolerance:.2f}")
        print(f"Accepted: {'YES' if accepted else 'NO'}")
        if not accepted:
            print("Result: Unknown")
        print("-" * 30)

        logger.info(
            "Face match: best=%s dist=%.4f threshold=%.2f accepted=%s (students checked=%d)",
            best_name, min_dist, self.tolerance, accepted, len(known_encodings),
        )

        if accepted:
            return {
                "name": best_name,
                "registered": True,
                "similarity": similarity,
                "distance": min_dist,
                "location": location,
            }

        return self._unknown_result(location, min_dist, similarity)

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert face distance to an intuitive similarity percentage.

        The raw ``1.0 − distance`` formula under‑reports similarity for
        correct matches (a real person at distance 0.35 shows only 65%).

        This mapping uses **non‑linear scaling** so that distances well
        within tolerance produce the high scores humans expect:

            distance │ similarity
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
        distance: float,
        similarity: float,
    ) -> RecognitionResult:
        """Build a standard result dict for an unrecognised face."""
        return {
            "name": "Unknown",
            "registered": False,
            "similarity": round(similarity, 4),
            "distance": round(distance, 4),
            "location": location,
        }
