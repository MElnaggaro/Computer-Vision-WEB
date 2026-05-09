"""
Face Detection Service
======================
Thin wrapper around ``face_recognition.face_locations`` that detects
face bounding boxes in a BGR frame (as received from OpenCV).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# Type alias: (top, right, bottom, left) — face_recognition convention
FaceLocation = Tuple[int, int, int, int]


class FaceDetector:
    """Detect face locations in video frames."""

    def __init__(self, model: Optional[str] = None) -> None:
        """
        Args:
            model: ``"hog"`` (fast / CPU) or ``"cnn"`` (accurate / GPU).
        """
        self.model = model or settings.FACE_RECOGNITION_MODEL

    def detect_faces(self, frame: np.ndarray) -> List[FaceLocation]:
        """Detect all faces in a BGR frame.

        Args:
            frame: A BGR image (np.ndarray) as returned by ``cv2.VideoCapture``.

        Returns:
            List of ``(top, right, bottom, left)`` bounding‑box tuples.
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame – skipping detection.")
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations: List[FaceLocation] = face_recognition.face_locations(
            rgb_frame, model=self.model
        )

        logger.debug("Detected %d face(s) in frame.", len(locations))
        return locations
