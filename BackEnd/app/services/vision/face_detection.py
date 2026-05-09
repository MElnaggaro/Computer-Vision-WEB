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
import face_recognition as fr_lib
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

    def detect_faces(self, frame: np.ndarray, is_rgb: bool = False) -> List[FaceLocation]:
        """Detect all faces in a frame.

        Args:
            frame: A BGR or RGB image (np.ndarray).
            is_rgb: True if the frame is already in RGB format.

        Returns:
            List of ``(top, right, bottom, left)`` bounding‑box tuples.
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame – skipping detection.")
            return []

        if not is_rgb:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
            
        locations: List[FaceLocation] = fr_lib.face_locations(
            rgb_frame, model=self.model
        )

        logger.debug("Detected %d face(s) in frame.", len(locations))
        return locations
