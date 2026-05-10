"""
Emotion Detection Service
=========================
Detects the dominant emotion using the `fer` library.
Model is loaded once at startup to ensure stable performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Classroom-friendly label map ─────────────────────────────────────
_LABEL_MAP: Dict[str, str] = {
    "happy":    "Happy",
    "sad":      "Sad",
    "neutral":  "Neutral",
    "angry":    "Angry",
    "fear":     "Anxious",
    "surprise": "Surprised",
    "disgust":  "Uncomfortable",
}

_MIN_FACE_PX = 40
EmotionResult = Dict[str, Any]

class EmotionDetector:
    """Predict the dominant emotion from a BGR face crop using FER.
    
    Initialized once to avoid repeated model loading and memory issues.
    """
    _instance: Optional['EmotionDetector'] = None
    _model: Optional[Any] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmotionDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, min_face_px: int = _MIN_FACE_PX) -> None:
        if not hasattr(self, 'initialized'):
            self.min_face_px = min_face_px
            try:
                from fer import FER
                # mtcnn=False uses Haar Cascades for face detection (lightweight)
                # We already cropped the face, so we just want the emotion.
                self._model = FER(mtcnn=False)
                logger.info("FER Emotion model loaded successfully.")
            except ImportError as exc:
                logger.error("fer library is not installed: %s", exc)
                self._model = None
            self.initialized = True

    def predict(self, face_bgr: np.ndarray) -> EmotionResult:
        if face_bgr is None or face_bgr.size == 0:
            return self._fallback("Empty face crop")

        h, w = face_bgr.shape[:2]
        if h < self.min_face_px or w < self.min_face_px:
            return self._fallback(f"Face crop too small ({w}x{h} px)")

        if self._model is None:
            return self._fallback("FER model not loaded")

        try:
            # FER expects RGB
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            
            # Since we pass an already cropped face, FER will still try to find a face.
            # If we just want it to predict on the whole image as a face crop:
            # FER analyze returns a list of faces and their emotions.
            emotions = self._model.detect_emotions(face_rgb)
            
            if not emotions:
                return self._fallback("No emotion detected in crop")
            
            # Take the first face found in the crop
            emotion_data = emotions[0]["emotions"]
            
            # Find the dominant emotion
            dominant_raw = max(emotion_data, key=emotion_data.get)
            confidence = emotion_data[dominant_raw]
            
            raw_scores = {k: float(v) for k, v in emotion_data.items()}
            label = _LABEL_MAP.get(dominant_raw.lower(), dominant_raw.capitalize())

            return {
                "label": label,
                "confidence": round(float(confidence), 4),
                "raw_scores": raw_scores,
            }

        except Exception as exc:
            logger.debug("Emotion prediction failed: %s", exc)
            return self._fallback(str(exc))

    @staticmethod
    def _fallback(reason: str = "") -> EmotionResult:
        if reason:
            logger.debug("Emotion fallback: %s", reason)
        return {
            "label": "Neutral",
            "confidence": 0.0,
            "raw_scores": {},
        }

    @staticmethod
    def normalize_label(raw_label: str) -> str:
        return _LABEL_MAP.get(raw_label.lower(), raw_label.capitalize())
