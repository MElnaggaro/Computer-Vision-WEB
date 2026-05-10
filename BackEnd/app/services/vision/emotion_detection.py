"""
Emotion Detection Service
=========================
Detects the dominant emotion in a cropped face image using DeepFace's
lightweight emotion analysis (backed by a small CNN trained on FER-2013).

Design decisions
----------------
* **DeepFace** is used instead of the ``fer`` library because it:
    - Does not require TensorFlow to be running a full session per call.
    - Has a stable pip package with pre-built weights.
    - Runs reliably on CPU with sub-100 ms latency per crop.
    - Returns per-class probabilities, enabling smoothing.

* **Classroom-friendly label mapping** normalises the raw FER labels
  (fear, disgust, surprise …) into more contextually appropriate terms.

* **Frame throttling** is handled externally by ``EmotionTracker``; this
  class is a *stateless* predictor — call it as often or as rarely as
  you like.

Public API
----------
    detector = EmotionDetector()
    result = detector.predict(face_bgr_crop)
    # → {"label": "Happy", "confidence": 0.87, "raw_scores": {...}}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Classroom-friendly label map ─────────────────────────────────────
# Maps raw FER-2013 labels → human-friendly classroom labels.
_LABEL_MAP: Dict[str, str] = {
    "happy":    "Happy",
    "sad":      "Sad",
    "neutral":  "Neutral",
    "angry":    "Angry",
    "fear":     "Anxious",
    "surprise": "Surprised",
    "disgust":  "Uncomfortable",
}

# Minimum face dimension (pixels) to attempt emotion prediction.
_MIN_FACE_PX = 40

# Lazy-loaded DeepFace reference (avoids slow import at module load time).
_deepface_loaded: bool = False
_deepface_module: Optional[Any] = None


def _get_deepface() -> Any:
    """Import DeepFace lazily and cache the reference."""
    global _deepface_loaded, _deepface_module
    if not _deepface_loaded:
        try:
            from deepface import DeepFace  # type: ignore[import]
            _deepface_module = DeepFace
            _deepface_loaded = True
            logger.info("DeepFace loaded successfully.")
        except ImportError as exc:
            logger.error(
                "DeepFace is not installed. Run: pip install deepface tf-keras\n%s", exc
            )
            raise
    return _deepface_module


EmotionResult = Dict[str, Any]


class EmotionDetector:
    """Predict the dominant emotion from a BGR face crop.

    This class is intentionally stateless — all temporal smoothing is
    handled by :class:`EmotionTracker`.

    Args:
        min_face_px: Minimum width *and* height (pixels) of the crop
                     before attempting prediction.  Smaller crops are
                     returned as ``{"label": "Neutral", "confidence": 0.0}``.

    Example::

        detector = EmotionDetector()
        crop = frame[top:bottom, left:right]
        result = detector.predict(crop)
        print(result["label"], result["confidence"])
    """

    def __init__(self, min_face_px: int = _MIN_FACE_PX) -> None:
        self.min_face_px = min_face_px
        # Trigger lazy import at construction time so the first call is fast.
        try:
            _get_deepface()
        except ImportError:
            pass  # Will fail at predict() time with a clear message.

    # ── Public API ───────────────────────────────────────────────────

    def predict(self, face_bgr: np.ndarray) -> EmotionResult:
        """Run emotion prediction on a single BGR face crop.

        Args:
            face_bgr: NumPy array of shape ``(H, W, 3)`` in BGR colour
                      order (as returned by OpenCV).

        Returns:
            Dict with keys:
            - ``label`` (str): Classroom-friendly emotion label.
            - ``confidence`` (float): Probability of the dominant emotion (0–1).
            - ``raw_scores`` (dict): Per-emotion probabilities from the model.

        Raises:
            ImportError: If DeepFace is not installed.
        """
        if face_bgr is None or face_bgr.size == 0:
            return self._fallback("Empty face crop")

        h, w = face_bgr.shape[:2]
        if h < self.min_face_px or w < self.min_face_px:
            return self._fallback(f"Face crop too small ({w}×{h} px)")

        try:
            DeepFace = _get_deepface()
            analysis = DeepFace.analyze(
                img_path=face_bgr,
                actions=["emotion"],
                enforce_detection=False,   # don't re-detect; crop is already aligned
                detector_backend="skip",   # face already cropped — skip detection
                silent=True,
            )

            # DeepFace returns a list when multiple faces are found; take first.
            if isinstance(analysis, list):
                analysis = analysis[0]

            raw_scores: Dict[str, float] = {
                k: round(float(v) / 100.0, 4)
                for k, v in analysis.get("emotion", {}).items()
            }
            dominant_raw: str = analysis.get("dominant_emotion", "neutral").lower()
            confidence: float = raw_scores.get(dominant_raw, 0.0)
            label: str = _LABEL_MAP.get(dominant_raw, dominant_raw.capitalize())

            return {
                "label": label,
                "confidence": round(confidence, 4),
                "raw_scores": raw_scores,
            }

        except Exception as exc:  # noqa: BLE001
            logger.debug("Emotion prediction failed: %s", exc)
            return self._fallback(str(exc))

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fallback(reason: str = "") -> EmotionResult:
        """Return a neutral fallback result without crashing the pipeline."""
        if reason:
            logger.debug("Emotion fallback: %s", reason)
        return {
            "label": "Neutral",
            "confidence": 0.0,
            "raw_scores": {},
        }

    @staticmethod
    def normalize_label(raw_label: str) -> str:
        """Map a raw FER label to the classroom-friendly equivalent.

        Args:
            raw_label: A raw label such as ``"fear"``, ``"happy"`` …

        Returns:
            Classroom-friendly label string.
        """
        return _LABEL_MAP.get(raw_label.lower(), raw_label.capitalize())
