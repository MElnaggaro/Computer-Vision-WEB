"""
Emotion Tracker — Per-Face Temporal Smoothing
=============================================
Buffers recent emotion predictions for each tracked face and applies a
majority-vote smoothing to eliminate per-frame flicker.

Architecture::

    FaceTracker (stable identity)
         │
         ▼
    EmotionTracker.update(track_id, face_crop, frame_count)
         │   ├─ every N frames  →  EmotionDetector.predict(crop)  →  push to buffer
         │   └─ between frames  →  reuse cached result
         ▼
    smoothed_emotion: {"label": "Happy", "confidence": 0.87}

Design decisions
----------------
* **Per-track buffers** — each ``track_id`` maintains its own deque of
  recent predictions so that two people's emotion histories never
  pollute each other.
* **Frame throttling** — ``emotion_interval`` controls how often the
  (relatively expensive) DeepFace call is made.  Between calls the
  last result is reused, keeping FPS smooth.
* **Majority-vote smoothing** — the most frequent label in the buffer
  wins; confidence is the mean of that label's probability scores.
* **Stale track cleanup** — tracks that haven't been updated for
  ``max_stale_frames`` are automatically pruned to avoid memory leaks.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from app.services.vision.emotion_detection import EmotionDetector, EmotionResult

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class _EmotionBuffer:
    """Rolling emotion history for a single tracked face."""

    buffer: Deque[Tuple[str, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    last_result: EmotionResult = field(
        default_factory=lambda: {"label": "Neutral", "confidence": 0.0, "raw_scores": {}}
    )
    last_updated_frame: int = -1
    last_seen_frame: int = -1       # last frame this track was actively updated

    def push(self, result: EmotionResult) -> None:
        """Add a new prediction to the rolling buffer."""
        self.last_result = result
        self.buffer.append((result["label"], result["confidence"]))

    @property
    def smoothed(self) -> EmotionResult:
        """Return the majority-voted, confidence-averaged emotion result.

        Falls back to the last raw result if the buffer is too small.
        """
        if not self.buffer:
            return self.last_result

        counter: Counter[str] = Counter(label for label, _ in self.buffer)
        dominant_label, _ = counter.most_common(1)[0]

        confidences = [conf for label, conf in self.buffer if label == dominant_label]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        return {
            "label": dominant_label,
            "confidence": avg_confidence,
            "raw_scores": self.last_result.get("raw_scores", {}),
        }


# ── Main tracker ─────────────────────────────────────────────────────

class EmotionTracker:
    """Manage per-face emotion history and throttled DeepFace inference.

    Args:
        emotion_interval: Run DeepFace every *N* frames per track.
                          Higher values → better FPS, slightly lagged updates.
                          Default is 5 (roughly every 5th recognition frame).
        buffer_size:      Number of recent predictions to smooth over.
        max_stale_frames: Drop a track's buffer after this many frames without an update.
        detector:         Optional pre-constructed :class:`EmotionDetector` (for DI/testing).

    Usage::

        tracker = EmotionTracker(emotion_interval=5)
        # Inside the webcam loop, after face recognition:
        emotion = tracker.update(
            track_id=result["track_id"],
            face_crop=frame[top:bottom, left:right],
            frame_count=self.frame_count,
        )
        result["emotion"] = emotion["label"]
        result["emotion_confidence"] = emotion["confidence"]
    """

    def __init__(
        self,
        emotion_interval: int = 5,
        buffer_size: int = 10,
        max_stale_frames: int = 30,
        detector: Optional[EmotionDetector] = None,
        min_stable_samples: int = 5,
    ) -> None:
        self.emotion_interval = max(1, emotion_interval)
        self.buffer_size = max(1, buffer_size)
        self.max_stale_frames = max_stale_frames
        self.detector = detector or EmotionDetector()
        # A track must have at least this many predictions before its
        # smoothed emotion is considered "stable" (averaging window
        # required by the project spec — prevents committing the first
        # raw prediction as the final result).
        self.min_stable_samples = max(1, min_stable_samples)

        self._buffers: Dict[int, _EmotionBuffer] = {}

    # ── Public API ───────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        face_crop: np.ndarray,
        frame_count: int,
    ) -> EmotionResult:
        """Get the current (possibly smoothed/cached) emotion for a track.

        Runs the detector only every ``emotion_interval`` frames; otherwise
        returns the cached smoothed result.

        Args:
            track_id:    Unique integer ID assigned by :class:`FaceTracker`.
            face_crop:   BGR face crop (already extracted from the frame).
            frame_count: Global frame counter from the webcam loop.

        Returns:
            Smoothed :class:`EmotionResult` dict:
            ``{"label": str, "confidence": float, "raw_scores": dict}``
        """
        buf = self._get_or_create_buffer(track_id)
        buf.last_seen_frame = frame_count  # mark as active this frame

        # Decide whether to run inference this frame
        frames_since_last_inference = frame_count - buf.last_updated_frame
        should_infer = (
            buf.last_updated_frame < 0  # first time
            or frames_since_last_inference >= self.emotion_interval
        )

        if should_infer:
            result = self.detector.predict(face_crop)
            buf.push(result)
            buf.last_updated_frame = frame_count
            sample_num = len(buf.buffer)
            logger.info(
                "🧠 Emotion sample %d/%d for track %d: %s (%.0f%%)",
                sample_num, self.min_stable_samples, track_id,
                result["label"], result["confidence"] * 100,
            )

        self._prune_stale(frame_count)
        return buf.smoothed

    def reset(self) -> None:
        """Clear all tracked emotion buffers."""
        self._buffers.clear()
        logger.debug("EmotionTracker reset.")

    def get_smoothed(self, track_id: int) -> Optional[EmotionResult]:
        """Return the current smoothed result for a track, or ``None``."""
        buf = self._buffers.get(track_id)
        if buf is None:
            return None
        return buf.smoothed

    def is_stable(self, track_id: int) -> bool:
        """Return ``True`` once the track has accumulated enough samples.

        Used by the attendance gate so we never commit a face's emotion
        based on a single raw prediction — we wait for at least
        ``min_stable_samples`` smoothed observations.
        """
        buf = self._buffers.get(track_id)
        if buf is None:
            return False
        return len(buf.buffer) >= self.min_stable_samples

    def sample_count(self, track_id: int) -> int:
        """Number of emotion samples collected so far for the given track."""
        buf = self._buffers.get(track_id)
        return 0 if buf is None else len(buf.buffer)

    # ── Internals ────────────────────────────────────────────────────

    def _get_or_create_buffer(self, track_id: int) -> _EmotionBuffer:
        if track_id not in self._buffers:
            buf = _EmotionBuffer()
            buf.buffer = deque(maxlen=self.buffer_size)
            self._buffers[track_id] = buf
        return self._buffers[track_id]

    def _prune_stale(self, current_frame: int) -> None:
        """Remove buffers for tracks that haven't been updated recently."""
        stale = [
            tid for tid, buf in self._buffers.items()
            if buf.last_seen_frame >= 0
            and (current_frame - buf.last_seen_frame) > self.max_stale_frames
        ]
        for tid in stale:
            del self._buffers[tid]
            logger.debug("Pruned stale emotion buffer for track %d.", tid)
