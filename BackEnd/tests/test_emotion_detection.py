"""
Tests — Emotion Detection Module
=================================
Unit tests for EmotionDetector and EmotionTracker.

These tests use mocking to avoid requiring a real GPU/CPU-heavy
DeepFace inference during CI.  The integration tests in
``test_integration_vision_emotion.py`` exercise the full pipeline.

Run:
    cd BackEnd
    pytest tests/test_emotion_detection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure BackEnd/ is on sys.path ────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.vision.emotion_detection import EmotionDetector, _LABEL_MAP
from app.services.vision.emotion_tracker import EmotionTracker, _EmotionBuffer


@pytest.fixture(autouse=True)
def reset_emotion_detector():
    """Reset the EmotionDetector singleton between tests to prevent state leakage."""
    EmotionDetector._instance = None
    yield
    EmotionDetector._instance = None


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _make_face_crop(h: int = 80, w: int = 80) -> np.ndarray:
    """Tiny fake BGR face crop."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_fer_response(dominant: str = "happy", score: float = 0.85) -> list:
    """Simulate a FER.detect_emotions() return value."""
    emotions = {k: 0.01 for k in _LABEL_MAP.keys()}
    emotions[dominant] = score
    return [{"emotions": emotions}]


# ══════════════════════════════════════════════════════════════════════
# EmotionDetector tests
# ══════════════════════════════════════════════════════════════════════

class TestEmotionDetector:
    """Tests for the stateless EmotionDetector predictor."""

    def test_predict_happy_face(self) -> None:
        """Should correctly map 'happy' → 'Happy' with high confidence."""
        detector = EmotionDetector()
        mock_fer = MagicMock()
        mock_fer.detect_emotions.return_value = _make_fer_response("happy", 0.92)
        detector._model = mock_fer

        result = detector.predict(_make_face_crop())

        assert result["label"] == "Happy"
        assert result["confidence"] > 0.0
        assert isinstance(result["raw_scores"], dict)

    def test_predict_fear_maps_to_anxious(self) -> None:
        """'fear' raw label should map to classroom-friendly 'Anxious'."""
        detector = EmotionDetector()
        mock_fer = MagicMock()
        mock_fer.detect_emotions.return_value = _make_fer_response("fear", 0.75)
        detector._model = mock_fer

        result = detector.predict(_make_face_crop())

        assert result["label"] == "Anxious"

    def test_predict_disgust_maps_to_uncomfortable(self) -> None:
        """'disgust' should map to 'Uncomfortable'."""
        detector = EmotionDetector()
        mock_fer = MagicMock()
        mock_fer.detect_emotions.return_value = _make_fer_response("disgust", 0.60)
        detector._model = mock_fer

        result = detector.predict(_make_face_crop())

        assert result["label"] == "Uncomfortable"

    def test_predict_returns_fallback_on_tiny_crop(self) -> None:
        """Crops smaller than min_face_px should return a neutral fallback."""
        detector = EmotionDetector(min_face_px=50)
        result = detector.predict(_make_face_crop(h=20, w=20))

        assert result["label"] == "Neutral"
        assert result["confidence"] == 0.0

    def test_predict_returns_fallback_on_empty_array(self) -> None:
        """An empty array should return a neutral fallback without crashing."""
        detector = EmotionDetector()
        result = detector.predict(np.array([]))

        assert result["label"] == "Neutral"

    def test_predict_returns_fallback_on_deepface_error(self) -> None:
        """If DeepFace raises, should return a neutral fallback."""
        detector = EmotionDetector()
        mock_fer = MagicMock()
        mock_fer.detect_emotions.side_effect = RuntimeError("Model error")
        detector._model = mock_fer

        result = detector.predict(_make_face_crop())

        assert result["label"] == "Neutral"

    def test_normalize_label_all_known_keys(self) -> None:
        """All FER label keys should map to non-empty strings."""
        for raw_label in _LABEL_MAP:
            mapped = EmotionDetector.normalize_label(raw_label)
            assert isinstance(mapped, str) and len(mapped) > 0

    def test_normalize_label_unknown_key(self) -> None:
        """Unknown raw labels should be capitalized and returned as-is."""
        result = EmotionDetector.normalize_label("confused")
        assert result == "Confused"

    def test_predict_list_response_uses_first_element(self) -> None:
        """DeepFace sometimes returns a list; should use the first element."""
        detector = EmotionDetector()
        mock_fer = MagicMock()
        # List with two faces — only first should be used
        mock_fer.detect_emotions.return_value = [
            {"emotions": {"sad": 0.70, "happy": 0.30}},
            {"emotions": {"happy": 0.90}},
        ]
        detector._model = mock_fer

        result = detector.predict(_make_face_crop())

        assert result["label"] == "Sad"


# ══════════════════════════════════════════════════════════════════════
# EmotionTracker tests
# ══════════════════════════════════════════════════════════════════════

class TestEmotionBuffer:
    """Tests for the _EmotionBuffer dataclass."""

    def test_smoothed_returns_majority_label(self) -> None:
        """Majority label should win the vote."""
        buf = _EmotionBuffer()
        buf.buffer = deque(maxlen=10)
        buf.buffer.extend([
            ("Happy", 0.90),
            ("Happy", 0.85),
            ("Neutral", 0.60),
            ("Happy", 0.88),
        ])
        result = buf.smoothed
        assert result["label"] == "Happy"

    def test_smoothed_confidence_is_average(self) -> None:
        """Smoothed confidence should average all wins for the majority label."""
        buf = _EmotionBuffer()
        buf.buffer = deque(maxlen=10)
        buf.buffer.extend([
            ("Sad", 0.80),
            ("Sad", 0.60),
            ("Happy", 0.95),
        ])
        result = buf.smoothed
        assert result["label"] == "Sad"
        assert abs(result["confidence"] - 0.70) < 0.01  # avg of 0.80, 0.60

    def test_smoothed_empty_buffer_returns_last_result(self) -> None:
        """When buffer is empty, falls back to last_result."""
        buf = _EmotionBuffer()
        buf.last_result = {"label": "Angry", "confidence": 0.5, "raw_scores": {}}
        result = buf.smoothed
        assert result["label"] == "Angry"

    def test_push_updates_last_result(self) -> None:
        """push() should update last_result."""
        buf = _EmotionBuffer()
        buf.buffer = deque(maxlen=10)
        new_result = {"label": "Surprised", "confidence": 0.77, "raw_scores": {}}
        buf.push(new_result)
        assert buf.last_result["label"] == "Surprised"


class TestEmotionTracker:
    """Tests for the EmotionTracker per-face throttling and smoothing."""

    def _make_tracker(self, interval: int = 1) -> EmotionTracker:
        """Return a tracker wired to a mocked EmotionDetector."""
        mock_detector = MagicMock(spec=EmotionDetector)
        mock_detector.predict.return_value = {
            "label": "Happy",
            "confidence": 0.90,
            "raw_scores": {"happy": 0.90},
        }
        return EmotionTracker(
            emotion_interval=interval,
            buffer_size=5,
            max_stale_frames=10,
            detector=mock_detector,
        )

    def test_first_call_runs_inference(self) -> None:
        """Inference should always run on the first call for a new track."""
        tracker = self._make_tracker(interval=5)
        result = tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=1)
        tracker.detector.predict.assert_called_once()  # type: ignore[attr-defined]
        assert result["label"] == "Happy"

    def test_throttle_skips_inference_between_intervals(self) -> None:
        """Between intervals, detector.predict should NOT be called."""
        tracker = self._make_tracker(interval=5)
        tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=1)
        call_count_after_first = tracker.detector.predict.call_count  # type: ignore[attr-defined]

        # Frames 2–5: should NOT trigger inference (interval=5)
        for fc in range(2, 5):
            tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=fc)

        assert tracker.detector.predict.call_count == call_count_after_first  # type: ignore[attr-defined]

    def test_throttle_runs_inference_at_interval(self) -> None:
        """At frame N + interval, inference should run again."""
        tracker = self._make_tracker(interval=5)
        tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=1)
        tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=6)
        assert tracker.detector.predict.call_count == 2  # type: ignore[attr-defined]

    def test_separate_tracks_get_separate_buffers(self) -> None:
        """Each track_id should have an independent emotion buffer."""
        tracker = self._make_tracker(interval=1)
        tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=1)
        tracker.update(track_id=1, face_crop=_make_face_crop(), frame_count=1)
        assert 0 in tracker._buffers
        assert 1 in tracker._buffers

    def test_stale_track_is_pruned(self) -> None:
        """A track not updated for max_stale_frames should be pruned.

        Track 0 is seeded at frame 1 (last_seen_frame=1).
        max_stale_frames=3, so pruning fires when current_frame > 1+3=4.
        Track 1 is updated at frames 2–10; at frame 5 the pruner will
        see current_frame - track0.last_seen_frame = 5-1 = 4 > 3 → pruned.
        """
        tracker = EmotionTracker(
            emotion_interval=1,
            buffer_size=5,
            max_stale_frames=3,
            detector=MagicMock(spec=EmotionDetector),
        )
        tracker.detector.predict.return_value = {"label": "Neutral", "confidence": 0.5, "raw_scores": {}}  # type: ignore[attr-defined]

        # Seed track 0 at frame 1
        tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=1)
        assert 0 in tracker._buffers

        # Update only track 1 from frame 2 onward — track 0 will be pruned
        # when current_frame - 1 > 3, i.e. at frame_count >= 5
        for fc in range(2, 15):
            tracker.update(track_id=1, face_crop=_make_face_crop(), frame_count=fc)

        assert 0 not in tracker._buffers  # must have been pruned

    def test_reset_clears_all_buffers(self) -> None:
        """reset() should remove all tracked emotion buffers."""
        tracker = self._make_tracker(interval=1)
        tracker.update(track_id=0, face_crop=_make_face_crop(), frame_count=1)
        tracker.update(track_id=1, face_crop=_make_face_crop(), frame_count=1)
        tracker.reset()
        assert len(tracker._buffers) == 0

    def test_smoothing_majority_vote(self) -> None:
        """After multiple inferences, majority label should dominate."""
        mock_detector = MagicMock(spec=EmotionDetector)
        responses = [
            {"label": "Happy", "confidence": 0.90, "raw_scores": {}},
            {"label": "Happy", "confidence": 0.85, "raw_scores": {}},
            {"label": "Neutral", "confidence": 0.55, "raw_scores": {}},
            {"label": "Happy", "confidence": 0.88, "raw_scores": {}},
            {"label": "Happy", "confidence": 0.92, "raw_scores": {}},
        ]
        mock_detector.predict.side_effect = responses

        tracker = EmotionTracker(
            emotion_interval=1,
            buffer_size=5,
            max_stale_frames=30,
            detector=mock_detector,
        )

        final_result = None
        for fc in range(1, 6):
            final_result = tracker.update(
                track_id=0, face_crop=_make_face_crop(), frame_count=fc
            )

        assert final_result is not None
        assert final_result["label"] == "Happy"

    def test_get_smoothed_returns_none_for_unknown_track(self) -> None:
        """get_smoothed() should return None for a track_id not yet seen."""
        tracker = self._make_tracker()
        assert tracker.get_smoothed(track_id=999) is None
