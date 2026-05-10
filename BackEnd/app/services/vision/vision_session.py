"""
Vision Session — HTTP-Friendly Pipeline Wrapper
=================================================
Wraps the full vision pipeline (encoding manager, face detector,
recognizer, tracker, emotion tracker, attendance service) into a
single process-level singleton suitable for stateless HTTP requests.

Each ``recognize_frame`` call:
    1. Decodes a BGR frame.
    2. Detects faces, recognises them, tracks identities, classifies emotion.
    3. Marks attendance for the most stable / highest-similarity face.
    4. Returns lightweight structured results for the frontend.

The same module also exposes:

    • ``start_camera`` / ``stop_camera`` — open / close a server-side
      ``cv2.VideoCapture`` so an MJPEG stream can be served.
    • ``mjpeg_generator`` — generator yielding multipart JPEG frames.
    • ``rebuild_encodings`` — rebuild the face-encoding cache.
    • ``reset_attendance`` — clear in-memory attendance state.

This module is intentionally side-effect free at import time so the
test suite can import it without opening a webcam.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.services.vision.attendance_service import AttendanceService
from app.services.vision.emotion_tracker import EmotionTracker
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognizer import FaceRecognizer
from app.services.vision.face_tracker import FaceTracker

logger = logging.getLogger(__name__)


# ── Public exception ────────────────────────────────────────────────


class VisionError(Exception):
    """Raised for unrecoverable vision pipeline errors."""


# ── Singleton session ───────────────────────────────────────────────


class VisionSession:
    """Process-level vision pipeline used by the FastAPI routes.

    Holds long-lived components (encodings, trackers, attendance) so
    state survives across HTTP calls within a single uvicorn worker.

    Args:
        enable_emotion: If ``False``, skip emotion detection entirely
                        (useful for unit tests).
    """

    def __init__(self, enable_emotion: bool = True) -> None:
        self.enable_emotion = enable_emotion
        self.encoding_manager = EncodingManager()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer(encoding_manager=self.encoding_manager)
        self.face_tracker = FaceTracker()
        self.attendance_service = AttendanceService()

        self.emotion_tracker: Optional[EmotionTracker] = None
        if enable_emotion:
            # Run the emotion detector on every recognise-frame call.
            # The frontend already throttles uploads to ~800 ms, so this
            # naturally gives ~one emotion sample per request without
            # spending CPU on extra detector runs.  Raw flicker is then
            # smoothed via the buffer's majority vote.
            self.emotion_tracker = EmotionTracker(
                emotion_interval=1,
                buffer_size=settings.EMOTION_BUFFER_SIZE,
                max_stale_frames=settings.EMOTION_MAX_STALE_FRAMES,
                min_stable_samples=settings.EMOTION_MIN_STABLE_SAMPLES,
            )

        self.frame_count: int = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_running: bool = False

        # Best-effort load on construction so the first frame is fast.
        try:
            self.encoding_manager.load_encodings()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not preload encodings: %s", exc)

    # ── Encodings ────────────────────────────────────────────────────

    def ensure_encodings(self) -> bool:
        """Make sure encodings are in memory; build them if needed."""
        if self.encoding_manager.is_loaded:
            return True
        if self.encoding_manager.load_encodings():
            return True
        try:
            self.encoding_manager.build_encodings()
        except FileNotFoundError as exc:
            logger.error("Encoding build failed: %s", exc)
            return False
        return self.encoding_manager.is_loaded

    def rebuild_encodings(self) -> Dict[str, Any]:
        """Force a full rebuild of the face-encoding cache."""
        summary = self.encoding_manager.build_encodings()
        # Reload in-memory cache so subsequent recognitions see new students
        self.encoding_manager.load_encodings()
        # Reset trackers so newly added faces get fresh stability counters
        self.face_tracker.reset()
        if self.emotion_tracker is not None:
            self.emotion_tracker.reset()
        return summary

    # ── Frame processing ─────────────────────────────────────────────

    def recognize_frame(
        self,
        frame: np.ndarray,
        mark_attendance: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run the full pipeline on one BGR frame and return per-face results.

        Each result dict contains::

            {
                "track_id": int,
                "name": str,
                "registered": bool,
                "similarity": float,
                "location": (top, right, bottom, left),
                "stable": bool,
                "stable_frames": int,
                "attendance_ready": bool,
                "emotion": str,
                "emotion_confidence": float,
            }
        """
        if frame is None or frame.size == 0:
            return []

        self.ensure_encodings()
        self.frame_count += 1

        # Detect → recognise on a downscaled RGB copy (faster).
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = self.face_detector.detect_faces(rgb_small, is_rgb=True)
        raw_results = (
            self.face_recognizer.recognize_faces(rgb_small, locations, is_rgb=True)
            if locations
            else []
        )

        # Scale boxes back up to original frame coords.
        fh, fw = frame.shape[:2]
        for res in raw_results:
            t, r, b, l = res["location"]
            res["location"] = (
                max(0, min(t * 2, fh - 1)),
                max(0, min(r * 2, fw - 1)),
                max(0, min(b * 2, fh - 1)),
                max(0, min(l * 2, fw - 1)),
            )

        stable_results = self.face_tracker.update(raw_results)

        # Emotion per stable result, including a "stable" flag that
        # determines whether attendance can be safely committed.
        for result in stable_results:
            label, confidence = "Neutral", 0.0
            emotion_stable = self.emotion_tracker is None  # True if disabled
            emotion_samples = 0
            if self.emotion_tracker is not None:
                tid = result.get("track_id", -1)
                loc = result.get("location")
                if loc is not None and tid >= 0:
                    top, right, bottom, left = loc
                    top, left = max(0, top), max(0, left)
                    bottom, right = min(fh, bottom), min(fw, right)
                    if bottom > top and right > left:
                        crop = frame[top:bottom, left:right]
                        emotion = self.emotion_tracker.update(
                            track_id=tid,
                            face_crop=crop,
                            frame_count=self.frame_count,
                        )
                        label = emotion.get("label", "Neutral")
                        confidence = float(emotion.get("confidence", 0.0))
                        emotion_stable = self.emotion_tracker.is_stable(tid)
                        emotion_samples = self.emotion_tracker.sample_count(tid)
            result["emotion"] = label
            result["emotion_confidence"] = round(confidence, 4)
            result["emotion_stable"] = bool(emotion_stable)
            result["emotion_samples"] = int(emotion_samples)

        # Attendance for stable + recognised + ready faces — but ONLY
        # once emotion has also stabilised.  This delivers the project
        # requirement that attendance must not commit before an averaged
        # emotion is available (~5s of samples).
        if mark_attendance:
            for res in stable_results:
                if (
                    res.get("attendance_ready")
                    and res.get("registered")
                    and res.get("emotion_stable")
                ):
                    self.attendance_service.mark_attendance(
                        name=res["name"],
                        registered=True,
                        similarity=res.get("similarity", 0.0),
                        emotion=res.get("emotion"),
                        emotion_confidence=res.get("emotion_confidence"),
                    )

        return stable_results

    # ── Camera streaming (server-side webcam) ────────────────────────

    def start_camera(self, camera_index: int = 0) -> bool:
        """Open the server-side webcam, returning ``True`` on success."""
        with self._capture_lock:
            if self._capture is not None and self._capture.isOpened():
                return True

            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logger.error("Could not open server-side camera %d", camera_index)
                return False
            self._capture = cap
            self._stream_running = True

        self._stream_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._stream_thread.start()
        logger.info("Server-side camera started (index=%d)", camera_index)
        return True

    def stop_camera(self) -> None:
        """Stop the server-side camera and release resources."""
        self._stream_running = False
        with self._capture_lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
        self._latest_frame = None
        logger.info("Server-side camera stopped")

    def is_camera_running(self) -> bool:
        return self._stream_running and self._capture is not None

    def _capture_loop(self) -> None:
        """Background thread: keep ``_latest_frame`` filled while running."""
        while self._stream_running:
            with self._capture_lock:
                cap = self._capture
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            self._latest_frame = frame
            time.sleep(0.01)

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        """Yield multipart JPEG chunks for an MJPEG stream.

        Each yielded frame is annotated with recognition + emotion overlays.
        """
        while self._stream_running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                results = self.recognize_frame(frame.copy())
                annotated = self._draw_overlays(frame.copy(), results)
            except Exception as exc:  # noqa: BLE001
                logger.debug("MJPEG frame processing failed: %s", exc)
                annotated = frame

            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    @staticmethod
    def _draw_overlays(
        frame: np.ndarray, results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draw bounding boxes + identity + emotion labels on the frame."""
        for r in results:
            loc = r.get("location")
            if loc is None:
                continue
            top, right, bottom, left = loc
            registered = r.get("registered", False)
            stable = r.get("stable", False)
            color = (0, 200, 0) if stable and registered else (
                (0, 0, 230) if stable else (0, 165, 255)
            )
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = r.get("name", "Unknown") if registered else "Unknown"
            emotion = r.get("emotion", "")
            text = f"{label}" + (f" - {emotion}" if emotion else "")

            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                frame,
                (left, max(top - th - 8, 0)),
                (left + tw + 8, max(top, th + 8)),
                (30, 30, 30),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                text,
                (left + 4, max(top - 4, th + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
            )
        return frame

    # ── Attendance helpers ───────────────────────────────────────────

    def reset_attendance(self) -> None:
        """Clear in-memory attendance state and tracker history."""
        self.attendance_service.reset_session()
        self.face_tracker.reset()
        if self.emotion_tracker is not None:
            self.emotion_tracker.reset()

    def get_active_student(self) -> Optional[str]:
        """Return the currently-recognised student, if any."""
        return self.attendance_service.get_active_student()

    def get_summaries(self) -> List[Dict[str, Any]]:
        """Return per-student summaries for the frontend."""
        return self.attendance_service.get_student_summary()


# ── Process-level singleton accessor ────────────────────────────────

_SESSION: Optional[VisionSession] = None
_SESSION_LOCK = threading.Lock()


def get_vision_session() -> VisionSession:
    """Return the shared :class:`VisionSession` (lazy-instantiated)."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = VisionSession()
        return _SESSION


def reset_vision_session() -> None:
    """Drop the singleton (used by tests to reset state between cases)."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION.stop_camera()
        _SESSION = None


# ── Frame helpers ───────────────────────────────────────────────────


def decode_base64_frame(image_base64: str) -> np.ndarray:
    """Decode a base64-encoded JPEG/PNG into a BGR ndarray.

    Accepts both raw base64 and ``data:image/...;base64,xxxx`` form.
    """
    if not image_base64:
        raise VisionError("Empty image payload")

    payload = image_base64
    if "," in payload and payload.lstrip().lower().startswith("data:"):
        payload = payload.split(",", 1)[1]

    try:
        data = base64.b64decode(payload, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise VisionError(f"Invalid base64 payload: {exc}") from exc

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise VisionError("Could not decode image bytes (cv2.imdecode returned None)")
    return frame
