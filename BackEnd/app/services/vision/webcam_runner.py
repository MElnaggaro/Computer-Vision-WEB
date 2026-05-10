"""
Webcam Runner — Standalone Live Camera Test
============================================
A fully self‑contained script that opens the webcam, runs the full
face‑recognition + emotion‑detection + attendance pipeline in real time,
and draws annotated bounding boxes on each frame.

Usage (from the BackEnd/ directory):
    python -m app.services.vision.webcam_runner
    python app/services/vision/webcam_runner.py

Controls:
    Q  — quit
    R  — reset attendance session (re‑mark everyone)
    B  — rebuild encodings from data/students_faces/

Design notes:
    • This script is intentionally importable so that ``test_live_camera.py``
      can reuse the ``ClassroomCamera`` class in a non‑interactive way.
    • All services are instantiated locally (no FastAPI dependency).
    • Emotion detection runs on a throttled schedule (every N recognition
      frames) and is smoothed via :class:`EmotionTracker` to avoid flicker.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Ensure BackEnd/ is on sys.path when run directly ─────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import settings
from app.services.vision.attendance_service import AttendanceService
from app.services.vision.emotion_tracker import EmotionTracker
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognizer import FaceRecognizer, RecognitionResult
from app.services.vision.face_tracker import FaceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ── Drawing constants ────────────────────────────────────────────────
_COLOR_KNOWN     = (0, 200, 0)       # green
_COLOR_UNKNOWN   = (0, 0, 230)       # red
_COLOR_UNSTABLE  = (0, 165, 255)     # orange (tracking, not yet stable)
_COLOR_TEXT_BG   = (30, 30, 30)      # dark overlay
_COLOR_EMOTION   = (255, 220, 60)    # yellow for emotion label
_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE      = 0.6
_THICKNESS       = 2


class ClassroomCamera:
    """Encapsulates the live webcam → recognition → emotion → attendance loop.

    Can be used interactively (``run()``) or programmatically (single‑frame
    processing via ``process_frame()``).

    Args:
        camera_index:       OpenCV device index (default 0).
        encoding_manager:   Pre-built encoding manager (for dependency injection).
        attendance_service: Pre-built attendance service (for dependency injection).
        emotion_tracker:    Pre-built emotion tracker (for dependency injection).
                            Pass ``None`` to disable emotion detection entirely.
        enable_emotion:     Master switch for emotion detection (default ``True``).
    """

    def __init__(
        self,
        camera_index: int = 0,
        encoding_manager: Optional[EncodingManager] = None,
        attendance_service: Optional[AttendanceService] = None,
        emotion_tracker: Optional[EmotionTracker] = None,
        enable_emotion: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.enable_emotion = enable_emotion

        # ── Services ─────────────────────────────────────────────────
        self.encoding_manager = encoding_manager or EncodingManager()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer(
            encoding_manager=self.encoding_manager,
        )
        self.face_tracker = FaceTracker()
        self.attendance_service = attendance_service or AttendanceService()

        # ── Emotion tracker ──────────────────────────────────────────
        if enable_emotion:
            self.emotion_tracker: Optional[EmotionTracker] = (
                emotion_tracker
                or EmotionTracker(
                    emotion_interval=settings.EMOTION_DETECTION_INTERVAL,
                    buffer_size=settings.EMOTION_BUFFER_SIZE,
                    max_stale_frames=settings.EMOTION_MAX_STALE_FRAMES,
                )
            )
        else:
            self.emotion_tracker = None

        # ── Optimisation variables ───────────────────────────────────
        self.frame_count = 0
        self.frame_skip = 3
        self.trackers: List[Tuple[Any, Dict[str, Any]]] = []

    def _create_tracker(self) -> Optional[Any]:
        """Create an OpenCV object tracker, falling back to MIL if KCF unavailable."""
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            try:
                return cv2.TrackerMIL_create()
            except AttributeError:
                return None

    # ── Public API ───────────────────────────────────────────────────

    def ensure_encodings(self) -> bool:
        """Load or build encodings so recognition is ready.

        Returns:
            ``True`` if encodings are available after this call.
        """
        if self.encoding_manager.is_loaded:
            return True

        if self.encoding_manager.load_encodings():
            return True

        logger.info("No cached encodings found — building from student images …")
        try:
            summary = self.encoding_manager.build_encodings()
            logger.info("Built encodings: %s", summary)
            return self.encoding_manager.is_loaded
        except FileNotFoundError:
            logger.error(
                "Cannot build encodings: %s does not exist.",
                settings.STUDENTS_FACES_DIR,
            )
            return False

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Run detection → recognition → emotion → tracking → attendance on a single frame.

        Args:
            frame: BGR image from ``cv2.VideoCapture``.

        Returns:
            ``(annotated_frame, stable_results)`` where ``annotated_frame`` has
            bounding boxes, identity labels, and emotion labels drawn on it.
            Each result dict contains:
            - ``name``, ``registered``, ``similarity``, ``location``, ``stable``
            - ``emotion`` (str)  — classroom-friendly label, e.g. "Happy"
            - ``emotion_confidence`` (float) — 0–1
        """
        self.frame_count += 1

        # Preprocessing: Apply CLAHE for lighting robustness
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # 1. Resize frame for faster processing
        small_frame = cv2.resize(enhanced_frame, (0, 0), fx=0.25, fy=0.25)
        # 2. Convert to RGB only once
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 3. Process every Nth frame
        if self.frame_count % self.frame_skip == 1 or self.frame_count == 1:
            locations = self.face_detector.detect_faces(rgb_small_frame, is_rgb=True)
            raw_results = (
                self.face_recognizer.recognize_faces(rgb_small_frame, locations, is_rgb=True)
                if locations
                else []
            )

            # Scale locations back up to original frame size
            for res in raw_results:
                t, r, bv, left = res["location"]
                res["location"] = (t * 4, r * 4, bv * 4, left * 4)

            # Re-initialize object trackers
            self.trackers = []
            for res in raw_results:
                tracker = self._create_tracker()
                if tracker is not None:
                    t, r, bv, left = res["location"]
                    bbox = (left, t, r - left, bv - t)
                    tracker.init(enhanced_frame, bbox)
                    self.trackers.append((tracker, dict(res)))

            tracking_results = raw_results
        else:
            # 4. Use Object Tracking for intermediate frames
            tracking_results = []
            for tracker, res in self.trackers:
                success, bbox = tracker.update(enhanced_frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    res["location"] = (y, x + w, y + h, x)
                tracking_results.append(res)

        # 5. Stabilize identities using face tracker
        stable_results = self.face_tracker.update(tracking_results)

        # 6. Emotion detection (per tracked face)
        for result in stable_results:
            emotion_label = "Neutral"
            emotion_confidence = 0.0

            if self.emotion_tracker is not None:
                track_id = result.get("track_id", -1)
                location = result.get("location")
                if location is not None and track_id >= 0:
                    top, right, bottom, left = location
                    # Guard against negative/zero-size crops
                    top = max(0, top)
                    left = max(0, left)
                    bottom = min(frame.shape[0], bottom)
                    right = min(frame.shape[1], right)
                    if bottom > top and right > left:
                        face_crop = frame[top:bottom, left:right]
                        emotion_result = self.emotion_tracker.update(
                            track_id=track_id,
                            face_crop=face_crop,
                            frame_count=self.frame_count,
                        )
                        emotion_label = emotion_result.get("label", "Neutral")
                        emotion_confidence = emotion_result.get("confidence", 0.0)

            result["emotion"] = emotion_label
            result["emotion_confidence"] = emotion_confidence

        # 7. Mark attendance for each recognised stable face
        for result in stable_results:
            if result.get("attendance_ready") and result.get("registered"):
                self.attendance_service.mark_attendance(
                    name=result["name"],
                    registered=result["registered"],
                    similarity=result.get("similarity", 0.0),
                    emotion=result.get("emotion"),
                    emotion_confidence=result.get("emotion_confidence"),
                )

        # 8. Save log only when attendance changes
        if getattr(self.attendance_service, "_has_unsaved_changes", False):
            self.attendance_service.save_log()

        annotated = self._draw_annotations(frame.copy(), stable_results)
        return annotated, stable_results

    def run(self) -> None:
        """Open the webcam and run the interactive attendance loop.

        Press **Q** to quit, **R** to reset session, **B** to rebuild encodings.
        """
        # Ensure encodings are loaded exactly once at startup
        if not self.ensure_encodings():
            logger.warning(
                "Starting camera WITHOUT encodings — all faces will be Unknown."
            )

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(
                "Could not open webcam at index %d. "
                "Check that a camera is connected and not in use.",
                self.camera_index,
            )
            return

        emotion_status = (
            f"every {settings.EMOTION_DETECTION_INTERVAL} frames"
            if self.emotion_tracker is not None
            else "DISABLED"
        )
        logger.info(
            "Camera opened (emotion detection: %s). "
            "Press Q to quit | R to reset session | B to rebuild encodings",
            emotion_status,
        )

        fps_time = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame — retrying …")
                    continue

                annotated, results = self.process_frame(frame)

                # ── FPS overlay ──────────────────────────────────────
                frame_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_time = time.time()
                else:
                    fps = frame_count / max(elapsed, 0.001)

                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 25),
                    _FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # ── Attendance status bar ────────────────────────────
                marked_count = len(self.attendance_service.marked_students)
                status_text = f"Marked: {marked_count} student(s)"
                cv2.putText(
                    annotated,
                    status_text,
                    (10, annotated.shape[0] - 15),
                    _FONT,
                    0.6,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Smart Classroom — Face Recognition + Emotion", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    logger.info("Quit requested.")
                    break
                elif key == ord("r") or key == ord("R"):
                    self.attendance_service.reset_session()
                    self.face_tracker.reset()
                    if self.emotion_tracker is not None:
                        self.emotion_tracker.reset()
                    logger.info("Session and tracker reset — you can re‑mark everyone.")
                elif key == ord("b") or key == ord("B"):
                    logger.info("Rebuilding encodings …")
                    try:
                        summary = self.encoding_manager.build_encodings()
                        logger.info("Rebuild complete: %s", summary)
                    except Exception as exc:
                        logger.error("Rebuild failed: %s", exc)

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            # Always save the attendance log on exit
            self.attendance_service.save_log()
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released. Attendance log saved.")

    # ── Drawing helpers ──────────────────────────────────────────────

    @staticmethod
    def _draw_annotations(
        frame: np.ndarray,
        results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw bounding boxes, identity labels, and emotion labels on the frame.

        Stable Known → green, Stable Unknown → red, Unstable → orange.
        Emotion shown as a secondary line below the identity label.
        """
        for result in results:
            location = result.get("location")
            if location is None:
                continue

            top, right, bottom, left = location
            name = result.get("name", "Unknown")
            similarity = result.get("similarity", 0.0)
            registered = result.get("registered", False)
            stable = result.get("stable", False)
            emotion = result.get("emotion", "")
            emotion_confidence = result.get("emotion_confidence", 0.0)

            if stable:
                color = _COLOR_KNOWN if registered else _COLOR_UNKNOWN
                identity_label = name if registered else "Unknown"
            else:
                color = _COLOR_UNSTABLE
                identity_label = f"Detecting... ({similarity:.0%})"

            # Bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, _THICKNESS)

            # ── Identity label (above box) ───────────────────────────
            (id_w, id_h), id_base = cv2.getTextSize(
                identity_label, _FONT, _FONT_SCALE, 1
            )
            label_y = max(top - 10, id_h + 10)

            cv2.rectangle(
                frame,
                (left, label_y - id_h - 6),
                (left + id_w + 8, label_y + id_base),
                _COLOR_TEXT_BG,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                identity_label,
                (left + 4, label_y - 2),
                _FONT,
                _FONT_SCALE,
                color,
                1,
            )

            # ── Emotion label (below identity, inside/below box) ─────
            if stable and emotion:
                conf_pct = int(emotion_confidence * 100)
                emotion_label = f"{emotion} ({conf_pct}%)" if conf_pct > 0 else emotion

                (em_w, em_h), em_base = cv2.getTextSize(
                    emotion_label, _FONT, _FONT_SCALE * 0.9, 1
                )
                em_y = label_y + id_h + 10

                cv2.rectangle(
                    frame,
                    (left, em_y - em_h - 4),
                    (left + em_w + 8, em_y + em_base),
                    _COLOR_TEXT_BG,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    emotion_label,
                    (left + 4, em_y - 2),
                    _FONT,
                    _FONT_SCALE * 0.9,
                    _COLOR_EMOTION,
                    1,
                )

        return frame


# ── CLI entry point ──────────────────────────────────────────────────

def main() -> None:
    """Parse optional CLI args and run the camera loop."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Classroom — Live Webcam Face Recognition + Emotion Detection",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force‑rebuild encodings before starting",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Disable emotion detection (higher FPS)",
    )
    args = parser.parse_args()

    runner = ClassroomCamera(
        camera_index=args.camera,
        enable_emotion=not args.no_emotion,
    )

    if args.rebuild:
        logger.info("Force‑rebuilding encodings …")
        try:
            summary = runner.encoding_manager.build_encodings()
            logger.info("Build complete: %s", summary)
        except Exception as exc:
            logger.error("Build failed: %s", exc)

    runner.run()


if __name__ == "__main__":
    main()
