"""
Webcam Runner — Standalone Live Camera Test
============================================
A fully self‑contained script that opens the webcam, runs the full
face‑recognition + attendance pipeline in real time, and draws annotated
bounding boxes on each frame.

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
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Ensure BackEnd/ is on sys.path when run directly ─────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import settings
from app.services.vision.attendance_service import AttendanceService
from app.services.vision.encoding_manager import EncodingManager
from app.services.vision.face_detection import FaceDetector
from app.services.vision.face_recognizer import FaceRecognizer, RecognitionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ── Drawing constants ────────────────────────────────────────────────
_COLOR_KNOWN = (0, 200, 0)        # green
_COLOR_UNKNOWN = (0, 0, 230)      # red
_COLOR_TEXT_BG = (30, 30, 30)     # dark overlay
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_THICKNESS = 2


class ClassroomCamera:
    """Encapsulates the live webcam → recognition → attendance loop.

    Can be used interactively (``run()``) or programmatically (single‑frame
    processing via ``process_frame()``).
    """

    def __init__(
        self,
        camera_index: int = 0,
        encoding_manager: Optional[EncodingManager] = None,
        attendance_service: Optional[AttendanceService] = None,
    ) -> None:
        self.camera_index = camera_index

        # ── Services ─────────────────────────────────────────────────
        self.encoding_manager = encoding_manager or EncodingManager()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer(
            encoding_manager=self.encoding_manager,
        )
        self.attendance_service = attendance_service or AttendanceService()

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
    ) -> Tuple[np.ndarray, List[RecognitionResult]]:
        """Run detection → recognition → attendance on a single frame.

        Args:
            frame: BGR image from ``cv2.VideoCapture``.

        Returns:
            ``(annotated_frame, results)`` where ``annotated_frame`` has
            bounding boxes and labels drawn on it.
        """
        locations = self.face_detector.detect_faces(frame)
        results = (
            self.face_recognizer.recognize_faces(frame, locations)
            if locations
            else []
        )

        # Mark attendance for each recognised face
        for result in results:
            self.attendance_service.mark_attendance(
                name=result["name"],
                known=result["known"],
                confidence=result["confidence"],
            )

        annotated = self._draw_annotations(frame.copy(), results)
        return annotated, results

    def run(self) -> None:
        """Open the webcam and run the interactive attendance loop.

        Press **Q** to quit, **R** to reset session, **B** to rebuild encodings.
        """
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

        logger.info(
            "Camera opened. Press Q to quit | R to reset session | B to rebuild encodings"
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

                cv2.imshow("Smart Classroom — Face Recognition", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    logger.info("Quit requested.")
                    break
                elif key == ord("r") or key == ord("R"):
                    self.attendance_service.reset_session()
                    logger.info("Session reset — you can re‑mark everyone.")
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
        results: List[RecognitionResult],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the frame.

        Known students → green box, Unknown → red box.
        """
        for result in results:
            location = result.get("location")
            if location is None:
                continue

            top, right, bottom, left = location
            name = result["name"]
            confidence = result["confidence"]
            known = result["known"]

            color = _COLOR_KNOWN if known else _COLOR_UNKNOWN

            # Bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, _THICKNESS)

            # Label background
            label = f"{name} ({confidence:.0%})"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, _FONT, _FONT_SCALE, 1
            )
            label_y = max(top - 10, text_h + 10)
            cv2.rectangle(
                frame,
                (left, label_y - text_h - 6),
                (left + text_w + 8, label_y + baseline),
                _COLOR_TEXT_BG,
                cv2.FILLED,
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (left + 4, label_y - 2),
                _FONT,
                _FONT_SCALE,
                color,
                1,
            )

        return frame


# ── CLI entry point ──────────────────────────────────────────────────

def main() -> None:
    """Parse optional CLI args and run the camera loop."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Classroom — Live Webcam Face Recognition",
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
    args = parser.parse_args()

    runner = ClassroomCamera(camera_index=args.camera)

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
