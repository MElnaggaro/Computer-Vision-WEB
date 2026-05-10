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
from typing import Any, Dict, Generator, List, Optional, Tuple

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

    # Identity-lock cache lifetime: how many frames a locked track may
    # coast on its cached identity before we force a fresh
    # ``face_encodings`` recompute.  Faces detected within this window
    # are served from the lock for ~free.
    IDENTITY_LOCK_TTL_FRAMES: int = 30

    # Minimum IoU required to reuse a locked identity for a detection.
    IDENTITY_LOCK_IOU: float = 0.35

    # ── Active-student session tracking ────────────────────────────
    # When a question is asked without an explicit student name, the
    # backend attributes it to the person whose face was last seen
    # within this many seconds.  Configurable via ACTIVE_STUDENT_TTL_S.
    ACTIVE_STUDENT_TTL_SECONDS: float = 8.0

    def __init__(self, enable_emotion: bool = False) -> None:
        # ── 1. Static configuration ──────────────────────────────────
        self.enable_emotion = enable_emotion

        # ── 2. Services ──────────────────────────────────────────────
        self.encoding_manager = EncodingManager()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer(encoding_manager=self.encoding_manager)
        self.face_tracker = FaceTracker()
        self.attendance_service = AttendanceService()

        # ── 3. Optional emotion tracker (off by default — performance) ─
        self.emotion_tracker: Optional[EmotionTracker] = None
        if enable_emotion:
            # Emotion runs every N frames (config: EMOTION_DETECTION_INTERVAL)
            # and ONLY for faces already marked for attendance — see
            # recognize_frame() for the performance-gating logic.
            self.emotion_tracker = EmotionTracker(
                emotion_interval=settings.EMOTION_DETECTION_INTERVAL,
                buffer_size=settings.EMOTION_BUFFER_SIZE,
                max_stale_frames=settings.EMOTION_MAX_STALE_FRAMES,
                min_stable_samples=settings.EMOTION_MIN_STABLE_SAMPLES,
            )

        # ── 4. Streaming / capture state ─────────────────────────────
        self.frame_count: int = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_running: bool = False

        # ── 4b. Active-student session ──────────────────────────────
        # Tracks the most recent person attached to the live mic/camera
        # context.  Updated by ``recognize_frame`` whenever a registered
        # face is seen, and by ``set_active_guest`` when the user clicks
        # "Continue as Guest".  ``ask-question`` consults this so a
        # question recorded while X is on screen is attributed to X.
        self._active_student: Optional[str] = None
        self._active_student_registered: bool = False
        self._active_student_seen_at: float = 0.0
        self._active_lock = threading.Lock()

        # ── 5. Fresh session: clear stale log from previous runs ─────
        # Each server start = clean session.  Prevents stale attendance
        # events from being replayed as duplicates on the frontend.
        self.attendance_service.reset_session()
        logger.info("Session cleared — fresh attendance log.")

        # ── 6. Encoding cache: load via fingerprint check ────────────
        # ``ensure_fresh`` reloads the .pkl if the dataset is unchanged
        # (milliseconds) and rebuilds it only when an image was added,
        # removed, or modified.  Replaces the old unconditional rebuild
        # which made every backend startup take 30-60 s.
        try:
            summary = self.encoding_manager.ensure_fresh()
            logger.info(
                "Encoding ready: status=%s students=%d total=%d",
                summary.get("status"),
                len(summary.get("students", {})),
                summary.get("total_encodings", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not prepare encodings: %s", exc)

    # ── Encodings ────────────────────────────────────────────────────

    def ensure_encodings(self) -> bool:
        """Make sure encodings are in memory.

        Hot path: this runs on every ``recognize_frame`` call, so it
        must be a near-zero-cost no-op once encodings are loaded.  We
        only call ``ensure_fresh`` (fingerprint check + maybe rebuild)
        when the in-memory cache is empty — i.e. on the first frame
        after server start, or after an explicit reset.
        """
        if self.encoding_manager.is_loaded:
            return True
        try:
            self.encoding_manager.ensure_fresh()
        except FileNotFoundError as exc:
            logger.error("Encoding build failed: %s", exc)
            return False
        return self.encoding_manager.is_loaded

    def rebuild_encodings(self) -> Dict[str, Any]:
        """Force a full rebuild of the face-encoding cache.

        Defensive sequence:
            1. Clear any in-memory encodings.
            2. Rebuild from the filesystem (``data/students_faces/``) only.
               ``build_encodings`` writes the new ``.pkl`` ATOMICALLY
               (tmp file + ``os.replace``), so the cache is *never*
               missing — even if the Python process is killed
               mid-rebuild.  A new fingerprint enforces correctness, so
               there is no need to ``unlink`` the previous cache: the
               filesystem-as-source-of-truth invariant comes from
               ``build_encodings`` scanning ``students_faces`` directly,
               not from preemptive deletion.
            3. Reset the per-track history so newly added students don't
               inherit any prior stability counters.
        """
        # 1 — wipe in-memory state.  We deliberately do NOT delete the
        # on-disk cache here.  Doing so was the root cause of the
        # "cache disappears between runs" bug: any registration approval
        # or aborted rebuild left the .pkl missing, forcing a 30-85 s
        # rebuild on the next startup.
        try:
            self.encoding_manager._names.clear()  # type: ignore[attr-defined]
            self.encoding_manager._encodings.clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass

        # 2 — rebuild from the live filesystem (atomic write inside).
        summary = self.encoding_manager.build_encodings()
        # Reload in-memory cache so subsequent recognitions see new students
        self.encoding_manager.load_encodings()

        # 4 — reset trackers so newly added faces get fresh stability counters
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

        Performance priorities (in order):
            1. Face detection + recognition — always runs
            2. Tracking + attendance logging — fires as soon as identity is stable
            3. Emotion — runs ONLY for faces already marked for attendance
               (skipped in the critical first-recognition path to save ~200-500 ms)

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
                "emotion_stable": bool,
                "emotion_samples": int,
                "newly_marked": bool,
            }
        """
        if frame is None or frame.size == 0:
            return []

        self.ensure_encodings()
        self.frame_count += 1
        t0 = time.perf_counter()

        # ── Stage 1: Detect on downscaled RGB ────────────────────────
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        t1 = time.perf_counter()
        small_locations = self.face_detector.detect_faces(rgb_small, is_rgb=True)
        t2 = time.perf_counter()

        # Pre-compute the upscaled boxes so we can match detections to
        # existing tracks BEFORE deciding whether to encode.
        fh, fw = frame.shape[:2]
        full_locations: List[Tuple[int, int, int, int]] = []
        for top, right, bottom, left in small_locations:
            full_locations.append(
                (
                    max(0, min(int(top * 2), fh - 1)),
                    max(0, min(int(right * 2), fw - 1)),
                    max(0, min(int(bottom * 2), fh - 1)),
                    max(0, min(int(left * 2), fw - 1)),
                )
            )

        # ── Stage 2: Identity-lock fast path ─────────────────────────
        # Per-detection decision: if the box overlaps an existing
        # track that has a *locked* identity within its TTL, reuse the
        # locked name without paying for ``face_encodings``.  This is
        # the single biggest performance lever for repeat frames.
        raw_results: List[Dict[str, Any]] = [None] * len(full_locations)  # type: ignore[list-item]
        encode_indices: List[int] = []
        used_tracks: set = set()

        for i, full_loc in enumerate(full_locations):
            track = self.face_tracker.find_track_for_location(
                full_loc, min_iou=self.IDENTITY_LOCK_IOU
            )
            if (
                track is not None
                and track.locked_name is not None
                and track.frames_since_recog < self.IDENTITY_LOCK_TTL_FRAMES
                and track.track_id not in used_tracks
            ):
                used_tracks.add(track.track_id)
                raw_results[i] = {
                    "name": track.locked_name,
                    "registered": track.locked_name != "Unknown",
                    "similarity": track.locked_similarity,
                    "distance": track.locked_distance,
                    "location": full_loc,
                    "from_cache": True,
                }
            else:
                encode_indices.append(i)

        # ── Stage 3: encode + match ONLY the un-cached subset ────────
        if encode_indices:
            sub_small = [small_locations[i] for i in encode_indices]
            sub_results = self.face_recognizer.recognize_faces(
                rgb_small, sub_small, is_rgb=True
            )
            for j, idx in enumerate(encode_indices):
                if j >= len(sub_results):
                    continue
                res = sub_results[j]
                res["location"] = full_locations[idx]
                res["from_cache"] = False
                raw_results[idx] = res

        # Drop any holes from sub-result indexing edge cases.
        raw_results = [r for r in raw_results if r is not None]
        t3 = time.perf_counter()

        # ── Stage 4: Track identities ────────────────────────────────
        stable_results = self.face_tracker.update(raw_results)
        t4 = time.perf_counter()

        # ── Stage 4b: Refresh active-student session ────────────────
        # Use the largest registered + stable face as the current
        # "owner" of the mic/camera.  If the active student is a guest
        # (registered=False, name starts with Guest_), keep them pinned
        # — recognising the user's own face shouldn't kick a guest out
        # mid-question, but a registered face WILL take over.
        best_registered = None
        best_area = 0
        for r in stable_results:
            if not r.get("registered"):
                continue
            loc = r.get("location")
            if not loc:
                continue
            top, right, bottom, left = loc
            area = max(0, (bottom - top) * (right - left))
            if area > best_area and r.get("stable"):
                best_area = area
                best_registered = r
        if best_registered is not None:
            self.set_active_student(best_registered["name"], registered=True)

        # ── Stage 3: Attendance — fires IMMEDIATELY, no emotion gate ─
        # Attendance is logged as soon as identity is stable + confirmed.
        # Emotion is NOT required — it updates asynchronously later.
        any_newly_marked = False
        if mark_attendance:
            for res in stable_results:
                if res.get("attendance_ready") and res.get("registered"):
                    record = self.attendance_service.mark_attendance(
                        name=res["name"],
                        registered=True,
                        similarity=res.get("similarity", 0.0),
                        emotion=None,  # emotion not yet available
                        emotion_confidence=None,
                    )
                    if record is not None:
                        res["newly_marked"] = True
                        any_newly_marked = True
        t5 = time.perf_counter()

        # ── Stage 4: Emotion — ONLY for already-marked faces ─────────
        # This is the key perf optimisation: FER inference (~200-500 ms)
        # is skipped entirely during the critical first-recognition path.
        # Once attendance is logged, emotion sampling begins and the
        # frontend shows the averaged result after ~5 samples.
        for result in stable_results:
            label, confidence = "Neutral", 0.0
            emotion_stable = self.emotion_tracker is None
            emotion_samples = 0
            name = result.get("name", "Unknown")
            already_marked = self.attendance_service.already_marked(name)

            if already_marked and self.emotion_tracker is not None:
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
            result.setdefault("newly_marked", False)

            # Debug logging for emotion buffer (only for marked faces)
            if already_marked and self.emotion_tracker is not None:
                tid = result.get("track_id", -1)
                buf = self.emotion_tracker._buffers.get(tid)
                buffer_labels = [lbl for lbl, _ in buf.buffer] if buf and buf.buffer else []
                logger.debug(
                    "Emotion [track %d / %s]: %d/%d stable=%s %s → %s",
                    tid, name, emotion_samples,
                    self.emotion_tracker.min_stable_samples,
                    emotion_stable, buffer_labels, label,
                )

        t6 = time.perf_counter()

        # ── Timing summary ───────────────────────────────────────────
        cache_hits = sum(1 for r in raw_results if r.get("from_cache"))
        encoded = len(raw_results) - cache_hits
        logger.info(
            "⏱ Frame %d: detect=%.0fms recog=%.0fms (encoded=%d cached=%d) "
            "track=%.0fms attend=%.0fms emotion=%.0fms TOTAL=%.0fms faces=%d%s",
            self.frame_count,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            encoded,
            cache_hits,
            (t4 - t3) * 1000,
            (t5 - t4) * 1000,
            (t6 - t5) * 1000,
            (t6 - t0) * 1000,
            len(stable_results),
            " [NEW_ATTENDANCE]" if any_newly_marked else "",
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
            emotion_stable = r.get("emotion_stable", False)
            emotion = r.get("emotion", "")
            if emotion_stable:
                text = f"{label}" + (f" - {emotion}" if emotion else "")
            else:
                text = f"{label} - Detecting..."

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
        self.clear_active_student()

    def get_active_student(
        self, ttl_seconds: Optional[float] = None
    ) -> Optional[str]:
        """Return the student/guest who owns the current mic context.

        The active student is whoever was last seen on camera *or* the
        guest the user explicitly chose to continue as.  Returns
        ``None`` when no one has been seen within the TTL window so the
        caller can decide between attributing to ``Unknown`` or
        rejecting the request.

        Args:
            ttl_seconds: Override the default
                :attr:`ACTIVE_STUDENT_TTL_SECONDS` window.
        """
        ttl = self.ACTIVE_STUDENT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        with self._active_lock:
            if not self._active_student:
                return None
            if (time.monotonic() - self._active_student_seen_at) > ttl:
                return None
            return self._active_student

    def get_active_student_info(
        self, ttl_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Return the current active student name + registered flag.

        Equivalent to :meth:`get_active_student` but returns the
        ``registered`` flag too so the caller can build a question
        event without a second lookup.
        """
        ttl = self.ACTIVE_STUDENT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        with self._active_lock:
            if not self._active_student:
                return {"name": None, "registered": False, "fresh": False}
            fresh = (time.monotonic() - self._active_student_seen_at) <= ttl
            return {
                "name": self._active_student if fresh else None,
                "registered": self._active_student_registered if fresh else False,
                "fresh": fresh,
            }

    def set_active_student(self, name: str, registered: bool) -> None:
        """Manually pin the active student (used by the guest flow).

        Args:
            name:        Either a registered student name or ``Guest_NNN``.
            registered:  ``True`` for known students, ``False`` for guests.
        """
        with self._active_lock:
            self._active_student = name
            self._active_student_registered = bool(registered)
            self._active_student_seen_at = time.monotonic()
        logger.info("Active student set → %s (registered=%s)", name, registered)

    def clear_active_student(self) -> None:
        with self._active_lock:
            self._active_student = None
            self._active_student_registered = False
            self._active_student_seen_at = 0.0

    def register_guest(self) -> Dict[str, Any]:
        """Allocate a new ``Guest_NNN`` identity, log it, and pin it as active.

        This is the backend half of the "Continue as Guest" button.
        Returns the persisted student record so the route can hand it
        back to the frontend.
        """
        record = self.attendance_service.register_guest()
        self.set_active_student(record["student"], registered=False)
        return record

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
