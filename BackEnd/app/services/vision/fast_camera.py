"""
Fast Camera — Pure Vision Performance Mode
============================================
Standalone webcam runner optimized for MAXIMUM SPEED face recognition.

Why a separate runner?
    The default webcam pipeline (``webcam_runner.py``) is wired to
    emotion detection, push-to-talk speech, NLP topic classification,
    and the HTTP/MJPEG API.  Every one of those adds latency, threads,
    or model load time that hurts cold-start recognition.

What this runner does (and ONLY this):
    1. Loads ``data/encodings/face_encodings.pkl`` once into a single
       stacked ``(N, 128)`` numpy matrix for vectorized matching.
    2. Captures from the local webcam via ``cv2.VideoCapture``.
    3. Runs HOG face detection on a downscaled frame.
    4. Computes 128-D face encodings ONLY every ``recog_every`` frames
       (or when a track has just been created / lost).
    5. Caches the identity per KCF tracker — between recognitions the
       tracker carries the bounding box and the cached name forward,
       so no recognition cost is paid for already-seen faces.
    6. Prints per-stage timings (detect / recog / track / total).

What this runner does NOT do:
    - No emotion detection (FER / DeepFace / TensorFlow).
    - No speech / NLP / question pipeline.
    - No attendance logging, no LogService, no admin/registration flow.
    - No HTTP API, no MJPEG stream, no frontend polling.
    - No CLAHE, no jitters, no per-frame logging spam.

Controls (in the OpenCV window):
    Q / Esc — quit
    R       — reset all tracks (re-recognize on next frame)
    B       — rebuild the encoding cache from data/students_faces/

Run it directly::

    python -m app.services.vision.fast_camera
    python BackEnd/tests/test_live_camera.py
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition as fr_lib
import numpy as np

# ── Allow `python app/services/vision/fast_camera.py` from BackEnd/ ──
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import settings  # noqa: E402

logger = logging.getLogger(__name__)


# ── Drawing constants ───────────────────────────────────────────────
_COLOR_KNOWN = (0, 200, 0)
_COLOR_UNKNOWN = (0, 0, 230)
_COLOR_PENDING = (0, 165, 255)
_COLOR_TEXT_BG = (30, 30, 30)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ════════════════════════════════════════════════════════════════════
#  FAST MATCHER — vectorized identity lookup
# ════════════════════════════════════════════════════════════════════


class FastMatcher:
    """Vectorized matcher over a stacked encoding matrix.

    The pickled encoding cache stores ``encodings`` as a list of
    128-D vectors.  We pre-stack them into a single ``(N, 128)``
    numpy array so a frame's encoding can be matched against ALL
    known faces in ONE BLAS call::

        distances = np.linalg.norm(matrix - encoding, axis=1)

    For 53 encodings this is ~50 µs — effectively free.  The actual
    cost of recognition is dominated by ``face_encodings`` itself,
    not the distance comparison, which is why this matcher is wired
    to the per-track caching strategy in :class:`FastClassroomCamera`.
    """

    def __init__(
        self,
        encodings_file: Optional[Path] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        self.encodings_file: Path = encodings_file or settings.ENCODINGS_FILE
        self.tolerance: float = (
            tolerance if tolerance is not None else settings.FACE_RECOGNITION_TOLERANCE
        )
        self.matrix: Optional[np.ndarray] = None  # shape (N, 128)
        self.names: List[str] = []
        self._student_index: Dict[str, List[int]] = {}

    # ── Loading / rebuilding ─────────────────────────────────────────

    def load(self) -> bool:
        """Load encodings from the ``.pkl`` cache into the matrix."""
        if not self.encodings_file.exists():
            logger.warning("Encoding cache missing: %s", self.encodings_file)
            return False

        try:
            with open(self.encodings_file, "rb") as fh:
                data = pickle.load(fh)
            names = list(data.get("names", []))
            encodings = data.get("encodings", [])
        except (pickle.UnpicklingError, EOFError, KeyError, ModuleNotFoundError) as exc:
            logger.error("Failed to load encoding cache: %s", exc)
            return False

        if not encodings:
            logger.warning("Encoding cache is empty.")
            self.matrix = None
            self.names = []
            self._student_index = {}
            return False

        self.matrix = np.stack([np.asarray(e, dtype=np.float64) for e in encodings])
        self.names = names

        # student -> [indices] grouping (FIX 2: smarter per-student lookup)
        self._student_index.clear()
        for idx, name in enumerate(names):
            self._student_index.setdefault(name, []).append(idx)

        logger.info(
            "FastMatcher loaded: %d encodings across %d students.",
            len(self.names),
            len(self._student_index),
        )
        return True

    def rebuild_from_disk(self) -> bool:
        """Force-rebuild the cache from ``data/students_faces/`` and reload."""
        from app.services.vision.encoding_manager import EncodingManager

        mgr = EncodingManager(encodings_file=self.encodings_file)
        try:
            summary = mgr.build_encodings()
        except FileNotFoundError as exc:
            logger.error("Cannot rebuild — %s", exc)
            return False

        logger.info(
            "Rebuilt %d encodings for %d students.",
            summary.get("total_encodings", 0),
            len(summary.get("students", {})),
        )
        return self.load()

    # ── Matching ─────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self.matrix is not None and len(self.names) > 0

    def match(self, encoding: np.ndarray) -> Tuple[str, float, float]:
        """Return ``(name, similarity, distance)`` for one encoding.

        Uses a single vectorized L2 distance over all known encodings,
        then picks the minimum.  ``similarity`` is a 0–1 score derived
        from the distance for display purposes.
        """
        if not self.is_loaded or self.matrix is None:
            return "Unknown", 0.0, 1.0

        # Single BLAS call: (N, 128) - (128,) → (N, 128) → norm → (N,)
        diff = self.matrix - encoding
        distances = np.linalg.norm(diff, axis=1)

        idx = int(np.argmin(distances))
        d = float(distances[idx])

        if d <= self.tolerance:
            return self.names[idx], self._sim(d), d
        return "Unknown", self._sim(d), d

    def _sim(self, distance: float) -> float:
        """Distance → similarity in [0, 1].  Non-linear inside tolerance."""
        if distance <= 0.0:
            return 1.0
        if distance <= self.tolerance:
            ratio = distance / self.tolerance
            return round(1.0 - 0.5 * (ratio ** 0.65), 4)
        overshoot = (distance - self.tolerance) / max(1.0 - self.tolerance, 0.001)
        return round(max(0.0, 0.5 * (1.0 - overshoot)), 4)


# ════════════════════════════════════════════════════════════════════
#  TRACKER FACTORY — KCF with graceful fallbacks
# ════════════════════════════════════════════════════════════════════


def _make_tracker() -> Optional[Any]:
    """Build a fast object tracker.  Prefers KCF (≈3-5 ms / face)."""
    candidates = (
        ("TrackerKCF_create", False),
        ("TrackerKCF_create", True),    # legacy submodule
        ("TrackerCSRT_create", False),  # slower but more robust
        ("TrackerCSRT_create", True),
        ("TrackerMIL_create", False),
    )
    for name, use_legacy in candidates:
        ns = getattr(cv2, "legacy", None) if use_legacy else cv2
        if ns is None:
            continue
        ctor = getattr(ns, name, None)
        if ctor is None:
            continue
        try:
            return ctor()
        except Exception:  # pragma: no cover — pure availability check
            continue
    return None


# ════════════════════════════════════════════════════════════════════
#  FAST TRACK — KCF tracker + cached identity
# ════════════════════════════════════════════════════════════════════


class FastTrack:
    """A single tracked face with frozen identity.

    Once the face has been recognized, its name + similarity are
    cached on the track and reused for every subsequent frame UNTIL
    the tracker fails or the periodic re-recognition interval fires.
    """

    __slots__ = (
        "track_id",
        "tracker",
        "bbox",
        "name",
        "registered",
        "similarity",
        "distance",
        "frames_since_recog",
        "frames_alive",
    )

    def __init__(
        self,
        track_id: int,
        tracker: Any,
        bbox: Tuple[int, int, int, int],
        name: str,
        registered: bool,
        similarity: float,
        distance: float,
    ) -> None:
        self.track_id = track_id
        self.tracker = tracker
        self.bbox = bbox  # (x, y, w, h) on the full-resolution frame
        self.name = name
        self.registered = registered
        self.similarity = similarity
        self.distance = distance
        self.frames_since_recog = 0
        self.frames_alive = 1

    def update(self, frame: np.ndarray) -> bool:
        """Advance the KCF tracker by one frame.  Returns success."""
        ok, bbox = self.tracker.update(frame)
        if not ok:
            return False
        x, y, w, h = (int(v) for v in bbox)
        if w <= 0 or h <= 0:
            return False
        self.bbox = (x, y, w, h)
        self.frames_alive += 1
        self.frames_since_recog += 1
        return True

    @property
    def location_trbl(self) -> Tuple[int, int, int, int]:
        """Bounding box in face_recognition's ``(top, right, bottom, left)`` form."""
        x, y, w, h = self.bbox
        return (y, x + w, y + h, x)


# ════════════════════════════════════════════════════════════════════
#  FAST CLASSROOM CAMERA — main runtime
# ════════════════════════════════════════════════════════════════════


class FastClassroomCamera:
    """Pure vision performance-mode webcam loop.

    Args:
        camera_index : OpenCV device index (default 0).
        scale        : Downscale factor for detection / encoding
                       (default 0.5).  ``0.25`` is faster but loses
                       accuracy on small faces in low-res streams.
        recog_every  : Run a full recognition pass every N frames.
        max_missed   : Drop a track after N consecutive failed
                       tracker updates.
        cam_width    : Capture width in pixels (default 1280).
        cam_height   : Capture height in pixels (default 720).
    """

    def __init__(
        self,
        camera_index: int = 0,
        scale: float = 0.5,
        recog_every: int = 10,
        max_missed: int = 6,
        cam_width: int = 640,
        cam_height: int = 480,
        encodings_file: Optional[Path] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        # ── 1. Static configuration (no I/O, no callbacks) ───────────
        self.camera_index = camera_index
        self.scale = float(scale)
        self.inv_scale = 1.0 / self.scale
        self.recog_every = max(1, int(recog_every))
        self.max_missed = max(1, int(max_missed))
        self.cam_width = cam_width
        self.cam_height = cam_height

        # ── 2. Runtime mutable state — set BEFORE any I/O ────────────
        # Anything ``rebuild_encodings`` / ``reset_session`` /
        # ``process_frame`` may touch must already exist by the time we
        # call them below.  This is the fix for the startup
        # AttributeError when a missing-cache rebuild fired before the
        # tracker dict existed.
        self.tracks: Dict[int, FastTrack] = {}
        self._next_id: int = 0
        self.frame_count: int = 0
        self.last_timing: Dict[str, float] = {
            "detect": 0.0,
            "recog": 0.0,
            "track": 0.0,
            "total": 0.0,
        }
        self.matcher: FastMatcher = FastMatcher(
            encodings_file=encodings_file, tolerance=tolerance
        )

        # ── 3. I/O — safe now ────────────────────────────────────────
        if not self.matcher.load():
            logger.warning(
                "Encoding cache missing. Auto-building from data/students_faces/ …"
            )
            self.rebuild_encodings()
            if not self.matcher.is_loaded:
                logger.warning(
                    "Still no encodings loaded — every face will be Unknown."
                )

        # FIX 9 — wipe the attendance log on every cold start.
        self._reset_log_file()

    # ── Log / state reset ────────────────────────────────────────────

    @staticmethod
    def _reset_log_file() -> None:
        """Truncate the configured attendance log to ``[]``."""
        log_path: Path = settings.ATTENDANCE_LOG_FILE
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("[]", encoding="utf-8")
            logger.info("Reset attendance log: %s", log_path)
        except OSError as exc:  # pragma: no cover — fs error
            logger.warning("Could not reset log file %s: %s", log_path, exc)

    def reset_session(self) -> None:
        """Drop all live tracks and re-truncate the log."""
        self.tracks.clear()
        self._next_id = 0
        self.frame_count = 0
        self._reset_log_file()
        logger.info("Session reset — %d tracks cleared.", 0)

    def rebuild_encodings(self) -> bool:
        """Rebuild the on-disk cache and reload the matcher."""
        ok = self.matcher.rebuild_from_disk()
        # Existing tracks have stale identities — drop them.
        self.tracks.clear()
        self._next_id = 0
        return ok

    # ── Frame processing ─────────────────────────────────────────────

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run the optimized detect → track → recognize pipeline."""
        if frame is None or frame.size == 0:
            return frame, dict(self.last_timing)

        self.frame_count += 1
        timing = {"detect": 0.0, "recog": 0.0, "track": 0.0, "total": 0.0}
        t_start = time.perf_counter()

        # ── 1. Update existing KCF trackers ──────────────────────────
        t_track0 = time.perf_counter()
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if not track.update(frame):
                # Tracker lost — remove so a fresh detection can recover it.
                del self.tracks[tid]
        timing["track"] = (time.perf_counter() - t_track0) * 1000.0

        # ── 2. Decide whether to re-detect / re-recognize ────────────
        need_recog = (
            not self.tracks
            or (self.frame_count % self.recog_every) == 0
            or any(
                t.frames_since_recog >= self.recog_every for t in self.tracks.values()
            )
        )

        if need_recog:
            self._recognize_pass(frame, timing)

        timing["total"] = (time.perf_counter() - t_start) * 1000.0
        self.last_timing = timing

        annotated = self._draw_annotations(frame.copy())
        return annotated, timing

    def _recognize_pass(
        self, frame: np.ndarray, timing: Dict[str, float]
    ) -> None:
        """Detect + encode + match on a downscaled frame, then reconcile."""
        # Downscale + RGB conversion once.
        small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ── Detection (HOG) ──────────────────────────────────────────
        t_d0 = time.perf_counter()
        small_locations = fr_lib.face_locations(rgb_small, model="hog")
        timing["detect"] = (time.perf_counter() - t_d0) * 1000.0

        if not small_locations:
            return

        # ── Encoding (the dominant cost) ─────────────────────────────
        t_r0 = time.perf_counter()
        encodings = fr_lib.face_encodings(
            rgb_small, known_face_locations=small_locations, num_jitters=1
        )
        timing["recog"] = (time.perf_counter() - t_r0) * 1000.0

        if not encodings:
            return

        # ── Scale boxes back to full-resolution frame ────────────────
        fh, fw = frame.shape[:2]
        full_locations: List[Tuple[int, int, int, int]] = []
        for top, right, bottom, left in small_locations:
            full_locations.append(
                (
                    max(0, min(int(top * self.inv_scale), fh - 1)),
                    max(0, min(int(right * self.inv_scale), fw - 1)),
                    max(0, min(int(bottom * self.inv_scale), fh - 1)),
                    max(0, min(int(left * self.inv_scale), fw - 1)),
                )
            )

        # ── Reconcile detections with existing tracks via IoU ────────
        self._reconcile(frame, full_locations, encodings)

    def _reconcile(
        self,
        frame: np.ndarray,
        locations: List[Tuple[int, int, int, int]],
        encodings: List[np.ndarray],
    ) -> None:
        """Match detected boxes to existing tracks (IoU); else create new."""
        track_locs: Dict[int, Tuple[int, int, int, int]] = {
            tid: t.location_trbl for tid, t in self.tracks.items()
        }

        # Greedy IoU pairing — sorted by overlap, highest first.
        pairs: List[Tuple[float, int, int]] = []
        for det_idx, loc in enumerate(locations):
            for tid, tloc in track_locs.items():
                iou = _iou(loc, tloc)
                if iou >= 0.30:
                    pairs.append((iou, det_idx, tid))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_dets: set = set()
        matched_tracks: set = set()
        for iou, det_idx, tid in pairs:
            if det_idx in matched_dets or tid in matched_tracks:
                continue
            matched_dets.add(det_idx)
            matched_tracks.add(tid)
            self._refresh_track(
                self.tracks[tid], frame, locations[det_idx], encodings[det_idx]
            )

        # Anything left unmatched → spin up a new track.
        for det_idx, (loc, enc) in enumerate(zip(locations, encodings)):
            if det_idx in matched_dets:
                continue
            self._create_track(frame, loc, enc)

    def _refresh_track(
        self,
        track: FastTrack,
        frame: np.ndarray,
        location: Tuple[int, int, int, int],
        encoding: np.ndarray,
    ) -> None:
        """Re-init the KCF tracker with the fresh detection box and re-match identity."""
        top, right, bottom, left = location
        x, y, w, h = left, top, right - left, bottom - top
        if w <= 0 or h <= 0:
            return

        new_tracker = _make_tracker()
        if new_tracker is not None:
            try:
                new_tracker.init(frame, (x, y, w, h))
                track.tracker = new_tracker
            except Exception as exc:  # pragma: no cover — opencv quirk
                logger.debug("Tracker re-init failed: %s", exc)

        name, similarity, distance = self.matcher.match(encoding)
        track.bbox = (x, y, w, h)
        track.name = name
        track.registered = name != "Unknown"
        track.similarity = similarity
        track.distance = distance
        track.frames_since_recog = 0

    def _create_track(
        self,
        frame: np.ndarray,
        location: Tuple[int, int, int, int],
        encoding: np.ndarray,
    ) -> None:
        top, right, bottom, left = location
        x, y, w, h = left, top, right - left, bottom - top
        if w <= 0 or h <= 0:
            return

        tracker = _make_tracker()
        if tracker is None:
            logger.debug("No tracker constructor available; skipping track creation.")
            return
        try:
            tracker.init(frame, (x, y, w, h))
        except Exception as exc:  # pragma: no cover — opencv quirk
            logger.debug("Tracker init failed: %s", exc)
            return

        name, similarity, distance = self.matcher.match(encoding)
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = FastTrack(
            track_id=tid,
            tracker=tracker,
            bbox=(x, y, w, h),
            name=name,
            registered=(name != "Unknown"),
            similarity=similarity,
            distance=distance,
        )

    # ── Drawing ──────────────────────────────────────────────────────

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        for track in self.tracks.values():
            x, y, w, h = track.bbox
            if track.registered:
                color = _COLOR_KNOWN
                label = f"{track.name} ({track.similarity:.0%})"
            elif track.name == "Unknown":
                color = _COLOR_UNKNOWN
                label = "Unknown"
            else:
                color = _COLOR_PENDING
                label = f"{track.name}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            (tw, th), base = cv2.getTextSize(label, _FONT, 0.6, 1)
            ly = max(y - 10, th + 10)
            cv2.rectangle(
                frame,
                (x, ly - th - 6),
                (x + tw + 8, ly + base),
                _COLOR_TEXT_BG,
                cv2.FILLED,
            )
            cv2.putText(frame, label, (x + 4, ly - 2), _FONT, 0.6, color, 1)
        return frame

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the local webcam and run the live recognition loop."""
        cap = self._open_capture()
        if cap is None:
            logger.error(
                "Could not open camera %d. Check that no other process is using it.",
                self.camera_index,
            )
            return

        logger.info(
            "Fast camera live: scale=%.2f recog_every=%d encodings=%d students=%d",
            self.scale,
            self.recog_every,
            len(self.matcher.names),
            len(self.matcher._student_index),
        )
        print(
            "\nControls:  Q/Esc=Quit | R=Reset tracks | B=Rebuild encodings\n"
        )

        last_frame_time = time.time()
        fps = 0.0

        try:
            while True:
                ok, frame = cap.read()
                
                # FIX 2: FRAME VALIDATION
                if not ok:
                    logger.warning("Frame read failed (ret=False) — skipping.")
                    continue
                if frame is None:
                    logger.warning("Frame is None — skipping.")
                    continue
                if frame.size == 0 or len(frame.shape) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    logger.warning("Frame size invalid — skipping.")
                    continue

                annotated, timing = self.process_frame(frame)

                # ── FPS overlay ──────────────────────────────────────
                t_now = time.time()
                dt = t_now - last_frame_time
                last_frame_time = t_now
                
                if dt > 0:
                    current_fps = 1.0 / dt
                    # FIX 4: Proper FPS calculation using exponential moving average
                    if fps == 0.0:
                        fps = current_fps
                    else:
                        fps = (fps * 0.9) + (current_fps * 0.1)

                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 25),
                    _FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # ── Per-stage timing footer ──────────────────────────
                footer = (
                    f"detect: {timing['detect']:5.1f}ms  "
                    f"recog: {timing['recog']:5.1f}ms  "
                    f"track: {timing['track']:5.1f}ms  "
                    f"total: {timing['total']:5.1f}ms  "
                    f"tracks: {len(self.tracks)}"
                )
                cv2.putText(
                    annotated,
                    footer,
                    (10, annotated.shape[0] - 15),
                    _FONT,
                    0.5,
                    (220, 220, 220),
                    1,
                )

                # ── Console timing (only on recognition frames) ──────
                if timing["recog"] > 0.0 or timing["detect"] > 0.0:
                    print(
                        f"⏱ frame={self.frame_count:5d}  "
                        f"detect={timing['detect']:6.1f}ms  "
                        f"recog={timing['recog']:6.1f}ms  "
                        f"track={timing['track']:5.1f}ms  "
                        f"total={timing['total']:6.1f}ms  "
                        f"tracks={len(self.tracks)}"
                    )

                cv2.imshow(
                    "Fast Camera — Vision Performance Mode", annotated
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    self.reset_session()
                    print("Session reset.")
                elif key in (ord("b"), ord("B")):
                    print("Rebuilding encodings from data/students_faces/ …")
                    if self.rebuild_encodings():
                        print(
                            f"Rebuild OK — {len(self.matcher.names)} encodings "
                            f"across {len(self.matcher._student_index)} students."
                        )
                    else:
                        print("Rebuild FAILED — see logs.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released.")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open the webcam with diagnostics, fallbacks and safe resolution."""
        backends = []
        if sys.platform.startswith("win"):
            backends = [
                ("CAP_DSHOW", cv2.CAP_DSHOW),
                ("CAP_MSMF", cv2.CAP_MSMF),
                ("CAP_ANY", cv2.CAP_ANY)
            ]
        else:
            backends = [("CAP_ANY", cv2.CAP_ANY)]

        indices = [0, 1, 2] if self.camera_index == 0 else [self.camera_index]
        
        for idx in indices:
            for backend_name, backend_flag in backends:
                cap = cv2.VideoCapture(idx, backend_flag)
                if not cap.isOpened():
                    cap.release()
                    continue

                # Apply safe resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                if sys.platform.startswith("win"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Read a test frame to validate
                for _ in range(3): # Try a few times to let camera warm up
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    time.sleep(0.1)

                print("=" * 40)
                print("CAMERA STARTUP DIAGNOSTICS")
                print(f"Backend: {backend_name}")
                print(f"Camera index: {idx}")
                print(f"ret={ret}")
                
                if ret and frame is not None and frame.size > 0:
                    print(f"Frame shape={frame.shape}")
                    print(f"Resolution={frame.shape[1]}x{frame.shape[0]}")
                    print("=" * 40)
                    self.camera_index = idx
                    return cap
                else:
                    print("Frame shape=Invalid")
                    print("=" * 40)
                    print("Warning: Camera opened but failed to read valid frames. Trying next...")
                    cap.release()

        return None


# ════════════════════════════════════════════════════════════════════
#  IoU helper
# ════════════════════════════════════════════════════════════════════


def _iou(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    """Intersection over Union for two ``(top, right, bottom, left)`` boxes."""
    t1, r1, b1, l1 = a
    t2, r2, b2, l2 = b
    inter_t = max(t1, t2)
    inter_l = max(l1, l2)
    inter_b = min(b1, b2)
    inter_r = min(r1, r2)
    if inter_b <= inter_t or inter_r <= inter_l:
        return 0.0
    inter = (inter_b - inter_t) * (inter_r - inter_l)
    a1 = (b1 - t1) * (r1 - l1)
    a2 = (b2 - t2) * (r2 - l2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ════════════════════════════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast Camera — pure vision performance mode (no emotion / NLP / API).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Downscale factor for detection/encoding (0.25 = fastest, 0.5 = default).",
    )
    parser.add_argument(
        "--recog-every",
        type=int,
        default=10,
        help="Run full recognition every N frames (default 10).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild encodings before starting.",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Webcam capture width."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Webcam capture height."
    )
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="Run in camera-only diagnostic mode (no recognition).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    )

    runner = FastClassroomCamera(
        camera_index=args.camera,
        scale=args.scale,
        recog_every=args.recog_every,
        cam_width=args.width,
        cam_height=args.height,
    )

    if args.test_camera:
        print("\n=== STARTING CAMERA DIAGNOSTIC MODE ===")
        cap = runner._open_capture()
        if cap is None:
            print("Failed to open camera in diagnostic mode.")
            return
            
        print("Camera opened successfully. Showing raw frames. Press Q to exit.")
        try:
            fps = 0.0
            last_t = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: ret=False")
                    continue
                if frame is None or frame.size == 0:
                    print("Warning: Invalid frame")
                    continue
                
                t_now = time.time()
                dt = t_now - last_t
                last_t = t_now
                if dt > 0:
                    if fps == 0.0:
                        fps = 1.0 / dt
                    else:
                        fps = fps * 0.9 + (1.0 / dt) * 0.1
                
                cv2.putText(frame, f"RAW FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera Diagnostic Mode", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    if args.rebuild:
        print("Rebuilding encodings from data/students_faces/ …")
        runner.rebuild_encodings()

    runner.run()


if __name__ == "__main__":
    main()
