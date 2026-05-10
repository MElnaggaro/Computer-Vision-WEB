"""
Face Tracker — Temporal Stabilization
======================================
Tracks face identities across consecutive frames using IoU matching
and majority voting to eliminate flicker and false positives.

Architecture:
    FaceDetector → FaceRecognizer → **FaceTracker** → stable output

Each detected face is assigned to a *track*. The track accumulates
per‑frame recognition results and only emits a **stable identity**
once a name has been consistently recognized across enough frames.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

FaceLocation = Tuple[int, int, int, int]


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class TrackedFace:
    """A single tracked face with recognition history."""

    track_id: int
    location: FaceLocation
    history: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=settings.TRACK_HISTORY_SIZE))
    frames_missed: int = 0
    stable_count: int = 0          # consecutive frames with same stable name
    _last_stable_name: Optional[str] = field(default=None, repr=False)

    def push(self, name: str, distance: float) -> None:
        """Record one frame's recognition result."""
        self.history.append((name, distance))
        self.frames_missed = 0

        # Track consecutive stability
        stable = self.stable_identity
        if stable is not None and stable[0] == self._last_stable_name:
            self.stable_count += 1
        elif stable is not None:
            self._last_stable_name = stable[0]
            self.stable_count = 1
        else:
            self._last_stable_name = None
            self.stable_count = 0

    @property
    def stable_identity(self) -> Optional[Tuple[str, float]]:
        """Return (name, avg_similarity) if majority vote passes threshold.

        Returns ``None`` if no name dominates the history.
        """
        if len(self.history) < min(3, settings.TRACK_STABILITY_THRESHOLD):
            return None

        names = [name for name, _ in self.history]
        counter = Counter(names)
        best_name, count = counter.most_common(1)[0]

        if count >= settings.TRACK_STABILITY_THRESHOLD:
            # Average distance for the winning name
            dists = [d for n, d in self.history if n == best_name]
            avg_dist = sum(dists) / len(dists)
            similarity = round(max(0.0, 1.0 - avg_dist), 4)
            return (best_name, similarity)

        return None

    @property
    def is_attendance_ready(self) -> bool:
        """True when identity is stable for enough consecutive frames."""
        return self.stable_count >= settings.ATTENDANCE_STABLE_FRAMES


# ── IoU helper ───────────────────────────────────────────────────────

def _iou(box1: FaceLocation, box2: FaceLocation) -> float:
    """Compute Intersection over Union for two (top, right, bottom, left) boxes."""
    t1, r1, b1, l1 = box1
    t2, r2, b2, l2 = box2

    inter_t = max(t1, t2)
    inter_l = max(l1, l2)
    inter_b = min(b1, b2)
    inter_r = min(r1, r2)

    if inter_b <= inter_t or inter_r <= inter_l:
        return 0.0

    inter_area = (inter_b - inter_t) * (inter_r - inter_l)
    area1 = (b1 - t1) * (r1 - l1)
    area2 = (b2 - t2) * (r2 - l2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ── Tracker ──────────────────────────────────────────────────────────

class FaceTracker:
    """Track faces across frames and stabilize identity via majority voting.

    Usage::

        tracker = FaceTracker()
        for frame in video:
            raw_results = recognizer.recognize_faces(frame, locations)
            stable_results = tracker.update(raw_results)
            # stable_results only contain confirmed identities
    """

    def __init__(
        self,
        iou_threshold: Optional[float] = None,
        max_missed: Optional[int] = None,
    ) -> None:
        self.iou_threshold = iou_threshold or settings.TRACK_IOU_THRESHOLD
        self.max_missed = max_missed or settings.TRACK_MAX_MISSED_FRAMES
        self._tracks: Dict[int, TrackedFace] = {}
        self._next_id: int = 0

    def update(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match detections to tracks, update histories, return stabilized results.

        Args:
            raw_results: Per‑frame recognition results from ``FaceRecognizer``.

        Returns:
            List of stabilized result dicts with added fields:
            ``track_id``, ``stable``, ``stable_frames``.
        """
        # Increment missed counter on all existing tracks
        for track in self._tracks.values():
            track.frames_missed += 1

        matched_track_ids: set = set()
        matched_det_idxs: set = set()
        output: List[Dict[str, Any]] = []

        # ── Greedy IoU matching ──────────────────────────────────────
        pairs = []
        for det_idx, det in enumerate(raw_results):
            loc = det.get("location")
            if loc is None:
                continue
            for tid, track in self._tracks.items():
                score = _iou(loc, track.location)
                if score >= self.iou_threshold:
                    pairs.append((score, det_idx, tid))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for score, det_idx, tid in pairs:
            if det_idx in matched_det_idxs or tid in matched_track_ids:
                continue
            # Match found
            det = raw_results[det_idx]
            track = self._tracks[tid]
            track.location = det["location"]
            track.push(det["name"], det.get("distance", 1.0 - det.get("similarity", 0.0)))
            matched_track_ids.add(tid)
            matched_det_idxs.add(det_idx)

        # ── Create new tracks for unmatched detections ───────────────
        for det_idx, det in enumerate(raw_results):
            if det_idx in matched_det_idxs:
                continue
            loc = det.get("location")
            if loc is None:
                continue
            new_track = TrackedFace(track_id=self._next_id, location=loc)
            new_track.push(det["name"], det.get("distance", 1.0 - det.get("similarity", 0.0)))
            self._tracks[self._next_id] = new_track
            matched_track_ids.add(self._next_id)
            self._next_id += 1

        # ── Remove stale tracks ──────────────────────────────────────
        stale = [tid for tid, t in self._tracks.items()
                 if tid not in matched_track_ids and t.frames_missed > self.max_missed]
        for tid in stale:
            del self._tracks[tid]

        # ── Build output from all active tracks ──────────────────────
        for tid in sorted(self._tracks):
            track = self._tracks[tid]
            stable = track.stable_identity

            if stable is not None:
                name, similarity = stable
                registered = name != "Unknown"
            else:
                # Use latest raw result
                if track.history:
                    last_name, last_dist = track.history[-1]
                    name = last_name
                    similarity = round(max(0.0, 1.0 - last_dist), 4)
                    registered = name != "Unknown"
                else:
                    name, similarity, registered = "Unknown", 0.0, False

            output.append({
                "track_id": track.track_id,
                "name": name,
                "registered": registered,
                "similarity": similarity,
                "location": track.location,
                "stable": stable is not None,
                "stable_frames": track.stable_count,
                "attendance_ready": track.is_attendance_ready,
            })

        return output

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0

    @property
    def tracks(self) -> Dict[int, TrackedFace]:
        return dict(self._tracks)
