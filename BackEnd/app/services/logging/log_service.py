"""
Log Service — Append-Only Event Logger
========================================
Shared logging service for the Smart Classroom project.

All classroom events (attendance, questions, etc.) are persisted as
an **append-only** JSON array in a single unified log file::

    BackEnd/logs/classroom_log.json

Design principles:
    • **Append-only** — existing events are NEVER overwritten.
    • **Thread-safe** — all file I/O is guarded by a ``threading.Lock``.
    • **Idempotent file init** — missing file → created; empty file → ``[]``.
    • **Windows-compatible** — uses ``pathlib.Path`` and ``utf-8`` encoding.

Event types
-----------
1. ``attendance`` — emitted when a student is recognised by the webcam.
2. ``question``   — emitted when a student asks a question (speech or text).

Usage::

    from app.services.logging.log_service import LogService

    log = LogService()                       # uses default log path
    log.log_attendance_event(...)
    log.log_question_event(...)

    all_events = log.load_logs()             # read everything back
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

EventRecord = Dict[str, Any]


class LogService:
    """Append-only event logger backed by a single JSON file.

    Args:
        log_file: Override the default log path
                  (``BackEnd/logs/classroom_log.json``).
    """

    def __init__(self, log_file: Optional[Path] = None) -> None:
        self.log_file: Path = log_file or settings.ATTENDANCE_LOG_FILE
        self._lock = threading.Lock()

        # Ensure the parent directory and file exist on construction
        self._ensure_file()

    # ── File helpers ─────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Create the log file with an empty JSON array if it doesn't exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self._write_events([])
            logger.info("Created new log file: %s", self.log_file)

    def _read_events(self) -> List[EventRecord]:
        """Read the full event list from disk.

        Returns an empty list on any I/O or JSON error (never raises).
        """
        try:
            content = self.log_file.read_text(encoding="utf-8").strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            logger.warning("Log file root is not a list — resetting.")
            return []
        except (json.JSONDecodeError, IOError, OSError) as exc:
            logger.warning("Could not read log file: %s", exc)
            return []

    def _write_events(self, events: List[EventRecord]) -> None:
        """Overwrite the log file with the given event list."""
        self.log_file.write_text(
            json.dumps(events, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Public API ───────────────────────────────────────────────────

    def load_logs(self) -> List[EventRecord]:
        """Load and return all events from the log file (thread-safe).

        Returns:
            A list of event dicts in chronological order.
        """
        with self._lock:
            events = self._read_events()
        logger.debug("Loaded %d events from %s", len(events), self.log_file)
        return events

    def append_event(self, event: EventRecord) -> EventRecord:
        """Append a single event to the log file (thread-safe).

        The event is written to disk immediately — no batching or
        buffering.  This guarantees that even if the process crashes,
        previously logged events are never lost.

        Args:
            event: A dict with at least an ``"event"`` key.

        Returns:
            The event dict that was written (with timestamp injected
            if missing).
        """
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            events = self._read_events()
            events.append(event)
            self._write_events(events)

        logger.info(
            "Appended %s event for %s",
            event.get("event", "unknown"),
            event.get("student", "N/A"),
        )
        return event

    # ── Convenience builders ─────────────────────────────────────────

    def log_attendance_event(
        self,
        student: str,
        attendance: str,
        registered: bool,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> EventRecord:
        """Build and append an ``attendance`` event.

        Args:
            student:            Student name (or ``"Unknown"``).
            attendance:         ``"Present"`` or ``"Not Registered"``.
            registered:         Whether the student was recognised.
            emotion:            Detected emotion label (optional).
            emotion_confidence: Emotion probability 0–1 (optional).
            timestamp:          ISO timestamp override (auto-generated
                                if not provided).

        Returns:
            The persisted event dict.
        """
        event: EventRecord = {
            "event": "attendance",
            "student": student,
            "attendance": attendance,
            "registered": registered,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }

        if emotion is not None:
            event["emotion"] = emotion
        if emotion_confidence is not None:
            event["emotion_confidence"] = round(emotion_confidence, 4)

        return self.append_event(event)

    def log_emotion_event(
        self,
        student: str,
        mood: str,
        registered: bool,
        samples: int,
        timestamp: Optional[str] = None,
    ) -> EventRecord:
        """Build and append an ``emotion`` event.

        Args:
            student:    Student name.
            mood:       The final smoothed emotion label.
            registered: Whether the student is registered.
            samples:    Number of predictions in the smoothing window.
            timestamp:  ISO timestamp override.
        """
        event: EventRecord = {
            "event": "emotion",
            "student": student,
            "mood": mood,
            "registered": registered,
            "samples": samples,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        return self.append_event(event)

    def log_registration_event(
        self,
        student: str,
        approved: bool,
        timestamp: Optional[str] = None,
    ) -> EventRecord:
        """Build and append a ``registration_approved`` / ``registration_rejected`` event.

        Args:
            student:   The candidate student name (``Firstname_Lastname``).
            approved:  ``True`` for an approved registration, ``False`` for rejection.
            timestamp: Optional ISO timestamp override.

        Returns:
            The persisted event dict.
        """
        event_type = "registration_approved" if approved else "registration_rejected"
        event: EventRecord = {
            "event": event_type,
            "student": student,
            "registered": bool(approved),
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        return self.append_event(event)

    def log_question_event(
        self,
        student: str,
        question: str,
        topic: str,
        classification_confidence: float = 0.0,
        registered: Optional[bool] = None,
        source: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> EventRecord:
        """Build and append a ``question`` event.

        Args:
            student:                     Student name (or ``"Unknown"``).
            question:                    The transcribed question text.
            topic:                       NLP-classified topic.
            classification_confidence:   Topic confidence 0–1.
            registered:                  Whether the student is registered
                                         (``None`` → omitted from event).
            source:                      Origin of the question, e.g.
                                         ``"manual_speech_test"``,
                                         ``"webcam_push_to_talk"``.
            timestamp:                   ISO timestamp override.

        Returns:
            The persisted event dict.
        """
        event: EventRecord = {
            "event": "question",
            "student": student,
            "question": question,
            "topic": topic,
            "classification_confidence": round(classification_confidence, 4),
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }

        if registered is not None:
            event["registered"] = registered
        if source is not None:
            event["source"] = source

        return self.append_event(event)
