"""
Attendance Service — Event-Based Logging
==========================================
Tracks student attendance and questions using an event-based log system.

All persistence is delegated to :class:`LogService` which maintains a
single, append-only JSON log at ``BackEnd/logs/classroom_log.json``.

Log format — each entry is a timestamped event::

    [
      {
        "event": "attendance",
        "student": "Mohammed_Ayman",
        "attendance": "Present",
        "emotion": "Happy",
        "emotion_confidence": 0.92,
        "timestamp": "2026-05-10T01:40:00+00:00",
        "registered": true
      },
      {
        "event": "question",
        "student": "Mohammed_Ayman",
        "question": "What is a semaphore?",
        "topic": "Operating System",
        "classification_confidence": 0.87,
        "timestamp": "2026-05-10T01:41:30+00:00"
      }
    ]

In-memory, each student also maintains a ``questions`` list for quick access::

    {
      "student": "Mohammed_Ayman",
      "attendance": "Present",
      "emotion": "Happy",
      "emotion_confidence": 0.92,
      "questions": [
        {"question": "What is a semaphore?", "topic": "Operating System", ...},
      ]
    }

Design decisions:
    • Uses an in-memory ``set`` for O(1) duplicate checks.
    • Event-based flat log for chronological traceability.
    • Per-student state dict with ``questions`` array for queries / overlay.
    • Unknown faces are logged with ``"attendance": "Not Registered"``.
    • Emotion and emotion_confidence are always included when available.
    • All file I/O is delegated to :class:`LogService` (thread-safe, append-only).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.services.logging.log_service import LogService

logger = logging.getLogger(__name__)

AttendanceRecord = Dict[str, Any]
EventRecord = Dict[str, Any]


class AttendanceService:
    """Mark attendance, record questions, and persist event-based JSON logs.

    Args:
        log_file: Override the default log file path.  Passed through to
                  :class:`LogService`.
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
    ) -> None:
        self.log_file = log_file or settings.ATTENDANCE_LOG_FILE

        # Shared log service handles all file I/O
        self._log_service = LogService(log_file=self.log_file)

        self._marked: Set[str] = set()                      # student names marked so far
        self._events: List[EventRecord] = []                 # chronological event log (session)
        self._student_state: Dict[str, AttendanceRecord] = {}  # per-student state

        # Guest counter — incremented per "Continue as Guest" click so
        # each unknown visitor gets a deterministic Guest_001, Guest_002, …
        self._guest_counter: int = 0
        self._guest_names: Set[str] = set()

        # Load students already present in the log file to prevent duplicates across runs
        self._load_existing_students()

    def _load_existing_students(self) -> None:
        """Populate the marked set from the existing log file."""
        try:
            existing = self._log_service.load_logs()
            highest_guest = 0
            for record in existing:
                event_type = record.get("event", "attendance")
                if event_type == "attendance":
                    name = record.get("student") or ""
                    registered = record.get("registered", False)
                    if name and registered and name != "Unknown":
                        self._marked.add(name)
                    if name.startswith("Guest_"):
                        self._guest_names.add(name)
                        try:
                            num = int(name.split("_", 1)[1])
                            highest_guest = max(highest_guest, num)
                        except (ValueError, IndexError):
                            pass
            self._guest_counter = highest_guest
            logger.info(
                "Loaded %d previously marked students, highest guest=%d.",
                len(self._marked),
                self._guest_counter,
            )
        except Exception as exc:
            logger.warning("Could not pre-load attendance log: %s", exc)

    # ── Public API ───────────────────────────────────────────────────

    def mark_attendance(
        self,
        name: str,
        registered: bool,
        similarity: float,
        emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
    ) -> Optional[AttendanceRecord]:
        """Record a student's attendance if not already marked.

        Creates both an event log entry (persisted immediately via
        :class:`LogService`) and a per-student state record.

        Args:
            name:               Student name or ``"Unknown"``.
            registered:         Whether the student was recognised.
            similarity:         Recognition similarity (0–1).
            emotion:            Classroom-friendly emotion label (optional).
            emotion_confidence: Probability of the predicted emotion (0–1).

        Returns:
            The ``AttendanceRecord`` dict if newly marked, or ``None``
            if attendance was already recorded for this student.
        """
        # Skip duplicates for registered students
        if registered and self.already_marked(name):
            logger.debug("Attendance already marked for %s – skipping.", name)
            return None

        timestamp = datetime.now(timezone.utc).isoformat()
        attendance_status = "Present" if registered else "Not Registered"

        # ── Persist via LogService (append-only, immediate write) ────
        persisted_event = self._log_service.log_attendance_event(
            student=name,
            attendance=attendance_status,
            registered=registered,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            timestamp=timestamp,
        )

        # ── Session-level event cache ────────────────────────────────
        self._events.append(persisted_event)

        # ── Per-student state ────────────────────────────────────────
        student_record: AttendanceRecord = {
            "student": name,
            "attendance": attendance_status,
            "emotion": emotion if emotion is not None else "Analyzing...",
            "emotion_confidence": emotion_confidence if emotion_confidence is not None else 0.0,
            "timestamp": timestamp,
            "registered": registered,
            "questions": [],
            "emotion_logged": False,
        }
        self._student_state[name] = student_record

        if registered:
            self._marked.add(name)

        logger.info(
            "Attendance marked: %s (%s, registered=%s)",
            name,
            attendance_status,
            registered,
        )
        return student_record

    def register_guest(self) -> AttendanceRecord:
        """Allocate a new ``Guest_NNN`` identity and log a guest attendance event.

        Called when the user clicks "Continue as Guest" on the dashboard.
        Each call produces a new monotonically-increasing guest id
        (``Guest_001``, ``Guest_002``, …) and persists an attendance
        event with ``registered=False`` so the dashboard can render it
        in the live feed and treat the guest like any tracked student
        for question attribution.

        Returns:
            The persisted student record for the new guest.
        """
        self._guest_counter += 1
        guest_name = f"Guest_{self._guest_counter:03d}"
        # Defensive: in case a guest with this id was already saved
        # (e.g. concurrent clicks), keep advancing.
        while guest_name in self._guest_names:
            self._guest_counter += 1
            guest_name = f"Guest_{self._guest_counter:03d}"
        self._guest_names.add(guest_name)

        timestamp = datetime.now(timezone.utc).isoformat()
        persisted_event = self._log_service.log_attendance_event(
            student=guest_name,
            attendance="Present",
            registered=False,
            timestamp=timestamp,
        )
        self._events.append(persisted_event)

        student_record: AttendanceRecord = {
            "student": guest_name,
            "attendance": "Present",
            "emotion": "Analyzing...",
            "emotion_confidence": 0.0,
            "timestamp": timestamp,
            "registered": False,
            "is_guest": True,
            "questions": [],
            "emotion_logged": False,
        }
        self._student_state[guest_name] = student_record
        logger.info("Guest registered: %s", guest_name)
        return student_record

    def is_guest(self, name: str) -> bool:
        """Return True when ``name`` is a registered guest in this session."""
        return bool(name) and name in self._guest_names

    def add_question(
        self,
        student_name: str,
        question: str,
        topic: str,
        topic_confidence: float = 0.0,
        source: Optional[str] = None,
    ) -> Optional[EventRecord]:
        """Record a question asked by a student.

        Creates a question event in the log (persisted immediately via
        :class:`LogService`) and appends to the student's ``questions``
        array.

        Args:
            student_name:    Name of the student who asked.  Either a
                             registered student name, a ``Guest_NNN``
                             identity, or ``"Unknown"``.
            question:        The transcribed question text.
            topic:           NLP-classified topic.
            topic_confidence: Classification confidence (0–1).
            source:          Origin of the question (e.g.
                             ``"webcam_push_to_talk"``).

        Returns:
            The question event dict, or ``None`` if student is unknown.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Determine registered status: registered students = True,
        # guests + Unknown = False (but the event is still persisted).
        if student_name in self._marked:
            registered = True
        else:
            registered = False

        # ── Persist via LogService (append-only, immediate write) ────
        persisted_event = self._log_service.log_question_event(
            student=student_name,
            question=question,
            topic=topic,
            classification_confidence=topic_confidence,
            registered=registered,
            source=source or "webcam_push_to_talk",
            timestamp=timestamp,
        )

        # ── Session-level event cache ────────────────────────────────
        self._events.append(persisted_event)

        # ── Update per-student state ─────────────────────────────────
        # Auto-create a state row for guests / unrecognised speakers so
        # the question still surfaces in the per-student summary card.
        if student_name and student_name not in self._student_state:
            self._student_state[student_name] = {
                "student": student_name,
                "attendance": "Present" if registered else (
                    "Guest" if student_name.startswith("Guest_") else "Unregistered"
                ),
                "emotion": "Analyzing...",
                "emotion_confidence": 0.0,
                "timestamp": timestamp,
                "registered": registered,
                "is_guest": student_name.startswith("Guest_"),
                "questions": [],
                "emotion_logged": False,
            }
        if student_name in self._student_state:
            self._student_state[student_name]["questions"].append({
                "question": question,
                "topic": topic,
                "classification_confidence": round(topic_confidence, 4),
                "timestamp": timestamp,
            })

        logger.info(
            "Question logged: %s asked '%s' → topic: %s (%.0f%%)",
            student_name,
            question[:50],
            topic,
            topic_confidence * 100,
        )
        print(f"\nSaved to logs.")
        return persisted_event

    def already_marked(self, name: str) -> bool:
        """Return ``True`` if the student was already marked present."""
        return name in self._marked

    def log_emotion(self, name: str, mood: str, samples: int) -> Optional[EventRecord]:
        """Record the final stable emotion for a tracked student.
        
        This separates the emotion logging from the initial attendance
        logging to avoid blocking the recognition pipeline.
        """
        state = self._student_state.get(name)
        if not state:
            return None
        if state.get("emotion_logged", False):
            return None
            
        registered = state.get("registered", False)
        persisted_event = self._log_service.log_emotion_event(
            student=name,
            mood=mood,
            registered=registered,
            samples=samples,
        )
        self._events.append(persisted_event)
        
        state["emotion"] = mood
        state["emotion_logged"] = True
        logger.info("Emotion finalised for %s: %s (samples=%d)", name, mood, samples)
        return persisted_event

    def get_student_state(self, name: str) -> Optional[AttendanceRecord]:
        """Get the current state for a specific student.

        Returns:
            The student's record with attendance + questions, or ``None``.
        """
        return self._student_state.get(name)

    def get_active_student(self) -> Optional[str]:
        """Get the most recently marked registered student's name.

        Useful for push-to-talk: when a question is asked, attribute
        it to the most recently seen/active student.

        Returns:
            Student name string, or ``None`` if no registered students.
        """
        # Return the last registered student from events
        for event in reversed(self._events):
            if event.get("event") == "attendance" and event.get("registered"):
                return event["student"]
        return None

    def save_log(self) -> Path:
        """Persist all events to the JSON log file.

        .. note::
            With the new :class:`LogService` backend, events are written
            immediately on each ``mark_attendance`` / ``add_question``
            call.  This method now exists for backward compatibility and
            is effectively a no-op — the log is already up-to-date.

        Returns:
            The ``Path`` to the log file.
        """
        # LogService already persists on every append_event() call.
        # Nothing additional to flush.
        return self.log_file

    def get_student_summary(self) -> List[AttendanceRecord]:
        """Get a summary of all students with their questions.

        Returns a list of per-student records, each containing the
        attendance info and their ``questions`` array.

        Returns:
            List of student summary dicts.
        """
        return list(self._student_state.values())

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def records(self) -> List[EventRecord]:
        """Current session's event records (chronological)."""
        return list(self._events)

    @property
    def marked_students(self) -> Set[str]:
        """Set of student names that have been marked present."""
        return set(self._marked)

    def reset_session(self) -> None:
        """Clear in-memory state AND wipe the log file for a fresh session.

        Called on server startup and via the ``reset-attendance`` API.
        """
        self._marked.clear()
        self._events.clear()
        self._student_state.clear()
        self._guest_counter = 0
        self._guest_names.clear()
        # Wipe the log file so the frontend's seedEventCursor
        # doesn't replay stale events from a prior run.
        self._log_service._write_events([])
        logger.info("Attendance session reset — log file cleared.")
