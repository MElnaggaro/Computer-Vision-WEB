"""
Attendance Service
==================
Tracks which students have been marked present during the current
session and persists structured JSON attendance logs.

Design decisions:
    • Uses an in‑memory ``set`` for O(1) duplicate checks.
    • Each session writes a *list* of attendance records to the log file.
    • Unknown faces are logged with ``"attendance": "Not Registered"``
      so that the future admin‑approval flow can cross‑reference them.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings

logger = logging.getLogger(__name__)

AttendanceRecord = Dict[str, Any]


class AttendanceService:
    """Mark attendance, prevent duplicates, and persist JSON logs."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
    ) -> None:
        self.log_file = log_file or settings.ATTENDANCE_LOG_FILE
        self._marked: Set[str] = set()          # student names marked so far
        self._records: List[AttendanceRecord] = []
        self._has_unsaved_changes: bool = False
        
        # Load students already present in the log file to prevent duplicates across runs
        self._load_existing_students()

    def _load_existing_students(self) -> None:
        """Populate the marked set from the existing log file."""
        if not self.log_file.exists():
            return
        try:
            with open(self.log_file, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for record in data:
                            name = record.get("student")
                            registered = record.get("registered", False)
                            if name and registered and name != "Unknown":
                                self._marked.add(name)
            logger.info("Loaded %d previously marked students.", len(self._marked))
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not pre-load attendance log: %s", exc)

    # ── Public API ───────────────────────────────────────────────────

    def mark_attendance(
        self,
        name: str,
        registered: bool,
        similarity: float,
    ) -> Optional[AttendanceRecord]:
        """Record a student's attendance if not already marked.

        Args:
            name:       Student name or ``"Unknown"``.
            registered: Whether the student was recognised.
            similarity: Recognition similarity (0–1).

        Returns:
            The ``AttendanceRecord`` dict if newly marked, or ``None``
            if attendance was already recorded for this student.
        """
        # Skip duplicates for registered students
        if registered and self.already_marked(name):
            logger.debug("Attendance already marked for %s – skipping.", name)
            return None

        record: AttendanceRecord = {
            "student": name,
            "attendance": "Present",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registered": registered,
        }

        if registered:
            self._marked.add(name)

        self._records.append(record)
        self._has_unsaved_changes = True
        logger.info(
            "Attendance marked: %s (Present, registered=%s)",
            name,
            registered,
        )
        return record

    def already_marked(self, name: str) -> bool:
        """Return ``True`` if the student was already marked present."""
        return name in self._marked

    def save_log(self) -> Path:
        """Persist all attendance records to a JSON file.

        Returns:
            The ``Path`` to the written log file.
        """
        if not self._has_unsaved_changes and self.log_file.exists():
            return self.log_file

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Merge with any existing log entries
        existing: List[AttendanceRecord] = []
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                    if content:
                        existing = json.loads(content)
                        if not isinstance(existing, list):
                            existing = []
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Could not read existing log – overwriting: %s", exc)

        combined = existing + self._records

        with open(self.log_file, "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, ensure_ascii=False)

        self._has_unsaved_changes = False
        logger.info("Saved %d records to %s", len(self._records), self.log_file)
        return self.log_file

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def records(self) -> List[AttendanceRecord]:
        """Current session's attendance records."""
        return list(self._records)

    @property
    def marked_students(self) -> Set[str]:
        """Set of student names that have been marked present."""
        return set(self._marked)

    def reset_session(self) -> None:
        """Clear in‑memory state for a fresh attendance session."""
        self._marked.clear()
        self._records.clear()
        logger.info("Attendance session reset.")
