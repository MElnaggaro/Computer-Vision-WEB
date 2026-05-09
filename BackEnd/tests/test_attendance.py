"""
Tests — Attendance Service
==========================
Covers:
    • Single attendance marking
    • Duplicate attendance prevention
    • Unknown face handling (no duplicate blocking)
    • JSON log creation and persistence
    • Session reset
"""

import sys
from pathlib import Path

# ── Ensure BackEnd/ is on sys.path when run directly ─────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import json

import pytest

from app.services.vision.attendance_service import AttendanceService


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def log_file(tmp_path: Path) -> Path:
    """Provide a temporary log file path."""
    return tmp_path / "test_attendance.json"


@pytest.fixture()
def service(log_file: Path) -> AttendanceService:
    """Provide a fresh AttendanceService writing to a temp file."""
    return AttendanceService(log_file=log_file)


# ── Mark Attendance ──────────────────────────────────────────────────

class TestMarkAttendance:
    """Tests for the ``mark_attendance`` method."""

    def test_mark_known_student(self, service: AttendanceService) -> None:
        """Marking a known student returns a valid record."""
        record = service.mark_attendance("Mohammed_Ayman", known=True, confidence=0.92)

        assert record is not None
        assert record["student"] == "Mohammed_Ayman"
        assert record["attendance"] == "Present"
        assert record["known"] is True
        assert record["confidence"] == 0.92
        assert "timestamp" in record

    def test_mark_unknown_face(self, service: AttendanceService) -> None:
        """Marking an unknown face returns 'Not Registered'."""
        record = service.mark_attendance("Unknown", known=False, confidence=0.21)

        assert record is not None
        assert record["student"] == "Unknown"
        assert record["attendance"] == "Not Registered"
        assert record["known"] is False

    def test_multiple_unknowns_all_logged(self, service: AttendanceService) -> None:
        """Multiple unknown detections should ALL be logged (not de‑duped)."""
        r1 = service.mark_attendance("Unknown", known=False, confidence=0.15)
        r2 = service.mark_attendance("Unknown", known=False, confidence=0.10)
        r3 = service.mark_attendance("Unknown", known=False, confidence=0.18)

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None
        assert len(service.records) == 3


# ── Duplicate Prevention ─────────────────────────────────────────────

class TestDuplicatePrevention:
    """Tests for the duplicate‑attendance guard."""

    def test_duplicate_known_student_returns_none(
        self, service: AttendanceService
    ) -> None:
        """Second call for the same known student should return None."""
        first = service.mark_attendance("Noreen_Osama", known=True, confidence=0.88)
        second = service.mark_attendance("Noreen_Osama", known=True, confidence=0.90)

        assert first is not None
        assert second is None

    def test_already_marked_true_after_first(
        self, service: AttendanceService
    ) -> None:
        """``already_marked`` should be True after the first call."""
        service.mark_attendance("Catherine_Adel", known=True, confidence=0.85)
        assert service.already_marked("Catherine_Adel") is True

    def test_already_marked_false_initially(
        self, service: AttendanceService
    ) -> None:
        """``already_marked`` should be False before any call."""
        assert service.already_marked("Anyone") is False

    def test_different_students_not_blocked(
        self, service: AttendanceService
    ) -> None:
        """Different known students should each get their own record."""
        r1 = service.mark_attendance("Student_A", known=True, confidence=0.9)
        r2 = service.mark_attendance("Student_B", known=True, confidence=0.85)

        assert r1 is not None
        assert r2 is not None
        assert len(service.marked_students) == 2

    def test_many_duplicate_frames(self, service: AttendanceService) -> None:
        """Simulating many frames – only the first marking should count."""
        results = []
        for _ in range(50):
            r = service.mark_attendance("Rewan_Mosad", known=True, confidence=0.91)
            results.append(r)

        marked = [r for r in results if r is not None]
        assert len(marked) == 1
        assert marked[0]["student"] == "Rewan_Mosad"


# ── JSON Log Persistence ────────────────────────────────────────────

class TestLogPersistence:
    """Tests for ``save_log`` and the generated JSON file."""

    def test_save_creates_file(
        self, service: AttendanceService, log_file: Path
    ) -> None:
        """``save_log`` should create the attendance JSON file."""
        service.mark_attendance("Menna_Abdo", known=True, confidence=0.89)
        result_path = service.save_log()

        assert result_path == log_file
        assert log_file.exists()

    def test_log_contains_correct_records(
        self, service: AttendanceService, log_file: Path
    ) -> None:
        """The JSON file should contain exactly the marked records."""
        service.mark_attendance("Student_X", known=True, confidence=0.95)
        service.mark_attendance("Unknown", known=False, confidence=0.12)
        service.save_log()

        with open(log_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["student"] == "Student_X"
        assert data[0]["attendance"] == "Present"
        assert data[1]["student"] == "Unknown"
        assert data[1]["attendance"] == "Not Registered"

    def test_log_merges_with_existing(
        self, service: AttendanceService, log_file: Path
    ) -> None:
        """Saving should append to existing log entries, not overwrite."""
        # Pre‑populate the file
        existing = [{"student": "Old_Student", "attendance": "Present",
                      "timestamp": "2026-01-01T00:00:00+00:00",
                      "known": True, "confidence": 0.99}]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as fh:
            json.dump(existing, fh)

        service.mark_attendance("New_Student", known=True, confidence=0.88)
        service.save_log()

        with open(log_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert len(data) == 2
        assert data[0]["student"] == "Old_Student"
        assert data[1]["student"] == "New_Student"

    def test_empty_log_creates_empty_list(
        self, service: AttendanceService, log_file: Path
    ) -> None:
        """Saving with no records should still produce valid JSON."""
        service.save_log()

        with open(log_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert data == []


# ── Session Reset ────────────────────────────────────────────────────

class TestSessionReset:
    """Tests for ``reset_session``."""

    def test_reset_clears_marked(self, service: AttendanceService) -> None:
        """After reset, previously marked students should be markable again."""
        service.mark_attendance("Student_A", known=True, confidence=0.9)
        assert service.already_marked("Student_A") is True

        service.reset_session()

        assert service.already_marked("Student_A") is False
        assert len(service.records) == 0

    def test_reset_allows_remarking(self, service: AttendanceService) -> None:
        """After reset, the same student can be marked again."""
        service.mark_attendance("Student_A", known=True, confidence=0.9)
        service.reset_session()

        record = service.mark_attendance("Student_A", known=True, confidence=0.88)
        assert record is not None
        assert record["student"] == "Student_A"
