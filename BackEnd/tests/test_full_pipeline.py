"""
Tests — Full Pipeline (Attendance + Questions + Logging)
=========================================================
End-to-end tests verifying event-based logging, question attachment,
and log record schema validation.

Run::

    cd BackEnd
    pytest tests/test_full_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure BackEnd/ is on sys.path ────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.nlp.Question_Classification import clear_cache, train_and_save
from app.services.vision.attendance_service import AttendanceService


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def log_file(tmp_path) -> Path:
    """Provide a temp log file path."""
    return tmp_path / "test_log.json"


@pytest.fixture
def attendance(log_file) -> AttendanceService:
    """Create an AttendanceService with a temp log file."""
    return AttendanceService(log_file=log_file)


# ── Removed trained_model fixture to avoid CV errors on small datasets ──


# ── Test 1 — Event-based log format ─────────────────────────────────


class TestEventBasedLogging:
    """Verify the event-based log format."""

    def test_attendance_event_has_correct_type(self, attendance):
        """Marking attendance creates an 'attendance' event."""
        attendance.mark_attendance(
            name="Ahmed_Ali",
            registered=True,
            similarity=0.85,
            emotion="Happy",
            emotion_confidence=0.92,
        )
        events = attendance.records
        assert len(events) == 1
        assert events[0]["event"] == "attendance"

    def test_attendance_event_schema(self, attendance):
        """Attendance event has all required fields."""
        attendance.mark_attendance(
            name="Ahmed_Ali",
            registered=True,
            similarity=0.85,
        )
        event = attendance.records[0]

        required_keys = {"event", "student", "attendance", "timestamp", "registered"}
        assert required_keys.issubset(event.keys())
        assert event["student"] == "Ahmed_Ali"
        assert event["attendance"] == "Present"
        assert event["registered"] is True

    def test_question_event_has_correct_type(self, attendance):
        """Adding a question creates a 'question' event."""
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        attendance.add_question(
            student_name="Ahmed_Ali",
            question="What is a semaphore?",
            topic="Operating System",
            topic_confidence=0.87,
        )
        events = attendance.records
        assert len(events) == 2
        assert events[0]["event"] == "attendance"
        assert events[1]["event"] == "question"

    def test_question_event_schema(self, attendance):
        """Question event has all required fields."""
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        attendance.add_question(
            student_name="Ahmed_Ali",
            question="What is a semaphore?",
            topic="Operating System",
            topic_confidence=0.87,
            source="webcam_push_to_talk",
        )
        event = attendance.records[1]

        required_keys = {
            "event", "student", "question", "topic", 
            "classification_confidence", "timestamp",
            "registered", "source"
        }
        assert required_keys.issubset(event.keys())
        assert event["student"] == "Ahmed_Ali"
        assert event["question"] == "What is a semaphore?"
        assert event["topic"] == "Operating System"
        assert event["classification_confidence"] == 0.87
        assert event["registered"] is True
        assert event["source"] == "webcam_push_to_talk"


# ── Test 2 — Multi-question support ─────────────────────────────────


class TestMultipleQuestions:
    """Verify that students can ask multiple questions."""

    def test_multiple_questions_in_event_log(self, attendance):
        """Each question creates a separate event entry."""
        attendance.mark_attendance(
            name="Mohammed_Ayman", registered=True, similarity=0.9
        )
        attendance.add_question(
            student_name="Mohammed_Ayman",
            question="Explain convolution",
            topic="Computer Networks",
            topic_confidence=0.85,
        )
        attendance.add_question(
            student_name="Mohammed_Ayman",
            question="What is regression?",
            topic="Mathematics",
            topic_confidence=0.78,
        )

        events = attendance.records
        assert len(events) == 3  # 1 attendance + 2 questions

        question_events = [e for e in events if e["event"] == "question"]
        assert len(question_events) == 2
        assert question_events[0]["question"] == "Explain convolution"
        assert question_events[1]["question"] == "What is regression?"

    def test_multiple_questions_in_student_state(self, attendance):
        """Student state maintains a questions array."""
        attendance.mark_attendance(
            name="Mohammed_Ayman", registered=True, similarity=0.9
        )
        attendance.add_question(
            student_name="Mohammed_Ayman",
            question="Explain convolution",
            topic="Computer Networks",
        )
        attendance.add_question(
            student_name="Mohammed_Ayman",
            question="What is regression?",
            topic="Mathematics",
        )

        state = attendance.get_student_state("Mohammed_Ayman")
        assert state is not None
        assert "questions" in state
        assert len(state["questions"]) == 2
        assert state["questions"][0]["question"] == "Explain convolution"
        assert state["questions"][1]["question"] == "What is regression?"


# ── Test 3 — Per-student state ───────────────────────────────────────


class TestStudentState:
    """Verify per-student state management."""

    def test_student_state_has_questions_array(self, attendance):
        """New students start with an empty questions array."""
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        state = attendance.get_student_state("Ahmed_Ali")
        assert state is not None
        assert state["questions"] == []

    def test_unknown_student_has_no_state(self, attendance):
        """Non-existent student returns None."""
        assert attendance.get_student_state("NonExistent") is None

    def test_student_summary_returns_all_students(self, attendance):
        """get_student_summary() returns all tracked students."""
        attendance.mark_attendance(
            name="Student_A", registered=True, similarity=0.85
        )
        attendance.mark_attendance(
            name="Student_B", registered=True, similarity=0.90
        )
        summary = attendance.get_student_summary()
        assert len(summary) == 2
        names = {s["student"] for s in summary}
        assert names == {"Student_A", "Student_B"}


# ── Test 4 — Log persistence ────────────────────────────────────────


class TestLogPersistence:
    """Verify JSON log file reading and writing."""

    def test_save_creates_json_file(self, attendance, log_file):
        """save_log() creates a valid JSON file."""
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85, emotion="Happy"
        )
        attendance.save_log()

        assert log_file.exists()
        data = json.loads(log_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1

    def test_save_includes_question_events(self, attendance, log_file):
        """Both attendance and question events are persisted."""
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        attendance.add_question(
            student_name="Ahmed_Ali",
            question="What is TCP?",
            topic="Computer Networks",
        )
        attendance.save_log()

        data = json.loads(log_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        types = [d["event"] for d in data]
        assert "attendance" in types
        assert "question" in types

    def test_duplicate_attendance_prevented(self, attendance):
        """Same student should not be marked twice."""
        result1 = attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        result2 = attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        assert result1 is not None
        assert result2 is None  # duplicate

    def test_emotion_in_log(self, attendance, log_file):
        """emotion is persisted in the log."""
        attendance.mark_attendance(
            name="Ahmed_Ali",
            registered=True,
            similarity=0.85,
        )
        attendance.log_emotion("Ahmed_Ali", "Tired", 5)
        attendance.save_log()

        data = json.loads(log_file.read_text(encoding="utf-8"))
        assert data[1]["event"] == "emotion"
        assert data[1]["mood"] == "Tired"
        assert data[1]["samples"] == 5


# ── Test 5 — Session reset ──────────────────────────────────────────


class TestSessionReset:
    """Verify session reset clears all state."""

    def test_reset_clears_marked_students(self, attendance):
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        assert len(attendance.marked_students) == 1

        attendance.reset_session()
        assert len(attendance.marked_students) == 0

    def test_reset_clears_events(self, attendance):
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        attendance.reset_session()
        assert len(attendance.records) == 0

    def test_reset_clears_student_state(self, attendance):
        attendance.mark_attendance(
            name="Ahmed_Ali", registered=True, similarity=0.85
        )
        attendance.reset_session()
        assert attendance.get_student_state("Ahmed_Ali") is None


# ── Test 6 — Unknown student handling ────────────────────────────────


class TestUnknownStudents:
    """Verify Unknown student behavior."""

    def test_unknown_not_marked_as_present(self, attendance):
        attendance.mark_attendance(
            name="Unknown", registered=False, similarity=0.0
        )
        assert "Unknown" not in attendance.marked_students

    def test_unknown_logged_as_not_registered(self, attendance):
        attendance.mark_attendance(
            name="Unknown", registered=False, similarity=0.0
        )
        events = attendance.records
        assert events[0]["attendance"] == "Not Registered"
        assert events[0]["registered"] is False


# ── Test 7 — Full integration with NLP ───────────────────────────────


class TestFullIntegrationWithNLP:
    """End-to-end: attendance + NLP question classification + logging."""

    @patch("app.services.orchestrator.question_pipeline.predict_topic_with_confidence")
    def test_attendance_then_question_classification(
        self, mock_predict, attendance, log_file
    ):
        """Simulate: student appears → attendance → asks question → log saved."""
        from app.services.orchestrator.question_pipeline import QuestionPipeline

        mock_predict.return_value = ("Computer Networks", 0.95)

        # 1. Mark attendance
        attendance.mark_attendance(
            name="Mohammed_Ayman",
            registered=True,
            similarity=0.92,
        )
        
        # Log emotion separately
        attendance.log_emotion("Mohammed_Ayman", "Happy", 5)

        # 2. Student asks a question (text-only, no mic)
        pipeline = QuestionPipeline(log_events=False)
        result = pipeline.process_text_question(
            "How does the sliding window protocol work?"
        )

        # 3. Log the question (AttendanceService logs via LogService)
        attendance.add_question(
            student_name="Mohammed_Ayman",
            question=result["question"],
            topic=result["topic"],
            topic_confidence=result["topic_confidence"],
        )

        # 4. Save and verify
        attendance.save_log()

        data = json.loads(log_file.read_text(encoding="utf-8"))
        assert len(data) == 3

        # Attendance event
        assert data[0]["event"] == "attendance"
        assert data[0]["student"] == "Mohammed_Ayman"

        # Emotion event
        assert data[1]["event"] == "emotion"
        assert data[1]["student"] == "Mohammed_Ayman"
        assert data[1]["mood"] == "Happy"

        # Question event
        assert data[2]["event"] == "question"
        assert data[2]["student"] == "Mohammed_Ayman"
        assert data[2]["question"] == "How does the sliding window protocol work?"
        assert data[2]["topic"] == "Computer Networks"
        assert data[2]["classification_confidence"] > 0.0

        # Student summary (internal state still uses classification_confidence)
        state = attendance.get_student_state("Mohammed_Ayman")
        assert len(state["questions"]) == 1
        assert state["questions"][0]["topic"] == "Computer Networks"
        assert "classification_confidence" in state["questions"][0]
