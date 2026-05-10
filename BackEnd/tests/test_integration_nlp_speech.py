"""
Tests — Speech + NLP Integration
==================================
Integration tests for the QuestionPipeline orchestrator.
Uses mocked speech input (no real microphone needed).

Run::

    cd BackEnd
    pytest tests/test_integration_nlp_speech.py -v

Manual (needs mic)::

    python tests/test_integration_nlp_speech.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure BackEnd/ is on sys.path ────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.nlp.Question_Classification import clear_cache, train_and_save
from app.services.orchestrator.question_pipeline import QuestionPipeline
from app.services.speech.speech_to_text import (
    SpeechNotUnderstoodError,
    SpeechResult,
    SpeechTimeoutError,
)

import speech_recognition as sr


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Train the NLP pipeline once for the entire test module."""
    clear_cache()
    model_dir = tmp_path_factory.mktemp("nlp_models")
    train_and_save(model_dir=model_dir)
    return str(model_dir)


@pytest.fixture
def pipeline(trained_model) -> QuestionPipeline:
    """Create a QuestionPipeline pointing to the temp-trained model."""
    return QuestionPipeline(nlp_model_dir=trained_model)


# ── Text-only pipeline tests ────────────────────────────────────────


class TestTextPipeline:
    """Test process_text_question() — no microphone needed."""

    def test_returns_dict_with_required_keys(self, pipeline):
        result = pipeline.process_text_question("What is a semaphore?")
        assert isinstance(result, dict)
        assert "question" in result
        assert "topic" in result
        assert "topic_confidence" in result
        assert "timestamp" in result

    def test_question_text_preserved(self, pipeline):
        question = "How does the sliding window protocol work?"
        result = pipeline.process_text_question(question)
        assert result["question"] == question

    def test_topic_is_string(self, pipeline):
        result = pipeline.process_text_question("What is virtual memory?")
        assert isinstance(result["topic"], str)
        assert len(result["topic"]) > 0

    def test_confidence_in_range(self, pipeline):
        result = pipeline.process_text_question("Explain binary search")
        assert 0.0 <= result["topic_confidence"] <= 1.0

    @pytest.mark.parametrize(
        "question, expected_topic",
        [
            ("How does the sliding window protocol work?", "Computer Networks"),
            ("What is the probability of rolling a 6?", "Mathematics"),
            ("Explain process synchronization with semaphores", "Operating System"),
        ],
    )
    def test_correct_topic_prediction(self, pipeline, question, expected_topic):
        result = pipeline.process_text_question(question)
        assert result["topic"] == expected_topic, (
            f"Expected '{expected_topic}', got '{result['topic']}' for: {question}"
        )


# ── Voice pipeline tests (mocked microphone) ────────────────────────


class TestVoicePipeline:
    """Test process_voice_question() with mocked speech recognition."""

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(
        sr.Recognizer,
        "recognize_google",
        return_value="How does the sliding window protocol work",
    )
    @patch.object(sr.Recognizer, "listen", return_value=MagicMock())
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_voice_to_topic_success(
        self, mock_ambient, mock_listen, mock_google, mock_mic, pipeline
    ):
        """Full voice pipeline: mic → text → NLP → result."""
        result = pipeline.process_voice_question()

        assert result is not None
        assert result["question"] == "How does the sliding window protocol work"
        assert isinstance(result["topic"], str)
        assert len(result["topic"]) > 0

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "listen", side_effect=sr.WaitTimeoutError("timeout"))
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_voice_timeout_returns_none(
        self, mock_ambient, mock_listen, mock_mic, pipeline
    ):
        """When speech times out, process_voice_question returns None in legacy mode."""
        result = pipeline.process_voice_question(raise_on_error=False)
        assert result is None

    @patch("app.services.speech.speech_to_text.sr.Microphone")
    @patch.object(sr.Recognizer, "recognize_google", side_effect=sr.UnknownValueError())
    @patch.object(sr.Recognizer, "listen", return_value=MagicMock())
    @patch.object(sr.Recognizer, "adjust_for_ambient_noise")
    def test_voice_unclear_returns_none(
        self, mock_ambient, mock_listen, mock_google, mock_mic, pipeline
    ):
        """When audio is unclear, process_voice_question returns None in legacy mode."""
        result = pipeline.process_voice_question(raise_on_error=False)
        assert result is None


# ── Manual test entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    )

    print("=" * 50)
    print("  Integration Test — Speech + NLP")
    print("=" * 50)

    pipeline = QuestionPipeline()

    # 1) Text-only test
    print("\n--- Text Test ---")
    result = pipeline.process_text_question("How does TCP handshake work?")
    if result:
        print(f"  Question: {result['question']}")
        print(f"  Topic:    {result['topic']} ({result['topic_confidence']:.0%})")

    # 2) Voice test
    print("\n--- Voice Test ---")
    print("🎤 Speak a question in English:")
    result = pipeline.process_voice_question()
    if result:
        print(f"\n  Question: {result['question']}")
        print(f"  Topic:    {result['topic']} ({result['topic_confidence']:.0%})")
    else:
        print("\n  ❌ No question captured.")