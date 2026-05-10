"""
Question Pipeline — Speech + NLP Orchestrator
===============================================
Coordinates the speech-to-text and NLP classification modules into
a single end-to-end question processing pipeline.

Usage::

    from app.services.orchestrator.question_pipeline import QuestionPipeline

    pipeline = QuestionPipeline()

    # From microphone (push-to-talk)
    result = pipeline.process_voice_question()

    # From text (testing / API)
    result = pipeline.process_text_question("What is a semaphore?")

Result format::

    {
        "question": "What is a semaphore?",
        "topic": "Operating System",
        "topic_confidence": 0.87,
        "timestamp": "2026-05-10T01:40:00+00:00"
    }

Event logging:
    When ``log_events=True`` (the default), every classified question
    is also appended to the shared classroom log via :class:`LogService`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.services.nlp.Question_Classification import (
    predict_topic_with_confidence,
)
from app.services.speech.speech_to_text import (
    SpeechError,
    SpeechRecognizer,
    SpeechResult,
)

logger = logging.getLogger(__name__)

# Type alias
QuestionResult = Dict[str, Any]


from app.core.config import settings
from app.services.logging.log_service import LogService

class QuestionPipeline:
    """End-to-end question processing: microphone → text → topic.

    Args:
        language:           Speech recognition language (default ``en-US``).
        timeout:            Seconds to wait for speech to begin (default 5).
        phrase_time_limit:  Max seconds for a single utterance (default 10).
        nlp_model_dir:      Path to the directory containing ``nlp_pipeline.joblib``.
        log_events:         If ``True``, every classified question is appended
                            to the shared classroom log via :class:`LogService`.
        student_name:       Name to attribute questions to (default
                            ``"Unknown"``).
        source:             Source label for event logging (default ``None``).
    """

    def __init__(
        self,
        language: str = "en-US",
        timeout: int = 5,
        phrase_time_limit: int = 10,
        nlp_model_dir: Union[str, Path, None] = None,
        log_events: bool = True,
        student_name: str = "Unknown",
        source: Optional[str] = None,
    ) -> None:
        self.speech_recognizer = SpeechRecognizer(
            language=language,
            timeout=timeout,
            phrase_time_limit=phrase_time_limit,
        )
        self.nlp_model_dir = nlp_model_dir or settings.NLP_MODEL_DIR

        # Event logging
        self._log_events = log_events
        self._log_service = LogService() if log_events else None
        self._student_name = student_name
        self._source = source

    # ── Public API ───────────────────────────────────────────────────

    def process_voice_question(self) -> Optional[QuestionResult]:
        """Record from microphone, transcribe, and classify the topic.

        Returns:
            A ``QuestionResult`` dict on success, or ``None`` if speech
            recognition failed (timeout, unclear audio, API error).
        """
        logger.info("Starting voice question pipeline …")

        try:
            speech_result: SpeechResult = self.speech_recognizer.listen_once()
        except SpeechError as exc:
            logger.warning("Speech recognition failed: %s", exc)
            print(f"\n[ERROR] Speech failed: {exc}")
            return None

        return self._classify(speech_result.text)

    def process_text_question(self, text: str) -> QuestionResult:
        """Classify a pre-existing text question (no microphone needed).

        This is useful for testing, API endpoints, or typed questions.

        Args:
            text: The question text to classify.

        Returns:
            A ``QuestionResult`` dict.
        """
        logger.info("Processing text question: %s", text)
        return self._classify(text)

    # ── Internal ─────────────────────────────────────────────────────

    def _classify(self, question_text: str) -> QuestionResult:
        """Run NLP classification and build the result dict."""
        topic, confidence = predict_topic_with_confidence(
            question_text,
            model_dir=self.nlp_model_dir,
        )

        result: QuestionResult = {
            "question": question_text,
            "topic": topic,
            "topic_confidence": round(confidence, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        print(f"\nPredicted topic:\n{topic} ({confidence:.0%})")
        logger.info("Question classified: '%s' -> %s (%.2f)", question_text, topic, confidence)

        # ── Persist question event via LogService ────────────────────
        if self._log_events and self._log_service is not None:
            self._log_service.log_question_event(
                student=self._student_name,
                question=question_text,
                topic=topic,
                classification_confidence=confidence,
                registered=False if self._student_name in ("Unknown", "Manual_Test_User") else None,
                source=self._source,
                timestamp=result["timestamp"],
            )
            print(f"\nSaved to logs.")

        return result


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    )

    print("=" * 50)
    print("  Question Pipeline — Voice + NLP Test")
    print("=" * 50)
    print()

    pipeline = QuestionPipeline(
        student_name="Manual_Test_User",
        source="question_pipeline_cli",
    )

    # Text test first
    print("--- Text-only test ---")
    result = pipeline.process_text_question("How does the sliding window protocol work?")
    print(f"Result: {result}")
    print()

    # Voice test
    print("--- Voice test ---")
    print("Speak a question in English:")
    result = pipeline.process_voice_question()
    if result:
        print(f"\nFull result: {result}")
    else:
        print("\nNo question captured.")
