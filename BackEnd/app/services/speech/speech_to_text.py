"""
Speech-to-Text Service
=======================
Production-quality speech recognition using Google's Web Speech API.

Provides both a stateful ``SpeechRecognizer`` class and a backwards-compatible
``speech_to_text()`` convenience function.

Usage (standalone test)::

    python -m app.services.speech.speech_to_text

Dependencies::

    pip install SpeechRecognition PyAudio
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import Optional

import speech_recognition as sr

logger = logging.getLogger(__name__)

# ── Custom exceptions ────────────────────────────────────────────────


class SpeechError(Exception):
    """Base exception for speech recognition failures."""


class SpeechTimeoutError(SpeechError):
    """Raised when no speech is detected within the timeout window."""


class SpeechNotUnderstoodError(SpeechError):
    """Raised when audio was captured but could not be transcribed."""


class SpeechAPIError(SpeechError):
    """Raised when the Google Speech API request fails."""


# ── Result dataclass ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SpeechResult:
    """Structured result from a speech recognition attempt."""

    text: str
    language: str
    success: bool = True


# ── Main service class ───────────────────────────────────────────────


class SpeechRecognizer:
    """Microphone-based speech recognizer with configurable parameters.

    Args:
        language:           BCP-47 language tag for recognition (default ``en-US``).
        timeout:            Max seconds to wait for speech to *begin* (default 5).
        phrase_time_limit:  Max seconds for a single phrase/utterance (default 10).
        ambient_duration:   Seconds of silence used to calibrate noise floor (default 1).
    """

    def __init__(
        self,
        language: str = "en-US",
        timeout: int = 5,
        phrase_time_limit: int = 10,
        ambient_duration: float = 1.0,
    ) -> None:
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.ambient_duration = ambient_duration
        self._recognizer = sr.Recognizer()

    # ── Public API ───────────────────────────────────────────────────

    def listen_once(self) -> SpeechResult:
        """Record a single utterance from the microphone and transcribe it.

        This is the primary method used by push-to-talk: it opens the mic,
        waits for speech, records until silence or ``phrase_time_limit``,
        then sends the audio to Google's API.

        Returns:
            A ``SpeechResult`` with the transcribed text.

        Raises:
            SpeechTimeoutError:       No speech detected within ``timeout`` seconds.
            SpeechNotUnderstoodError: Audio captured but unintelligible.
            SpeechAPIError:           Google API returned an error.
        """
        print("\n[MIC STARTED]")
        logger.info("Microphone activated — listening for speech …")

        try:
            with sr.Microphone() as source:
                # Calibrate for ambient noise
                logger.debug(
                    "Calibrating for ambient noise (%.1fs) …", self.ambient_duration
                )
                self._recognizer.adjust_for_ambient_noise(
                    source, duration=self.ambient_duration
                )

                print("Listening...")
                logger.info("Listening (timeout=%ds, phrase_limit=%ds) …",
                            self.timeout, self.phrase_time_limit)

                try:
                    audio = self._recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_time_limit,
                    )
                except sr.WaitTimeoutError:
                    print("[MIC STOPPED] No speech detected (timeout)")
                    logger.warning("No speech detected within %ds timeout.", self.timeout)
                    raise SpeechTimeoutError(
                        f"No speech detected within {self.timeout}s."
                    )

        except SpeechTimeoutError:
            raise
        except OSError as exc:
            logger.error("Microphone hardware error: %s", exc)
            raise SpeechAPIError(f"Microphone error: {exc}") from exc

        # ── Send to Google Speech API ────────────────────────────────
        print("[MIC STOPPED] Processing audio...")
        logger.info("Sending audio to Google Speech API (language=%s) …", self.language)

        try:
            text: str = self._recognizer.recognize_google(
                audio,
                language=self.language,
            )
        except sr.UnknownValueError:
            logger.warning(
                "Google API could not understand the audio.", exc_info=True
            )
            raise SpeechNotUnderstoodError("Audio was not clear enough to transcribe.")
        except sr.RequestError as exc:
            logger.error(
                "Google Speech API request failed: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            raise SpeechAPIError(f"Google API error: {exc}") from exc

        text = text.strip()
        logger.info("Recognized text: %s", text)
        print(f"\nRecognized text:\n{text}")

        return SpeechResult(text=text, language=self.language)


# ── Audio-data (browser upload) transcription ───────────────────────


def transcribe_audio_data(
    audio: sr.AudioData,
    language: str = "en-US",
) -> SpeechResult:
    """Transcribe a pre-decoded :class:`sr.AudioData` via Google.

    Used by ``POST /api/v1/speech/transcribe-audio`` when the
    frontend uploads audio it captured itself (MediaRecorder) and the
    server side never opens its own microphone.

    Args:
        audio:    The decoded audio data.
        language: BCP-47 language tag.

    Returns:
        :class:`SpeechResult` containing the transcribed text.

    Raises:
        SpeechNotUnderstoodError: Audio captured but unintelligible.
        SpeechAPIError: Google API failure (network, blocked, …).
    """
    rec = sr.Recognizer()
    logger.info(
        "transcribe_audio_data: rate=%s width=%s frame_bytes=%d",
        audio.sample_rate,
        audio.sample_width,
        len(audio.frame_data),
    )
    try:
        text: str = rec.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        logger.warning(
            "Google API could not understand uploaded audio.", exc_info=True
        )
        raise SpeechNotUnderstoodError("Audio was not clear enough to transcribe.")
    except sr.RequestError as exc:
        logger.error(
            "Google Speech API request failed for uploaded audio: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise SpeechAPIError(f"Google API error: {exc}") from exc

    text = (text or "").strip()
    logger.info("Recognised uploaded audio: %s", text)
    return SpeechResult(text=text, language=language)


# ── Backwards-compatible convenience function ────────────────────────


def speech_to_text(
    language: str = "en-US",
    timeout: int = 5,
    phrase_time_limit: int = 10,
) -> Optional[str]:
    """Record from the microphone and return the transcribed text.

    This is a convenience wrapper around ``SpeechRecognizer.listen_once()``.
    Returns ``None`` on any failure (timeout, unclear audio, API error).

    Args:
        language:          BCP-47 language code (default ``en-US``).
        timeout:           Seconds to wait for speech to begin.
        phrase_time_limit: Max seconds for a single utterance.

    Returns:
        The recognized text as a lowercase string, or ``None`` on failure.
    """
    recognizer = SpeechRecognizer(
        language=language,
        timeout=timeout,
        phrase_time_limit=phrase_time_limit,
    )
    try:
        result = recognizer.listen_once()
        return result.text.lower().strip()
    except SpeechError as exc:
        logger.warning("Speech recognition failed: %s", exc)
        return None


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    )

    print("=" * 50)
    print("  Speech-to-Text — Manual Test")
    print("=" * 50)
    print("Speak a question in English.")
    print()

    recognizer = SpeechRecognizer(language="en-US", timeout=5, phrase_time_limit=10)

    try:
        result = recognizer.listen_once()
        print(f"\n✅ Recognized: {result.text}")
        print(f"   Language:   {result.language}")
    except SpeechTimeoutError:
        print("\n[ERROR] Timeout — no speech detected.")
    except SpeechNotUnderstoodError:
        print("\n[ERROR] Could not understand the audio.")
    except SpeechAPIError as exc:
        print(f"\n[ERROR] API error: {exc}")