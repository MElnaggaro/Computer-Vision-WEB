"""
Audio Decoder — Browser-Friendly Audio Pipeline
================================================
Converts raw audio bytes uploaded by the browser into the
``speech_recognition.AudioData`` form that the Google recogniser
expects.

The dashboard records audio via :class:`MediaRecorder` which on
Chrome / Edge produces ``audio/webm;codecs=opus`` chunks.  Firefox
prefers ``audio/ogg;codecs=opus``.  The Python ``speech_recognition``
library only knows how to decode WAV/AIFF/FLAC natively, so we have
two paths:

1. **Fast path** — payload is already a WAV/AIFF/FLAC file.  We hand
   it directly to :class:`speech_recognition.AudioFile` and read it
   in-process.

2. **Conversion path** — payload is anything else (WebM, OGG, MP3,
   M4A, raw Opus, …).  We shell out to ``ffmpeg`` to transcode to a
   16-bit PCM WAV at 16 kHz mono and then read that WAV.

Each step is logged with structured fields so the ``/speech/debug``
endpoint can report exactly which stage produced an error.

This module is intentionally side-effect free at import time and
exposes a single :func:`decode_audio` function.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import speech_recognition as sr

logger = logging.getLogger(__name__)


class AudioDecodeError(Exception):
    """Raised when the uploaded payload cannot be turned into AudioData."""


@dataclass(frozen=True)
class DecodeReport:
    """Diagnostic record describing how an audio payload was decoded."""

    success: bool
    used_path: str           # "native" | "ffmpeg" | "n/a"
    bytes_in: int
    sample_rate: Optional[int]
    sample_width: Optional[int]
    duration_ms: Optional[int]
    detected_format: Optional[str]
    error: Optional[str] = None


# ── ffmpeg discovery ────────────────────────────────────────────────


_FFMPEG_CACHED: Optional[str] = None


def ffmpeg_path() -> Optional[str]:
    """Return the absolute path of ``ffmpeg`` on this host or ``None``."""
    global _FFMPEG_CACHED
    if _FFMPEG_CACHED is not None:
        return _FFMPEG_CACHED or None
    found = shutil.which("ffmpeg")
    _FFMPEG_CACHED = found or ""
    return found


# ── Format sniffing ─────────────────────────────────────────────────


def sniff_format(data: bytes) -> str:
    """Best-effort format detection from the leading magic bytes.

    Returns one of: ``wav``, ``flac``, ``aiff``, ``ogg``, ``webm``,
    ``mp3``, ``m4a``, or ``unknown``.  This is purely informational —
    the actual decode is attempted regardless.
    """
    if len(data) < 16:
        return "unknown"
    head = data[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"FORM") and (head[8:12] == b"AIFF" or head[8:12] == b"AIFC"):
        return "aiff"
    if head.startswith(b"OggS"):
        return "ogg"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"  # EBML container — Matroska/WebM
    if head[:3] == b"ID3" or head[0] == 0xFF:
        return "mp3"
    if head[4:8] == b"ftyp":
        return "m4a"
    return "unknown"


# ── Core decoder ────────────────────────────────────────────────────


def _decode_native(data: bytes) -> Tuple[sr.AudioData, int, int, int]:
    """Decode a WAV/AIFF/FLAC payload via :class:`sr.AudioFile`.

    Returns ``(audio_data, sample_rate, sample_width, duration_ms)``.
    """
    rec = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(data)) as source:
        audio = rec.record(source)
    duration_ms = int((len(audio.frame_data) / audio.sample_width / audio.sample_rate) * 1000)
    return audio, audio.sample_rate, audio.sample_width, duration_ms


def _decode_via_ffmpeg(data: bytes) -> Tuple[sr.AudioData, int, int, int]:
    """Transcode arbitrary audio to 16-bit PCM WAV at 16 kHz mono via ffmpeg.

    Raises:
        AudioDecodeError: ffmpeg missing or returned a non-zero exit
            code.  The captured stderr is included in the message so
            ``/speech/debug`` can report it verbatim.
    """
    bin_path = ffmpeg_path()
    if not bin_path:
        raise AudioDecodeError(
            "ffmpeg not found on PATH — cannot decode this audio format. "
            "Install ffmpeg or upload WAV/FLAC instead."
        )

    with tempfile.TemporaryDirectory(prefix="scl_speech_") as tmp:
        in_path = Path(tmp) / "input.bin"
        out_path = Path(tmp) / "output.wav"
        in_path.write_bytes(data)

        cmd = [
            bin_path,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(in_path),
            "-ac", "1",            # mono
            "-ar", "16000",        # 16 kHz — what Google expects
            "-acodec", "pcm_s16le",
            str(out_path),
        ]
        logger.debug("ffmpeg cmd: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioDecodeError(
                f"ffmpeg timed out after 30 s while decoding "
                f"({len(data)} bytes)"
            ) from exc

        if result.returncode != 0 or not out_path.exists():
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise AudioDecodeError(
                f"ffmpeg failed (rc={result.returncode}): {stderr or 'no stderr'}"
            )

        wav_bytes = out_path.read_bytes()

    return _decode_native(wav_bytes)


def decode_audio(data: bytes) -> Tuple[sr.AudioData, DecodeReport]:
    """Convert raw audio bytes into a ``sr.AudioData`` ready for transcription.

    Args:
        data: The full audio payload.  May be WAV, FLAC, AIFF, WebM,
              OGG/Opus, MP3, M4A, …

    Returns:
        A pair ``(audio_data, report)`` where ``report`` is a
        :class:`DecodeReport` describing what happened — useful for
        diagnostic responses.

    Raises:
        AudioDecodeError: When neither the native nor the ffmpeg path
            succeeded.  The exception message is safe to surface to
            the frontend during debugging.
    """
    if not data:
        return None, DecodeReport(
            success=False,
            used_path="n/a",
            bytes_in=0,
            sample_rate=None,
            sample_width=None,
            duration_ms=None,
            detected_format=None,
            error="empty payload",
        )

    fmt = sniff_format(data)
    logger.info(
        "decode_audio: %d bytes, sniffed=%s, ffmpeg=%s",
        len(data),
        fmt,
        bool(ffmpeg_path()),
    )

    # Fast path — native decode (WAV / FLAC / AIFF).
    if fmt in ("wav", "flac", "aiff"):
        try:
            audio, rate, width, duration_ms = _decode_native(data)
            return audio, DecodeReport(
                success=True,
                used_path="native",
                bytes_in=len(data),
                sample_rate=rate,
                sample_width=width,
                duration_ms=duration_ms,
                detected_format=fmt,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Native decode of %s payload failed (%d bytes): %s",
                fmt, len(data), exc, exc_info=True,
            )
            # Fall through to ffmpeg — sometimes recorded WAVs have
            # quirks (RF64, oversize chunks) that ffmpeg can salvage.

    # Conversion path — ffmpeg transcode.
    try:
        audio, rate, width, duration_ms = _decode_via_ffmpeg(data)
        return audio, DecodeReport(
            success=True,
            used_path="ffmpeg",
            bytes_in=len(data),
            sample_rate=rate,
            sample_width=width,
            duration_ms=duration_ms,
            detected_format=fmt,
        )
    except AudioDecodeError as exc:
        return None, DecodeReport(
            success=False,
            used_path="ffmpeg",
            bytes_in=len(data),
            sample_rate=None,
            sample_width=None,
            duration_ms=None,
            detected_format=fmt,
            error=str(exc),
        )
