"""
Registration Service — Stranger → Pending → Approved Workflow
================================================================
Implements the four-step registration pipeline used by the dashboard:

    1. ``start``    — open a registration session (returns a session_id).
    2. ``capture``  — append a face image to ``data/pending_students/<session>``.
    3. ``submit``   — bind a ``Firstname_Lastname`` name to the session and
                      rename the temp folder to that final name.
    4. ``approve``  — admin codeword verified server-side; move the folder
                      to ``data/students_faces/<name>`` and rebuild encodings.
       ``reject``   — discard the pending folder.

The service uses an in-process dict to track session state, guarded by
a re-entrant lock for thread-safety.  All file I/O goes through
``pathlib.Path`` so the workflow is Windows-compatible.
"""

from __future__ import annotations

import logging
import re
import secrets
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.services.logging.log_service import LogService
from app.services.vision.vision_session import (
    VisionError,
    decode_base64_frame,
    get_vision_session,
)

logger = logging.getLogger(__name__)


# ── Custom exceptions ───────────────────────────────────────────────


class RegistrationError(Exception):
    """Base exception for the registration workflow."""


class InvalidNameError(RegistrationError):
    """Raised when the supplied name does not match Firstname_Lastname."""


class SessionNotFoundError(RegistrationError):
    """Raised when the supplied session_id is unknown or expired."""


class NotEnoughImagesError(RegistrationError):
    """Raised when fewer than ``REGISTRATION_MIN_IMAGES`` were captured."""


class UnauthorizedError(RegistrationError):
    """Raised when the admin codeword is missing or incorrect."""


# ── Session record ──────────────────────────────────────────────────


class _RegistrationSession:
    """In-memory metadata for a single registration flow."""

    __slots__ = ("session_id", "temp_dir", "name", "image_count", "created_at")

    def __init__(self, session_id: str, temp_dir: Path) -> None:
        self.session_id: str = session_id
        self.temp_dir: Path = temp_dir
        self.name: Optional[str] = None
        self.image_count: int = 0
        self.created_at: datetime = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "image_count": self.image_count,
            "created_at": self.created_at.isoformat(),
            "temp_dir": str(self.temp_dir),
        }


# ── Service ─────────────────────────────────────────────────────────


class RegistrationService:
    """Manage stranger registration sessions, captures, and admin approval.

    Args:
        pending_root:     Override the root directory that holds pending
                          captures. Defaults to ``data/pending_students/``.
        students_root:    Override the final students directory.
                          Defaults to ``data/students_faces/``.
        admin_codeword:   Override the admin codeword (for tests).
        log_service:      Inject a ``LogService`` (otherwise the default
                          singleton path is used).
    """

    def __init__(
        self,
        pending_root: Optional[Path] = None,
        students_root: Optional[Path] = None,
        admin_codeword: Optional[str] = None,
        log_service: Optional[LogService] = None,
    ) -> None:
        self.pending_root: Path = pending_root or settings.PENDING_STUDENTS_DIR
        self.students_root: Path = students_root or settings.STUDENTS_FACES_DIR
        self.admin_codeword: str = admin_codeword or settings.ADMIN_CODEWORD
        self.log_service: LogService = log_service or LogService()
        self.name_pattern = re.compile(settings.REGISTRATION_NAME_PATTERN)

        self._sessions: Dict[str, _RegistrationSession] = {}
        self._lock = threading.RLock()

        self.pending_root.mkdir(parents=True, exist_ok=True)
        self.students_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1 — start ───────────────────────────────────────────────

    def start(self) -> Dict[str, Any]:
        """Create a new pending registration session."""
        with self._lock:
            session_id = uuid.uuid4().hex
            temp_dir = self.pending_root / f".session_{session_id}"
            temp_dir.mkdir(parents=True, exist_ok=False)
            session = _RegistrationSession(session_id, temp_dir)
            self._sessions[session_id] = session
        logger.info("Registration session started: %s", session_id)
        return session.to_dict()

    # ── Step 2 — capture ─────────────────────────────────────────────

    def capture(self, session_id: str, image_base64: str) -> Dict[str, Any]:
        """Save a single base64 frame into the pending session folder."""
        session = self._require_session(session_id)

        if session.image_count >= settings.REGISTRATION_MAX_IMAGES:
            raise RegistrationError(
                f"Maximum of {settings.REGISTRATION_MAX_IMAGES} images reached"
            )

        try:
            frame = decode_base64_frame(image_base64)
        except VisionError as exc:
            raise RegistrationError(f"Invalid image: {exc}") from exc

        with self._lock:
            session.image_count += 1
            file_path = session.temp_dir / f"img_{session.image_count:02d}.jpg"
            ok = cv2.imwrite(str(file_path), frame)
            if not ok:
                # rollback the count if write failed
                session.image_count -= 1
                raise RegistrationError(f"Could not write image to {file_path}")

        logger.info(
            "Captured image %d for session %s -> %s",
            session.image_count, session_id, file_path,
        )
        return {
            "session_id": session_id,
            "image_count": session.image_count,
            "min_required": settings.REGISTRATION_MIN_IMAGES,
            "max_allowed": settings.REGISTRATION_MAX_IMAGES,
            "ready_for_submit": session.image_count >= settings.REGISTRATION_MIN_IMAGES,
        }

    # ── Step 3 — submit ──────────────────────────────────────────────

    def submit(self, session_id: str, name: str) -> Dict[str, Any]:
        """Bind a Firstname_Lastname name to the session.

        Renames the temp pending folder to ``pending_students/<name>``.
        """
        session = self._require_session(session_id)
        clean_name = (name or "").strip()

        if not self.name_pattern.match(clean_name):
            raise InvalidNameError(
                "Name must match Firstname_Lastname (letters and a single underscore)."
            )

        if session.image_count < settings.REGISTRATION_MIN_IMAGES:
            raise NotEnoughImagesError(
                f"Need at least {settings.REGISTRATION_MIN_IMAGES} images "
                f"(have {session.image_count})."
            )

        with self._lock:
            final_pending = self.pending_root / clean_name
            if final_pending.exists():
                # collision — append a short suffix to keep prior data intact
                suffix = secrets.token_hex(2)
                final_pending = self.pending_root / f"{clean_name}_{suffix}"

            session.temp_dir.rename(final_pending)
            session.temp_dir = final_pending
            session.name = clean_name

        logger.info(
            "Submitted registration session %s as %s (folder=%s)",
            session_id, clean_name, final_pending,
        )
        return session.to_dict()

    # ── Step 4a — approve ────────────────────────────────────────────

    def approve(self, session_id: str, codeword: str) -> Dict[str, Any]:
        """Verify admin codeword, move pending → students_faces, rebuild encodings."""
        session = self._require_session(session_id)
        if not session.name:
            raise RegistrationError(
                "Session has not been submitted yet — call /registration/submit first."
            )
        self._verify_codeword(codeword)

        target = self.students_root / session.name
        if target.exists():
            # If a student folder already exists, merge the new images into it
            # rather than overwriting prior captures.
            target.mkdir(parents=True, exist_ok=True)
            for src in session.temp_dir.iterdir():
                if src.is_file():
                    dst = target / f"new_{src.name}"
                    shutil.move(str(src), str(dst))
            shutil.rmtree(session.temp_dir, ignore_errors=True)
        else:
            shutil.move(str(session.temp_dir), str(target))

        # Rebuild the encoding cache so the new student is immediately recognisable
        try:
            session_vision = get_vision_session()
            summary = session_vision.rebuild_encodings()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Encoding rebuild failed after approval")
            summary = {"error": str(exc)}

        # Persist the approval event
        self.log_service.log_registration_event(student=session.name, approved=True)

        approved_name = session.name
        with self._lock:
            self._sessions.pop(session_id, None)

        logger.info("Approved registration for %s", approved_name)
        return {
            "student": approved_name,
            "approved": True,
            "encoding_summary": summary,
        }

    # ── Step 4b — reject ─────────────────────────────────────────────

    def reject(
        self, session_id: str, *, delete_files: bool = True
    ) -> Dict[str, Any]:
        """Discard the pending session.

        If ``delete_files`` is ``True`` the temp folder is removed; otherwise
        it is preserved for manual review (still inside ``pending_students``).
        """
        session = self._require_session(session_id)
        student_name = session.name or session.session_id

        if delete_files:
            shutil.rmtree(session.temp_dir, ignore_errors=True)

        self.log_service.log_registration_event(
            student=student_name, approved=False
        )

        with self._lock:
            self._sessions.pop(session_id, None)

        logger.info(
            "Rejected registration session %s (student=%s, deleted=%s)",
            session_id, student_name, delete_files,
        )
        return {
            "student": student_name,
            "approved": False,
            "deleted_pending": delete_files,
        }

    # ── Introspection ────────────────────────────────────────────────

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return all currently active registration sessions."""
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self._require_session(session_id).to_dict()

    # ── Internals ────────────────────────────────────────────────────

    def _require_session(self, session_id: str) -> _RegistrationSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(
                    f"Unknown registration session: {session_id}"
                )
            return session

    def _verify_codeword(self, codeword: str) -> None:
        if not codeword or not isinstance(codeword, str):
            raise UnauthorizedError("Admin codeword is required.")
        # Constant-time comparison to avoid trivial timing attacks
        if not secrets.compare_digest(codeword.strip(), self.admin_codeword):
            raise UnauthorizedError("Invalid admin codeword.")


# ── Process-level singleton accessor ────────────────────────────────


_REGISTRATION_SERVICE: Optional[RegistrationService] = None
_LOCK = threading.Lock()


def get_registration_service() -> RegistrationService:
    """Return the shared :class:`RegistrationService` (lazy-instantiated)."""
    global _REGISTRATION_SERVICE
    with _LOCK:
        if _REGISTRATION_SERVICE is None:
            _REGISTRATION_SERVICE = RegistrationService()
        return _REGISTRATION_SERVICE


def reset_registration_service() -> None:
    """Drop the singleton — used by tests to reset state between cases."""
    global _REGISTRATION_SERVICE
    with _LOCK:
        _REGISTRATION_SERVICE = None
