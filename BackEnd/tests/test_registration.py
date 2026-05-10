"""
Tests — Registration Workflow
==============================
Covers the full stranger → pending → admin-approval pipeline:
    • start
    • capture (5+ valid frames)
    • submit (Firstname_Lastname validation)
    • approve  — codeword required, must be server-validated
    • reject   — pending folder is removed, rejection event emitted

The encoding rebuild step is patched out so dlib is not invoked.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# ── Ensure BackEnd/ is on sys.path ───────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.logging.log_service import LogService
from app.services.registration.registration_service import (
    InvalidNameError,
    NotEnoughImagesError,
    RegistrationService,
    SessionNotFoundError,
    UnauthorizedError,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _b64_frame(width: int = 80, height: int = 80) -> str:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


@pytest.fixture
def reg_service(tmp_path):
    """Build an isolated RegistrationService with an in-tmpdir layout."""
    pending = tmp_path / "pending_students"
    students = tmp_path / "students_faces"
    log_file = tmp_path / "log.json"

    log_service = LogService(log_file=log_file)
    svc = RegistrationService(
        pending_root=pending,
        students_root=students,
        admin_codeword="testpass-2026",
        log_service=log_service,
    )
    return svc, log_service


# ── Lifecycle ────────────────────────────────────────────────────────


class TestStart:
    def test_start_creates_session(self, reg_service):
        svc, _ = reg_service
        info = svc.start()
        assert info["session_id"]
        assert info["image_count"] == 0
        assert Path(info["temp_dir"]).is_dir()

    def test_start_creates_unique_sessions(self, reg_service):
        svc, _ = reg_service
        a = svc.start()["session_id"]
        b = svc.start()["session_id"]
        assert a != b


# ── Capture ──────────────────────────────────────────────────────────


class TestCapture:
    def test_capture_writes_frame(self, reg_service):
        svc, _ = reg_service
        sid = svc.start()["session_id"]
        info = svc.capture(sid, _b64_frame())
        assert info["image_count"] == 1
        assert info["ready_for_submit"] is False  # need at least 5

    def test_ready_for_submit_after_min(self, reg_service):
        svc, _ = reg_service
        sid = svc.start()["session_id"]
        for _ in range(5):
            info = svc.capture(sid, _b64_frame())
        assert info["image_count"] == 5
        assert info["ready_for_submit"] is True

    def test_capture_unknown_session(self, reg_service):
        svc, _ = reg_service
        with pytest.raises(SessionNotFoundError):
            svc.capture("nonexistent-id", _b64_frame())

    def test_capture_invalid_image(self, reg_service):
        svc, _ = reg_service
        sid = svc.start()["session_id"]
        with pytest.raises(Exception):
            svc.capture(sid, "###not-base64###")


# ── Submit (name validation) ─────────────────────────────────────────


class TestSubmit:
    def _populate(self, svc, n=5):
        sid = svc.start()["session_id"]
        for _ in range(n):
            svc.capture(sid, _b64_frame())
        return sid

    def test_submit_with_valid_name(self, reg_service):
        svc, _ = reg_service
        sid = self._populate(svc)
        info = svc.submit(sid, "Ahmed_Ali")
        assert info["name"] == "Ahmed_Ali"
        assert Path(info["temp_dir"]).name == "Ahmed_Ali"

    def test_submit_rejects_bad_name(self, reg_service):
        svc, _ = reg_service
        sid = self._populate(svc)
        for bad in ["AhmedAli", "ahmed ali", "Ahmed_Ali_Smith", "Ahmed-Ali", "Ahmed1_Ali", ""]:
            with pytest.raises(InvalidNameError):
                svc.submit(sid, bad)

    def test_submit_rejects_too_few_images(self, reg_service):
        svc, _ = reg_service
        sid = svc.start()["session_id"]
        svc.capture(sid, _b64_frame())  # only 1 image
        with pytest.raises(NotEnoughImagesError):
            svc.submit(sid, "Ahmed_Ali")


# ── Approve / Reject ─────────────────────────────────────────────────


class TestApproveReject:
    def _ready_session(self, svc, name="Ahmed_Ali"):
        sid = svc.start()["session_id"]
        for _ in range(5):
            svc.capture(sid, _b64_frame())
        svc.submit(sid, name)
        return sid

    def test_approve_with_valid_codeword(self, reg_service):
        svc, log_service = reg_service
        sid = self._ready_session(svc)

        # Patch encoding rebuild to avoid dlib in tests
        with patch(
            "app.services.registration.registration_service.get_vision_session"
        ) as mock_session:
            mock_session.return_value.rebuild_encodings.return_value = {
                "students": {"Ahmed_Ali": 5},
                "total_encodings": 5,
            }
            result = svc.approve(sid, "testpass-2026")

        assert result["approved"] is True
        assert result["student"] == "Ahmed_Ali"
        # Pending → students_faces move
        assert (svc.students_root / "Ahmed_Ali").is_dir()
        assert not (svc.pending_root / "Ahmed_Ali").exists()
        # Event emitted
        events = log_service.load_logs()
        assert any(
            e["event"] == "registration_approved" and e["student"] == "Ahmed_Ali"
            for e in events
        )

    def test_approve_rejects_wrong_codeword(self, reg_service):
        svc, _ = reg_service
        sid = self._ready_session(svc)
        with pytest.raises(UnauthorizedError):
            svc.approve(sid, "wrong-codeword")

    def test_approve_rejects_empty_codeword(self, reg_service):
        svc, _ = reg_service
        sid = self._ready_session(svc)
        with pytest.raises(UnauthorizedError):
            svc.approve(sid, "")

    def test_approve_requires_submit_first(self, reg_service):
        svc, _ = reg_service
        sid = svc.start()["session_id"]
        for _ in range(5):
            svc.capture(sid, _b64_frame())
        with pytest.raises(Exception):
            svc.approve(sid, "testpass-2026")

    def test_reject_removes_pending(self, reg_service):
        svc, log_service = reg_service
        sid = self._ready_session(svc, name="Test_Reject")
        result = svc.reject(sid, delete_files=True)
        assert result["approved"] is False
        assert result["deleted_pending"] is True
        assert not (svc.pending_root / "Test_Reject").exists()

        events = log_service.load_logs()
        assert any(
            e["event"] == "registration_rejected" and e["student"] == "Test_Reject"
            for e in events
        )

    def test_reject_can_preserve_pending(self, reg_service):
        svc, _ = reg_service
        sid = self._ready_session(svc, name="Keep_Me")
        svc.reject(sid, delete_files=False)
        assert (svc.pending_root / "Keep_Me").exists()
