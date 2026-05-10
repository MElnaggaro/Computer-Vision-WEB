"""
Smart Classroom Assistant — One-Shot Launcher
==============================================
Single command to bring up the entire stack:

    python BackEnd/run.py

Workflow
--------
    [1/5] Spawn the FastAPI backend (uvicorn) at http://127.0.0.1:8000
    [2/5] Spawn a static file server at http://127.0.0.1:5500 serving FrontEnd/
    [3/5] Poll GET /health until {"status": "online"} or timeout
    [4/5] Open the default web browser at the frontend URL
    [5/5] Block until Ctrl+C, then cleanly terminate both children

Both servers run as child processes so we can guarantee clean teardown
on Ctrl+C, on any unexpected child exit, and on parent termination.

Windows-compatible: we always use ``127.0.0.1`` (rather than ``0.0.0.0``)
to dodge the Windows Firewall prompt, and use ``CREATE_NEW_PROCESS_GROUP``
so we can isolate signals to each child.

CLI options
-----------
    --backend-port      override port for FastAPI (default 8000)
    --frontend-port     override port for the static server (default 5500)
    --no-browser        skip the auto-open step (useful for headless CI)
    --health-timeout    seconds to wait for backend readiness (default 30)
"""

from __future__ import annotations

import argparse
import atexit
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

# ── Paths ────────────────────────────────────────────────────────────

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
FRONTEND_DIR = PROJECT_ROOT / "FrontEnd"

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_HOST = "127.0.0.1"
DEFAULT_FRONTEND_PORT = 5500
DEFAULT_HEALTH_PATH = "/health"
DEFAULT_HEALTH_TIMEOUT = 30   # seconds
HEALTH_POLL_INTERVAL = 0.5    # seconds


# ── Console output helpers ───────────────────────────────────────────


def _banner(title: str) -> None:
    bar = "=" * 50
    print(bar, flush=True)
    print(f"  {title}", flush=True)
    print(bar, flush=True)


def _step(num: int, total: int, message: str) -> None:
    print(f"[{num}/{total}] {message}", flush=True)


def _info(message: str) -> None:
    print(f"      {message}", flush=True)


def _error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)


# ── Port-busy detection ──────────────────────────────────────────────


def is_port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return ``True`` if a TCP listener is already bound to ``(host, port)``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


# ── Subprocess builder ───────────────────────────────────────────────


_IS_WINDOWS = os.name == "nt"

# On Windows we put each child in its own process group so we can send
# CTRL_BREAK_EVENT for graceful shutdown without disturbing the parent.
_POPEN_KWARGS_WIN = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}


# ── Windows Job Object — auto-kill children if parent dies ──────────


class _WindowsJobObject:
    """A best-effort Windows Job Object that kills its assigned processes
    when the parent dies for *any* reason — Ctrl+C, normal exit, hard
    crash, or external ``taskkill /F``.  This guarantees no orphan
    backend / frontend processes outlive ``run.py``.

    On non-Windows platforms or when ctypes attachment fails, the
    object is a transparent no-op.
    """

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JobObjectExtendedLimitInformation = 9

    def __init__(self) -> None:
        self.handle = None
        if not _IS_WINDOWS:
            return
        try:
            import ctypes
            import ctypes.wintypes as wt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", wt.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wt.LARGE_INTEGER),
                    ("LimitFlags", wt.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_void_p),
                    ("PriorityClass", wt.DWORD),
                    ("SchedulingClass", wt.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            self._ctypes = ctypes
            self._kernel32 = kernel32
            self._JOBOBJECT_EXTENDED_LIMIT_INFORMATION = JOBOBJECT_EXTENDED_LIMIT_INFORMATION

            kernel32.CreateJobObjectW.restype = wt.HANDLE
            kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]

            self.handle = kernel32.CreateJobObjectW(None, None)
            if not self.handle:
                self.handle = None
                return

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )

            kernel32.SetInformationJobObject.restype = wt.BOOL
            kernel32.SetInformationJobObject.argtypes = [
                wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD,
            ]
            ok = kernel32.SetInformationJobObject(
                self.handle,
                self._JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                kernel32.CloseHandle(self.handle)
                self.handle = None

            kernel32.AssignProcessToJobObject.restype = wt.BOOL
            kernel32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
            kernel32.OpenProcess.restype = wt.HANDLE
            kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        except Exception:  # noqa: BLE001
            # Any failure (ctypes import, missing symbol, locked-down
            # environment) downgrades to a no-op so the launcher still
            # functions; we just lose the auto-kill guarantee.
            self.handle = None

    def assign(self, proc: subprocess.Popen) -> None:
        """Add a Popen child to the job so it dies when the parent does."""
        if self.handle is None:
            return
        try:
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_TERMINATE = 0x0001
            access = PROCESS_SET_QUOTA | PROCESS_TERMINATE
            child_handle = self._kernel32.OpenProcess(access, False, proc.pid)
            if not child_handle:
                return
            try:
                self._kernel32.AssignProcessToJobObject(self.handle, child_handle)
            finally:
                self._kernel32.CloseHandle(child_handle)
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        """Close the job — Windows then kills every assigned process."""
        if self.handle is None:
            return
        try:
            self._kernel32.CloseHandle(self.handle)
        except Exception:  # noqa: BLE001
            pass
        self.handle = None


def _spawn(cmd: List[str], cwd: Path, label: str) -> subprocess.Popen:
    """Launch a child process with platform-appropriate creation flags."""
    env = os.environ.copy()
    # Ensure children stream their logs immediately rather than buffering.
    env.setdefault("PYTHONUNBUFFERED", "1")
    kwargs = dict(
        cwd=str(cwd),
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    if _IS_WINDOWS:
        kwargs.update(_POPEN_KWARGS_WIN)
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Failed to start {label}: command not found ({cmd[0]})") from exc


def _terminate(proc: Optional[subprocess.Popen], label: str, timeout: float = 5.0) -> None:
    """Stop a child cleanly: signal first, escalate to kill on timeout."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if _IS_WINDOWS:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except (ValueError, OSError):
                proc.terminate()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        proc.wait(timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        _error(f"Failed to terminate {label}: {exc}")


# ── Health check ─────────────────────────────────────────────────────


def wait_for_backend_health(
    host: str,
    port: int,
    path: str = DEFAULT_HEALTH_PATH,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
) -> bool:
    """Poll ``GET <path>`` until the backend reports online, or timeout."""
    deadline = time.monotonic() + timeout
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2.0)
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            if resp.status == 200:
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = {}
                if str(data.get("status", "")).lower() == "online":
                    return True
                last_error = f"unexpected /health body: {body[:120]!r}"
            else:
                last_error = f"HTTP {resp.status} on {path}"
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
        time.sleep(HEALTH_POLL_INTERVAL)

    if last_error:
        _error(f"Backend health check timed out: {last_error}")
    else:
        _error("Backend health check timed out.")
    return False


# ── Frontend config writer ───────────────────────────────────────────


def write_frontend_config(api_base_url: str) -> None:
    """Refresh ``FrontEnd/JS/config.js`` so the dashboard knows the API URL.

    This makes the launcher the single source of truth for the API base.
    Users can still override via environment overrides on the page itself.
    """
    config_path = FRONTEND_DIR / "JS" / "config.js"
    config_payload = (
        "// Auto-generated by BackEnd/run.py — do not edit by hand.\n"
        "// Override before any other script tag if needed.\n"
        "window.APP_CONFIG = window.APP_CONFIG || {\n"
        f"    API_BASE_URL: '{api_base_url}',\n"
        "    HEALTH_PATH: '/health',\n"
        "    EVENT_POLL_MS: 2000,\n"
        "    HEALTH_INTERVAL_MS: 5000,\n"
        "    RECOGNIZE_INTERVAL_MS: 800,\n"
        "};\n"
    )
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_payload, encoding="utf-8")
    except OSError as exc:
        _error(f"Could not write frontend config at {config_path}: {exc}")


# ── Main launcher ────────────────────────────────────────────────────


class Launcher:
    """Owns the backend + frontend child processes and their lifecycle."""

    def __init__(
        self,
        backend_host: str,
        backend_port: int,
        frontend_host: str,
        frontend_port: int,
        health_timeout: float,
        open_browser: bool,
    ) -> None:
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port
        self.health_timeout = health_timeout
        self.open_browser = open_browser

        self.backend_proc: Optional[subprocess.Popen] = None
        self.frontend_proc: Optional[subprocess.Popen] = None
        self._shutdown_called = threading.Event()
        # Windows Job Object — auto-kills children if the parent dies
        # for any reason (Ctrl+C, crash, taskkill /F).  No-op elsewhere.
        self._job = _WindowsJobObject()

    # ── lifecycle ────────────────────────────────────────────────────

    def run(self) -> int:
        _banner("Smart Classroom Assistant Startup")

        # 0. Pre-flight checks
        if not FRONTEND_DIR.is_dir():
            _error(f"FrontEnd directory not found at {FRONTEND_DIR}")
            return 1
        if is_port_in_use(self.backend_host, self.backend_port):
            _error(
                f"Backend port {self.backend_port} is already in use on "
                f"{self.backend_host}. Stop the other process or pass "
                f"--backend-port <free-port>."
            )
            return 1
        if is_port_in_use(self.frontend_host, self.frontend_port):
            _error(
                f"Frontend port {self.frontend_port} is already in use on "
                f"{self.frontend_host}. Stop the other process or pass "
                f"--frontend-port <free-port>."
            )
            return 1

        # Refresh frontend config so it points at this launch's backend
        write_frontend_config(f"http://{self.backend_host}:{self.backend_port}")

        # Make sure cleanup runs even on hard failure
        atexit.register(self.shutdown)
        self._install_signal_handlers()

        # 1. Backend
        _step(1, 5, "Starting backend...")
        try:
            self.backend_proc = self._start_backend()
        except RuntimeError as exc:
            _error(str(exc))
            return 1
        self._job.assign(self.backend_proc)
        _info(f"Backend online at http://{self.backend_host}:{self.backend_port}")

        # 2. Frontend
        _step(2, 5, "Starting frontend...")
        try:
            self.frontend_proc = self._start_frontend()
        except RuntimeError as exc:
            _error(str(exc))
            self.shutdown()
            return 1
        self._job.assign(self.frontend_proc)
        _info(f"Frontend online at http://{self.frontend_host}:{self.frontend_port}")

        # 3. Health check
        _step(3, 5, "Checking backend health...")
        ok = wait_for_backend_health(
            host=self.backend_host,
            port=self.backend_port,
            path=DEFAULT_HEALTH_PATH,
            timeout=self.health_timeout,
        )
        if not ok:
            _error("Backend did not become healthy in time. Shutting down.")
            self.shutdown()
            return 1
        _info("Backend healthy")

        # 4. Browser
        _step(4, 5, "Opening browser...")
        frontend_url = f"http://{self.frontend_host}:{self.frontend_port}"
        if self.open_browser:
            try:
                webbrowser.open(frontend_url)
                _info("Browser opened")
            except Exception as exc:  # noqa: BLE001
                _info(f"Could not auto-open browser: {exc}")
                _info(f"Open manually: {frontend_url}")
        else:
            _info(f"Skipped (--no-browser). Visit {frontend_url}")

        # 5. Ready
        _step(5, 5, "System ready")
        print("=" * 50)
        print("  Press Ctrl+C to stop.")
        print("=" * 50)

        return self._supervise()

    # ── child commands ───────────────────────────────────────────────

    def _start_backend(self) -> subprocess.Popen:
        # We delegate to ``_uvicorn_bootstrap.py`` to work around a known
        # Windows + Python ProactorEventLoop bug (CPython #91227) that
        # silently breaks uvicorn's accept loop.  The bootstrap file
        # forces ``WindowsSelectorEventLoopPolicy`` before uvicorn boots
        # and is a no-op on non-Windows platforms.
        bootstrap = BACKEND_ROOT / "_uvicorn_bootstrap.py"
        if not bootstrap.is_file():
            raise RuntimeError(f"Backend bootstrap not found at {bootstrap}")
        cmd = [
            sys.executable,
            "-u",
            str(bootstrap),
            self.backend_host,
            str(self.backend_port),
            "info",
        ]
        return _spawn(cmd, cwd=BACKEND_ROOT, label="backend")

    def _start_frontend(self) -> subprocess.Popen:
        # Python ships ``http.server`` with a ``--directory`` flag that
        # avoids changing CWD globally. ``--bind`` keeps it on loopback
        # so Windows Firewall doesn't prompt.
        cmd = [
            sys.executable,
            "-m",
            "http.server",
            str(self.frontend_port),
            "--bind",
            self.frontend_host,
            "--directory",
            str(FRONTEND_DIR),
        ]
        return _spawn(cmd, cwd=FRONTEND_DIR, label="frontend")

    # ── supervision ──────────────────────────────────────────────────

    def _supervise(self) -> int:
        """Block until Ctrl+C or a child dies unexpectedly."""
        try:
            while True:
                if self._shutdown_called.is_set():
                    return 0
                if self.backend_proc and self.backend_proc.poll() is not None:
                    _error(
                        f"Backend exited unexpectedly with code "
                        f"{self.backend_proc.returncode}."
                    )
                    self.shutdown()
                    return 1
                if self.frontend_proc and self.frontend_proc.poll() is not None:
                    _error(
                        f"Frontend exited unexpectedly with code "
                        f"{self.frontend_proc.returncode}."
                    )
                    self.shutdown()
                    return 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()
            _info("Ctrl+C received — shutting down...")
            self.shutdown()
            return 0

    # ── shutdown ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._shutdown_called.is_set():
            return
        self._shutdown_called.set()
        _terminate(self.frontend_proc, "frontend")
        _terminate(self.backend_proc, "backend")
        # Closing the Job Object guarantees any stragglers are killed too.
        self._job.close()
        print("Stopped. Ports released.", flush=True)

    # ── signal wiring ────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        def _handle(_signum, _frame) -> None:
            self.shutdown()
            sys.exit(0)

        # SIGINT comes naturally via KeyboardInterrupt during the
        # supervise() loop; SIGTERM/SIGBREAK are wired explicitly so
        # external `kill` commands and Windows console-close events
        # don't leave orphan children behind.
        try:
            signal.signal(signal.SIGTERM, _handle)
        except (ValueError, OSError):
            pass
        if _IS_WINDOWS and hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, _handle)  # type: ignore[attr-defined]
            except (ValueError, OSError):
                pass


# ── CLI entry point ──────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Smart Classroom Assistant (backend + frontend)."
    )
    parser.add_argument(
        "--backend-host", default=DEFAULT_BACKEND_HOST,
        help=f"Backend host (default {DEFAULT_BACKEND_HOST}).",
    )
    parser.add_argument(
        "--backend-port", type=int, default=DEFAULT_BACKEND_PORT,
        help=f"Backend port (default {DEFAULT_BACKEND_PORT}).",
    )
    parser.add_argument(
        "--frontend-host", default=DEFAULT_FRONTEND_HOST,
        help=f"Frontend host (default {DEFAULT_FRONTEND_HOST}).",
    )
    parser.add_argument(
        "--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT,
        help=f"Frontend port (default {DEFAULT_FRONTEND_PORT}).",
    )
    parser.add_argument(
        "--health-timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT,
        help="Seconds to wait for backend /health to become online.",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open the default browser.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    launcher = Launcher(
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
        health_timeout=args.health_timeout,
        open_browser=not args.no_browser,
    )
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
