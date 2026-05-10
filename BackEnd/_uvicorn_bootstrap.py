"""
Internal launcher bootstrap (do not run directly).

Spawned by ``run.py`` to start uvicorn with a Windows-friendly
``SelectorEventLoop``.  This works around a known CPython bug
(`bpo/issues/91227 <https://github.com/python/cpython/issues/91227>`_)
where the default ``ProactorEventLoop`` raises
``OSError(WinError 10014)`` from ``finish_accept`` and silently breaks
uvicorn's listener.

CLI:
    python -u _uvicorn_bootstrap.py <host> <port> [<log_level>]
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: python -u _uvicorn_bootstrap.py <host> <port> [<log_level>]",
            file=sys.stderr,
        )
        return 2

    host = sys.argv[1]
    port = int(sys.argv[2])
    log_level = sys.argv[3] if len(sys.argv) >= 4 else "info"

    # Ensure BackEnd/ (the directory holding this file) is on sys.path so
    # that ``import app.main`` works regardless of the parent's cwd.
    backend_root = Path(__file__).resolve().parent
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    # Windows: force SelectorEventLoop to dodge ProactorLoop accept bug.
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    import uvicorn

    config = uvicorn.Config(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        loop="asyncio",
        access_log=True,
    )
    server = uvicorn.Server(config)
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            loop.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
