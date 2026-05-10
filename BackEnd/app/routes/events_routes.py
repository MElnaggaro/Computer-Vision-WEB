"""
Events Route
============
Read-only endpoint exposing the unified event log
(``BackEnd/logs/classroom_log.json``) to the frontend so it can replay
prior events on reconnect or page-load.

* ``GET /events``           — return all events.
* ``GET /events?since=<n>`` — return events at index >= n (cheap polling).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Query

from app.services.logging.log_service import LogService

logger = logging.getLogger(__name__)
router = APIRouter()
_log_service = LogService()


@router.get("/events")
async def list_events(
    since: int = Query(default=0, ge=0, description="Return events at index >= since"),
) -> Dict[str, Any]:
    """Return the full event log (or a slice from ``since`` onwards)."""
    events: List[Dict[str, Any]] = _log_service.load_logs()
    total = len(events)
    if since:
        events = events[since:]
    return {"count": len(events), "total": total, "events": events}


# Alias requested by the project spec — same payload, served at
# ``/logs/events`` so the dashboard can read attendance directly from a
# stable URL that doesn't depend on the API version prefix.
@router.get("/logs/events")
async def list_events_logs_alias(
    since: int = Query(default=0, ge=0, description="Return events at index >= since"),
) -> Dict[str, Any]:
    """Public ``/logs/events`` alias — identical payload to ``/events``."""
    return await list_events(since=since)
