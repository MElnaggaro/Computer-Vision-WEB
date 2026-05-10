"""
Health Route
============
Lightweight liveness endpoint used by the frontend to detect whether
the backend is reachable and ready to serve requests.

Returns:
    ``{"status": "online"}`` whenever the FastAPI application is up.
"""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> Dict[str, str]:
    """Return a simple liveness payload."""
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }
