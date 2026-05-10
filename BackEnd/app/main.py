"""
FastAPI Application Factory
===========================
Creates and configures the Smart Classroom Assistant API and registers
all sub-routers under the versioned ``/api/v1`` prefix.

Routes
------
* ``/api/v1/health``        — liveness probe
* ``/api/v1/vision/*``      — face recognition + emotion + attendance
* ``/api/v1/speech/*``      — server-side mic transcription
* ``/api/v1/nlp/*``         — text-only topic classification
* ``/api/v1/interaction/*`` — combined speech + NLP, with student attribution
* ``/api/v1/registration/*``— stranger registration + admin approval
* ``/api/v1/events``        — append-only event log read API

The frontend (``FrontEnd/index.html``) is also mounted at ``/ui`` when
the directory is present, so a single uvicorn instance serves both the
API and the dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routes.events_routes import router as events_router
from app.routes.health_routes import router as health_router
from app.routes.interaction_routes import router as interaction_router
from app.routes.nlp_routes import router as nlp_router
from app.routes.registration_routes import router as registration_router
from app.routes.speech_routes import router as speech_router
from app.routes.vision_routes import router as vision_router

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory — returns a fully configured ``FastAPI`` instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Route registration ───────────────────────────────────────────
    api_prefix = settings.API_V1_STR

    app.include_router(health_router, prefix=api_prefix, tags=["Health"])
    # Also expose /health at the application root so frontends do not
    # need to know the API prefix to detect liveness.
    app.include_router(health_router, tags=["Health"])

    app.include_router(vision_router, prefix=f"{api_prefix}/vision", tags=["Vision"])
    app.include_router(speech_router, prefix=f"{api_prefix}/speech", tags=["Speech"])
    app.include_router(nlp_router, prefix=f"{api_prefix}/nlp", tags=["NLP"])
    app.include_router(
        interaction_router,
        prefix=f"{api_prefix}/interaction",
        tags=["Interaction"],
    )
    app.include_router(
        registration_router,
        prefix=f"{api_prefix}/registration",
        tags=["Registration"],
    )
    app.include_router(events_router, prefix=api_prefix, tags=["Events"])
    # Also expose the events router at the application root so the
    # frontend can hit ``GET /logs/events`` per the project spec.
    app.include_router(events_router, tags=["Events"])

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "ui": "/ui",
        }

    # ── Optional static frontend ─────────────────────────────────────
    frontend_dir = settings.BASE_DIR.parent / "FrontEnd"
    if frontend_dir.is_dir():
        app.mount(
            "/ui",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )
        logger.info("Frontend mounted at /ui from %s", frontend_dir)

    logger.info(
        "%s v%s started — routes registered under %s",
        settings.PROJECT_NAME,
        settings.VERSION,
        api_prefix,
    )
    return app


app = create_app()
