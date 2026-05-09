"""
FastAPI Application Factory
===========================
Creates and configures the main FastAPI application instance,
registering all sub‑routers under the versioned API prefix.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes.vision_routes import router as vision_router
from app.routes.speech_routes import router as speech_router
from app.routes.nlp_routes import router as nlp_router

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory – returns a fully configured ``FastAPI`` instance."""
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
    app.include_router(
        vision_router,
        prefix=f"{settings.API_V1_STR}/vision",
        tags=["Vision"],
    )
    app.include_router(
        speech_router,
        prefix=f"{settings.API_V1_STR}/speech",
        tags=["Speech"],
    )
    app.include_router(
        nlp_router,
        prefix=f"{settings.API_V1_STR}/nlp",
        tags=["NLP"],
    )

    @app.get("/")
    async def root():
        return {
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
        }

    logger.info(
        "%s v%s started – routes registered under %s",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.API_V1_STR,
    )
    return app


app = create_app()
