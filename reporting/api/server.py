"""FastAPI application — initialisation, middleware, lifecycle events."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import init_dependencies, shutdown_dependencies
from .routes import analytics, health, incidents, metrics, reports


# ------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: init DB & singletons.  Shutdown: cleanup."""
    init_dependencies()
    yield
    shutdown_dependencies()


# ------------------------------------------------------------------
# Application
# ------------------------------------------------------------------

app = FastAPI(
    title="Warroom Simulator — Reporting API",
    version="1.0.0",
    description="REST API for the Reporting & Analysis Layer.",
    lifespan=_lifespan,
)


# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_logger = logging.getLogger("reporting.api")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Log every request with method, path, status, and duration."""
    start = time.monotonic()
    correlation_id = request.headers.get("X-Correlation-ID", "")
    try:
        response = await call_next(request)
    except Exception:
        _logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    elapsed = round((time.monotonic() - start) * 1000, 2)
    _logger.info(
        "%s %s → %s  (%.1fms)%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        f"  cid={correlation_id}" if correlation_id else "",
    )
    if correlation_id:
        response.headers["X-Correlation-ID"] = correlation_id
    return response


# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(reports.router, prefix="/api/v1/reports")
app.include_router(incidents.router, prefix="/api/v1/incidents")
app.include_router(analytics.router, prefix="/api/v1/analytics")
app.include_router(metrics.router, prefix="/api/v1/metrics")
app.include_router(health.router, prefix="/api/v1/health")
