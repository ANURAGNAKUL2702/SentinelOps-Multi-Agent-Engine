"""Health-check endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..dependencies import get_db, get_config
from ..models import HealthResponse

router = APIRouter(tags=["health"])

_start_time = time.monotonic()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Service health check",
)
def health() -> HealthResponse:
    """Return service health, database status, version, and uptime."""
    db_status = "connected"
    try:
        db = get_db()
        # Quick connectivity probe
        with db.session() as sess:
            sess.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
    except Exception:
        db_status = "disconnected"

    uptime = round(time.monotonic() - _start_time, 2)
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        database=db_status,
        version="1.0.0",
        uptime=uptime,
    )
