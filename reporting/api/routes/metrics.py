"""Prometheus metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["metrics"])

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False


@router.get(
    "/prometheus",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
)
def prometheus_metrics() -> PlainTextResponse:
    """Export Prometheus metrics in text exposition format."""
    if not _HAS_PROM:
        return PlainTextResponse(
            "# prometheus_client not installed\n",
            media_type="text/plain",
        )
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)
