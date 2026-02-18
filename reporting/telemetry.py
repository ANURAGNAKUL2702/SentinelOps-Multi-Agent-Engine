"""Structured JSON logging and Prometheus metrics for the Reporting layer."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram

    _HAS_PROM = True
except ImportError:  # pragma: no cover
    _HAS_PROM = False


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = record.correlation_id
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with JSON output.

    Args:
        name: Logger name (usually ``__name__``).

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_registry: Optional[CollectorRegistry] = None

if _HAS_PROM:
    _registry = CollectorRegistry()
    report_generation_seconds = Histogram(
        "report_generation_seconds",
        "Time to generate a report",
        labelnames=["format"],
        registry=_registry,
    )
    reports_generated_total = Counter(
        "reports_generated_total",
        "Total reports generated",
        labelnames=["format"],
        registry=_registry,
    )
    database_operations_total = Counter(
        "reporting_database_operations_total",
        "Total database operations in reporting layer",
        labelnames=["operation"],
        registry=_registry,
    )
