"""Structured logging with correlation IDs via *structlog*."""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

_CONFIGURED = False


# ── public helpers ─────────────────────────────────────────────────


def new_correlation_id() -> str:
    """Generate and store a new correlation ID for the current context."""
    cid = str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Return the current context's correlation ID (empty if unset)."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Explicitly set the correlation ID for the current context."""
    _correlation_id.set(cid)


# ── structlog processors ──────────────────────────────────────────


def _add_correlation_id(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject the current correlation ID into every log entry."""
    cid = _correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


# ── setup ──────────────────────────────────────────────────────────


def setup_logging(log_level: str = "INFO") -> None:
    """Configure *structlog* + stdlib logging.

    Safe to call multiple times — subsequent calls are no-ops.

    Args:
        log_level: One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    _CONFIGURED = True

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # stdlib root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally named *name*."""
    log = structlog.get_logger()
    if name:
        log = log.bind(logger=name)
    return log
