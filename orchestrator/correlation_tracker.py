"""Correlation-ID propagation through the pipeline."""

from __future__ import annotations

import contextvars
import uuid
from typing import Any, Callable, Optional

from .telemetry import get_logger

_logger = get_logger(__name__)

# Thread-safe / async-safe context variable for correlation_id.
_correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationTracker:
    """Generate, store, and propagate correlation IDs."""

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate and return a new UUID4 correlation ID.

        Returns:
            A UUID4 string.
        """
        return str(uuid.uuid4())

    @staticmethod
    def set_correlation_id(correlation_id: str) -> None:
        """Set the active correlation ID in context.

        Args:
            correlation_id: UUID string.

        Raises:
            ValueError: If *correlation_id* is not a valid UUID.
        """
        try:
            uuid.UUID(correlation_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid correlation_id: {correlation_id!r}") from exc
        _correlation_id_var.set(correlation_id)

    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Return the active correlation ID, or ``None``."""
        return _correlation_id_var.get()

    @staticmethod
    async def propagate_to_agent(
        agent_call: Callable[..., Any],
        correlation_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call *agent_call* ensuring *correlation_id* is propagated.

        The correlation_id is injected as a keyword argument if the
        callable accepts one.

        Args:
            agent_call: Agent method to invoke.
            correlation_id: ID to propagate.
            *args: Positional args forwarded.
            **kwargs: Keyword args forwarded.

        Returns:
            Result of the agent call.
        """
        _correlation_id_var.set(correlation_id)
        kwargs.setdefault("correlation_id", correlation_id)
        result = agent_call(*args, **kwargs)
        # Support both sync and async callables
        if hasattr(result, "__await__"):
            result = await result
        return result
