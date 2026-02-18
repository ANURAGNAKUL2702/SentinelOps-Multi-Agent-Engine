"""Timeout enforcement via ``asyncio.wait_for``."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Coroutine, Dict, TypeVar

from .telemetry import get_logger

_logger = get_logger(__name__)
T = TypeVar("T")


class TimeoutManager:
    """Enforce per-agent timeouts and collect violation statistics."""

    def __init__(self) -> None:
        self._timeout_counts: Dict[str, int] = defaultdict(int)

    async def execute_with_timeout(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: float,
        agent_name: str = "",
        correlation_id: str = "",
    ) -> T:
        """Execute *coro* with an upper-bound *timeout* (seconds).

        Args:
            coro: Awaitable to execute.
            timeout: Maximum seconds to wait.
            agent_name: Agent label for logging / stats.
            correlation_id: Pipeline correlation id.

        Returns:
            Result of the coroutine.

        Raises:
            ValueError: If *timeout* is ``0``.
            asyncio.TimeoutError: If *coro* does not finish in time.
        """
        if timeout == 0:
            raise ValueError("timeout must not be 0")

        if timeout is None or timeout < 0:
            # No timeout — wait indefinitely
            return await coro

        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self._timeout_counts[agent_name] += 1
            _logger.warning(
                f"Timeout exceeded for {agent_name} ({timeout}s)",
                extra={"correlation_id": correlation_id, "agent_name": agent_name},
            )
            raise

    def get_timeout_stats(self) -> Dict[str, int]:
        """Return mapping of agent_name → timeout count."""
        return dict(self._timeout_counts)

    def reset_stats(self) -> None:
        """Clear all timeout statistics."""
        self._timeout_counts.clear()
