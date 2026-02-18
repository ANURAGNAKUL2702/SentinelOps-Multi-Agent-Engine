"""Exponential-backoff retry policy with jitter."""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict

from .circuit_breaker import CircuitBreaker
from .config import OrchestratorConfig
from .schema import CircuitBreakerState, CircuitBreakerOpenError
from .telemetry import get_logger

_logger = get_logger(__name__)


class RetryPolicy:
    """Retry failed async agent calls with exponential back-off.

    back-off delay for attempt *i* (0-indexed):
        ``base × multiplier^i × (1 + uniform(−jitter, jitter))``
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.max_retries = config.max_retries
        self.backoff_base = config.retry_backoff_base
        self.backoff_multiplier = config.retry_backoff_multiplier
        self.jitter = config.retry_jitter
        self._retry_counts: Dict[str, int] = defaultdict(int)

    def calculate_backoff(self, attempt: int) -> float:
        """Return the delay in seconds for the given *attempt* (0-indexed).

        Args:
            attempt: Zero-based attempt index (0 = first retry).

        Returns:
            Delay in seconds (always ≥ 0).
        """
        delay = self.backoff_base * (self.backoff_multiplier ** attempt)
        jitter_factor = 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, delay * jitter_factor)

    async def execute_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        agent_name: str,
        circuit_breaker: CircuitBreaker | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* up to ``max_retries + 1`` times.

        Args:
            func: Async callable to execute.
            agent_name: Agent label (for logging / stats).
            circuit_breaker: Optional circuit breaker to consult.
            *args: Forwarded to *func*.
            **kwargs: Forwarded to *func*.

        Returns:
            Result of *func*.

        Raises:
            The last exception if all attempts fail.
        """
        last_exc: BaseException | None = None

        for attempt in range(self.max_retries + 1):
            # Check circuit breaker before each attempt
            if circuit_breaker is not None:
                state = circuit_breaker.get_state()
                if state == CircuitBreakerState.OPEN:
                    _logger.info(
                        f"Circuit breaker OPEN for {agent_name}, skipping retry",
                        extra={"agent_name": agent_name},
                    )
                    raise CircuitBreakerOpenError(agent_name)

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.calculate_backoff(attempt)
                    self._retry_counts[agent_name] += 1
                    _logger.info(
                        f"Retry {attempt + 1}/{self.max_retries} for {agent_name} "
                        f"(backoff {delay:.2f}s)",
                        extra={"agent_name": agent_name},
                    )
                    await asyncio.sleep(delay)
                else:
                    _logger.warning(
                        f"Max retries ({self.max_retries}) exceeded for {agent_name}",
                        extra={"agent_name": agent_name},
                    )

        raise last_exc  # type: ignore[misc]

    def get_retry_stats(self) -> Dict[str, int]:
        """Return mapping of agent_name → retry count."""
        return dict(self._retry_counts)

    def reset_stats(self) -> None:
        """Clear retry statistics."""
        self._retry_counts.clear()
