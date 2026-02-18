"""Per-agent circuit breaker — CLOSED → OPEN → HALF_OPEN → CLOSED cycle."""

from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Callable, Coroutine

from .schema import CircuitBreakerState, CircuitBreakerOpenError
from .telemetry import get_logger

_logger = get_logger(__name__)


class CircuitBreaker:
    """Circuit breaker that wraps async agent calls.

    State machine:
        CLOSED  — normal operation; failures tracked.
        OPEN    — failing; all calls rejected immediately.
        HALF_OPEN — recovery probe; allow *half_open_max_calls* test calls.
    """

    def __init__(
        self,
        agent_name: str = "",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.agent_name = agent_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    # ---- public API -------------------------------------------------------

    async def call(self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker.

        Args:
            func: Async callable to execute.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            Result of *func*.

        Raises:
            CircuitBreakerOpenError: If breaker is OPEN and recovery timeout
                has not elapsed.
        """
        self._maybe_transition_to_half_open()
        state = self.get_state()

        if state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(self.agent_name)

        if state == CircuitBreakerState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(self.agent_name)
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure()
            raise

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Recovery test passed — close breaker
                _logger.info(
                    "Circuit breaker recovery succeeded, closing",
                    extra={"agent_name": self.agent_name},
                )
                self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count += 1
            self._half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed call and potentially open the breaker."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitBreakerState.HALF_OPEN:
                _logger.warning(
                    "Circuit breaker recovery failed, re-opening",
                    extra={"agent_name": self.agent_name},
                )
                self._state = CircuitBreakerState.OPEN
                self._half_open_calls = 0
            elif self._failure_count >= self.failure_threshold:
                _logger.warning(
                    f"Circuit breaker opened after {self._failure_count} failures",
                    extra={"agent_name": self.agent_name},
                )
                self._state = CircuitBreakerState.OPEN

    def get_state(self) -> CircuitBreakerState:
        """Return current breaker state."""
        self._maybe_transition_to_half_open()
        with self._lock:
            return self._state

    def reset(self) -> None:
        """Force-reset the breaker to CLOSED."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = 0.0

    # ---- internal ---------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """If OPEN and recovery_timeout has elapsed, → HALF_OPEN."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    _logger.info(
                        "Circuit breaker transitioning to HALF_OPEN",
                        extra={"agent_name": self.agent_name},
                    )
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
