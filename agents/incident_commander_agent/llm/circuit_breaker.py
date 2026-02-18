"""
File: llm/circuit_breaker.py
Purpose: Thread-safe circuit breaker for LLM calls.
Dependencies: Standard library only.
Performance: O(1) state checks.

State transitions::

    CLOSED --[N failures]--> OPEN --[cooldown]--> HALF_OPEN
    HALF_OPEN --[M successes]--> CLOSED
    HALF_OPEN --[failure]--> OPEN
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Optional

from agents.incident_commander_agent.telemetry import (
    TelemetryCollector,
    get_logger,
)

logger = get_logger("incident_commander_agent.llm.circuit_breaker")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker for LLM provider calls.

    Args:
        failure_threshold: Consecutive failures to open circuit.
        cooldown_seconds: Seconds before half-open attempt.
        success_threshold: Consecutive successes to close circuit.
        telemetry: Optional telemetry collector.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        success_threshold: int = 2,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._success_threshold = success_threshold
        self._telemetry = telemetry
        self._lock = threading.Lock()

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current circuit state (evaluates half-open on read)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._consecutive_successes = 0
            return self._state

    def can_execute(self) -> bool:
        """Check if a call is allowed."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful LLM call."""
        with self._lock:
            self._consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._success_threshold:
                    self._state = CircuitState.CLOSED
            else:
                self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed LLM call."""
        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()
                if self._telemetry:
                    self._telemetry.circuit_breaker_trips.inc()
            elif self._consecutive_failures >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()
                if self._telemetry:
                    self._telemetry.circuit_breaker_trips.inc()

    def reset(self) -> None:
        """Reset to CLOSED state (for testing)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._last_failure_time = 0.0
