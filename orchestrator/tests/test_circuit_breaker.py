"""Tests for orchestrator.circuit_breaker."""

from __future__ import annotations

import asyncio
import time
import threading

import pytest

from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.schema import CircuitBreakerOpenError, CircuitBreakerState


@pytest.fixture
def breaker() -> CircuitBreaker:
    return CircuitBreaker(
        agent_name="test_agent",
        failure_threshold=3,
        recovery_timeout=0.1,
        half_open_max_calls=1,
    )


class TestClosedState:
    @pytest.mark.asyncio
    async def test_success_stays_closed(self, breaker: CircuitBreaker) -> None:
        async def ok():
            return "ok"

        result = await breaker.call(ok)
        assert result == "ok"
        assert breaker.get_state() == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_failures_below_threshold_stay_closed(
        self, breaker: CircuitBreaker
    ) -> None:
        async def fail():
            raise RuntimeError("boom")

        for _ in range(2):  # threshold = 3
            with pytest.raises(RuntimeError):
                await breaker.call(fail)
        assert breaker.get_state() == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(
        self, breaker: CircuitBreaker
    ) -> None:
        async def fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail)
        assert breaker.get_state() == CircuitBreakerState.OPEN


class TestOpenState:
    @pytest.mark.asyncio
    async def test_rejects_calls(self, breaker: CircuitBreaker) -> None:
        async def fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail)

        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(fail)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(
        self, breaker: CircuitBreaker
    ) -> None:
        async def fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail)

        assert breaker.get_state() == CircuitBreakerState.OPEN
        await asyncio.sleep(0.15)  # > recovery_timeout=0.1
        assert breaker.get_state() == CircuitBreakerState.HALF_OPEN


class TestHalfOpenState:
    @pytest.mark.asyncio
    async def test_success_closes(self, breaker: CircuitBreaker) -> None:
        async def fail():
            raise RuntimeError("boom")

        async def ok():
            return "ok"

        # Open it
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail)
        await asyncio.sleep(0.15)
        assert breaker.get_state() == CircuitBreakerState.HALF_OPEN

        # Success closes
        result = await breaker.call(ok)
        assert result == "ok"
        assert breaker.get_state() == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_reopens(self, breaker: CircuitBreaker) -> None:
        async def fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail)
        await asyncio.sleep(0.15)
        assert breaker.get_state() == CircuitBreakerState.HALF_OPEN

        with pytest.raises(RuntimeError):
            await breaker.call(fail)
        assert breaker.get_state() == CircuitBreakerState.OPEN


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_to_closed(self, breaker: CircuitBreaker) -> None:
        async def fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail)
        assert breaker.get_state() == CircuitBreakerState.OPEN
        breaker.reset()
        assert breaker.get_state() == CircuitBreakerState.CLOSED


class TestThreadSafety:
    def test_concurrent_record_failure(self) -> None:
        breaker = CircuitBreaker(
            agent_name="test",
            failure_threshold=100,
            recovery_timeout=60.0,
        )
        errors: list = []

        def stress():
            try:
                for _ in range(50):
                    breaker.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert breaker._failure_count == 250
