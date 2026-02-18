"""Tests for orchestrator.retry_policy."""

from __future__ import annotations

import asyncio
import time

import pytest

from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.config import OrchestratorConfig
from orchestrator.retry_policy import RetryPolicy
from orchestrator.schema import CircuitBreakerOpenError


@pytest.fixture
def policy() -> RetryPolicy:
    cfg = OrchestratorConfig(
        max_retries=2,
        retry_backoff_base=0.01,
        retry_backoff_multiplier=2.0,
        retry_jitter=0.0,
    )
    return RetryPolicy(cfg)


class TestRetryExecution:
    @pytest.mark.asyncio
    async def test_no_retry_needed(self, policy: RetryPolicy) -> None:
        async def ok():
            return "ok"

        result = await policy.execute_with_retry(ok, "agent")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_once_then_succeed(self, policy: RetryPolicy) -> None:
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")
            return "ok"

        result = await policy.execute_with_retry(flaky, "agent")
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, policy: RetryPolicy) -> None:
        async def always_fail():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            await policy.execute_with_retry(always_fail, "agent")

    @pytest.mark.asyncio
    async def test_retry_stats_tracked(self, policy: RetryPolicy) -> None:
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            return "ok"

        await policy.execute_with_retry(flaky, "my_agent")
        stats = policy.get_retry_stats()
        assert stats["my_agent"] == 2  # 2 retries

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_skips_retry(
        self, policy: RetryPolicy
    ) -> None:
        breaker = CircuitBreaker(
            agent_name="open_agent",
            failure_threshold=1,
            recovery_timeout=60.0,
        )
        # Force open
        breaker.record_failure()
        assert breaker.get_state().value == "open"

        async def should_not_run():
            return "nope"

        with pytest.raises(CircuitBreakerOpenError):
            await policy.execute_with_retry(
                should_not_run, "open_agent", circuit_breaker=breaker
            )


class TestBackoffCalculation:
    def test_exponential_backoff(self, policy: RetryPolicy) -> None:
        d0 = policy.calculate_backoff(0)
        d1 = policy.calculate_backoff(1)
        d2 = policy.calculate_backoff(2)
        assert d0 == pytest.approx(0.01, abs=0.005)
        assert d1 == pytest.approx(0.02, abs=0.005)
        assert d2 == pytest.approx(0.04, abs=0.01)

    def test_jitter_applied(self) -> None:
        cfg = OrchestratorConfig(
            max_retries=2,
            retry_backoff_base=1.0,
            retry_backoff_multiplier=2.0,
            retry_jitter=0.5,
        )
        policy = RetryPolicy(cfg)
        delays = {policy.calculate_backoff(0) for _ in range(20)}
        # With 50% jitter, we should see some variation
        assert len(delays) > 1
