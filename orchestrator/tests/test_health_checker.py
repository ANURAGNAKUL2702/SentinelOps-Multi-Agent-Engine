"""Tests for orchestrator.health_checker."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.health_checker import HealthChecker
from orchestrator.schema import HealthStatusValue


@pytest.fixture
def hc() -> HealthChecker:
    return HealthChecker(OrchestratorConfig(health_check_timeout=0.5))


class TestCheckAgentHealth:
    @pytest.mark.asyncio
    async def test_healthy(self, hc: HealthChecker) -> None:
        agent = MagicMock()
        agent.health_check.return_value = True
        status = await hc.check_agent_health("my_agent", agent)
        assert status.is_healthy
        assert status.status == HealthStatusValue.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy(self, hc: HealthChecker) -> None:
        agent = MagicMock()
        agent.health_check.return_value = False
        status = await hc.check_agent_health("my_agent", agent)
        assert not status.is_healthy
        assert status.status == HealthStatusValue.UNHEALTHY

    @pytest.mark.asyncio
    async def test_no_health_check_method(self, hc: HealthChecker) -> None:
        agent = object()  # no health_check attribute
        status = await hc.check_agent_health("basic_agent", agent)
        assert status.is_healthy
        assert status.status == HealthStatusValue.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_exception(self, hc: HealthChecker) -> None:
        agent = MagicMock()
        agent.health_check.side_effect = RuntimeError("broken")
        status = await hc.check_agent_health("bad_agent", agent)
        assert not status.is_healthy
        assert "broken" in (status.error_message or "")

    @pytest.mark.asyncio
    async def test_health_check_timeout(self) -> None:
        hc = HealthChecker(OrchestratorConfig(health_check_timeout=0.05))
        agent = MagicMock()

        async def slow_check():
            await asyncio.sleep(5)
            return True

        agent.health_check = slow_check
        status = await hc.check_agent_health("slow_agent", agent)
        assert not status.is_healthy


class TestCheckAllAgents:
    @pytest.mark.asyncio
    async def test_check_all(self, hc: HealthChecker) -> None:
        agents = {}
        for name in ["a", "b", "c"]:
            m = MagicMock()
            m.health_check.return_value = True
            agents[name] = m
        results = await hc.check_all_agents(agents)
        assert len(results) == 3
        assert all(s.is_healthy for s in results.values())


class TestIsAgentReady:
    @pytest.mark.asyncio
    async def test_never_checked_is_ready(self, hc: HealthChecker) -> None:
        assert hc.is_agent_ready("unknown") is True

    @pytest.mark.asyncio
    async def test_after_check(self, hc: HealthChecker) -> None:
        agent = MagicMock()
        agent.health_check.return_value = True
        await hc.check_agent_health("tested", agent)
        assert hc.is_agent_ready("tested") is True
