"""Shared fixtures for orchestrator tests."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.schema import PipelineStatus


@pytest.fixture
def config() -> OrchestratorConfig:
    """Default orchestrator config with fast timeouts for tests."""
    return OrchestratorConfig(
        log_agent_timeout=2.0,
        metrics_agent_timeout=2.0,
        dependency_agent_timeout=2.0,
        hypothesis_agent_timeout=2.0,
        root_cause_agent_timeout=2.0,
        validation_agent_timeout=2.0,
        incident_commander_timeout=2.0,
        pipeline_timeout=30.0,
        max_retries=1,
        retry_backoff_base=0.01,
        retry_backoff_multiplier=2.0,
        retry_jitter=0.0,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout=0.1,
        circuit_breaker_half_open_max_calls=1,
        enable_prometheus_metrics=False,
        enable_health_checks=False,
    )


def make_mock_agent(name: str = "mock_agent", output: Any = "output") -> MagicMock:
    """Create a MagicMock agent that returns *output* synchronously."""
    agent = MagicMock()
    agent.analyze.return_value = output
    agent.validate.return_value = output
    agent.command.return_value = output
    agent.health_check.return_value = True
    return agent


def make_async_mock_agent(
    name: str = "mock_agent", output: Any = "output"
) -> AsyncMock:
    """Create an AsyncMock agent that returns *output*."""
    agent = AsyncMock(return_value=output)
    return agent


def make_agents(outputs: Dict[str, Any] | None = None) -> Dict[str, MagicMock]:
    """Create 7 mock agents keyed by name."""
    defaults = {
        "log_agent": {"result": "log_analysis"},
        "metrics_agent": {"result": "metrics_analysis"},
        "dependency_agent": {"result": "dependency_analysis"},
        "hypothesis_agent": {"result": "hypothesis"},
        "root_cause_agent": {"result": "root_cause"},
        "validation_agent": {"result": "validation_report"},
        "incident_commander_agent": {"result": "incident_response"},
    }
    if outputs:
        defaults.update(outputs)

    agents: Dict[str, MagicMock] = {}
    for name, out in defaults.items():
        agents[name] = make_mock_agent(name, out)
    return agents
