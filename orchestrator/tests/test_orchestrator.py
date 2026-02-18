"""Tests for orchestrator.orchestrator — end-to-end pipeline tests."""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.orchestrator import Orchestrator
from orchestrator.schema import CircuitBreakerState, PipelineStatus


def _make_agents(overrides: dict | None = None) -> dict[str, MagicMock]:
    """Create 7 mock agents returning simple dicts."""
    defaults = {
        "log_agent": {"result": "log"},
        "metrics_agent": {"result": "metrics"},
        "dependency_agent": {"result": "dependency"},
        "hypothesis_agent": {"result": "hypothesis"},
        "root_cause_agent": {"result": "root_cause"},
        "validation_agent": {"result": "validation"},
        "incident_commander_agent": {"result": "commander"},
    }
    if overrides:
        defaults.update(overrides)
    agents: dict[str, MagicMock] = {}
    for name, out in defaults.items():
        m = MagicMock()
        m.analyze.return_value = out
        m.validate.return_value = out
        m.command.return_value = out
        agents[name] = m
    return agents


def _make_orchestrator(
    agents: dict | None = None,
    config: OrchestratorConfig | None = None,
) -> Orchestrator:
    cfg = config or OrchestratorConfig(
        max_retries=0,
        retry_backoff_base=0.001,
        retry_jitter=0.0,
        pipeline_timeout=10.0,
        enable_prometheus_metrics=False,
        enable_health_checks=False,
    )
    return Orchestrator(config=cfg, agents=agents or _make_agents())


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_all_success(self) -> None:
        orch = _make_orchestrator()
        result = await orch.run_pipeline(
            scenario={"name": "test"},
            observability_data={"logs": []},
        )
        assert result.status == PipelineStatus.SUCCESS
        assert result.execution_time > 0
        assert len(result.errors) == 0
        # All outputs populated
        ao = result.agent_outputs
        assert ao.log_output is not None
        assert ao.metrics_output is not None
        assert ao.dependency_output is not None
        assert ao.hypothesis_output is not None
        assert ao.root_cause_output is not None
        assert ao.validation_output is not None
        assert ao.incident_response is not None

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(self) -> None:
        cid = str(uuid.uuid4())
        orch = _make_orchestrator()
        result = await orch.run_pipeline(
            scenario={}, observability_data={}, correlation_id=cid
        )
        assert result.correlation_id == cid

    @pytest.mark.asyncio
    async def test_auto_generated_correlation_id(self) -> None:
        orch = _make_orchestrator()
        result = await orch.run_pipeline(scenario={}, observability_data={})
        uuid.UUID(result.correlation_id)  # valid UUID

    @pytest.mark.asyncio
    async def test_log_agent_fails_partial_success(self) -> None:
        agents = _make_agents()
        agents["log_agent"].analyze.side_effect = RuntimeError("boom")
        orch = _make_orchestrator(agents=agents)
        result = await orch.run_pipeline(scenario={}, observability_data={})
        # log_agent is non-critical, so pipeline continues
        assert result.status in (PipelineStatus.PARTIAL_SUCCESS, PipelineStatus.SUCCESS)
        assert result.agent_outputs.log_output is None

    @pytest.mark.asyncio
    async def test_hypothesis_fails_pipeline_fails(self) -> None:
        agents = _make_agents()
        agents["hypothesis_agent"].analyze.side_effect = RuntimeError("fail")
        orch = _make_orchestrator(agents=agents)
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert result.status == PipelineStatus.FAILED
        assert result.agent_outputs.hypothesis_output is None
        assert result.agent_outputs.root_cause_output is None  # skipped

    @pytest.mark.asyncio
    async def test_pipeline_timeout(self) -> None:
        agents = _make_agents()

        async def slow(*a, **kw):
            await asyncio.sleep(10)

        agents["log_agent"].analyze.side_effect = slow
        cfg = OrchestratorConfig(
            pipeline_timeout=0.1,
            log_agent_timeout=5.0,
            max_retries=0,
            retry_backoff_base=0.001,
            enable_prometheus_metrics=False,
        )
        orch = _make_orchestrator(agents=agents, config=cfg)
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert result.status == PipelineStatus.TIMEOUT


class TestCircuitBreakers:
    @pytest.mark.asyncio
    async def test_circuit_breaker_states_returned(self) -> None:
        orch = _make_orchestrator()
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert "log_agent" in result.circuit_breaker_states
        assert result.circuit_breaker_states["log_agent"] == CircuitBreakerState.CLOSED


class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self) -> None:
        agents = _make_agents()
        call_count = 0

        def flaky(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient")
            return {"result": "ok"}

        agents["log_agent"].analyze.side_effect = flaky

        cfg = OrchestratorConfig(
            max_retries=2,
            retry_backoff_base=0.001,
            retry_jitter=0.0,
            enable_prometheus_metrics=False,
        )
        orch = _make_orchestrator(agents=agents, config=cfg)
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert result.agent_outputs.log_output is not None


class TestTelemetry:
    @pytest.mark.asyncio
    async def test_telemetry_populated(self) -> None:
        orch = _make_orchestrator()
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert result.telemetry is not None
        assert result.telemetry.total_llm_cost >= 0
        assert len(result.telemetry.agent_latencies) > 0


class TestMetadata:
    @pytest.mark.asyncio
    async def test_metadata_present(self) -> None:
        orch = _make_orchestrator()
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert result.metadata is not None
        assert result.metadata.start_time is not None
        assert result.metadata.end_time is not None


class TestStageResults:
    @pytest.mark.asyncio
    async def test_stage_results_populated(self) -> None:
        orch = _make_orchestrator()
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert "stage_1" in result.stage_results
        sr1 = result.stage_results["stage_1"]
        assert set(sr1.agents) == {
            "log_agent", "metrics_agent", "dependency_agent"
        }
        assert sr1.status == "SUCCESS"

    @pytest.mark.asyncio
    async def test_all_five_stages(self) -> None:
        orch = _make_orchestrator()
        result = await orch.run_pipeline(scenario={}, observability_data={})
        assert len(result.stage_results) == 5


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        agents = _make_agents()
        orch = _make_orchestrator(agents=agents)
        health = await orch.health_check()
        assert len(health) == 7
        assert all(v is True for v in health.values())


class TestExportMetrics:
    def test_export_disabled(self) -> None:
        orch = _make_orchestrator()
        assert orch.export_metrics() == ""

    def test_export_enabled(self) -> None:
        cfg = OrchestratorConfig(
            max_retries=0,
            retry_backoff_base=0.001,
            enable_prometheus_metrics=True,
        )
        orch = _make_orchestrator(config=cfg)
        # Just confirm it returns a string (metrics may be empty)
        text = orch.export_metrics()
        assert isinstance(text, str)


class TestConcurrentPipelines:
    @pytest.mark.asyncio
    async def test_concurrent_runs_no_interference(self) -> None:
        """Run 5 pipelines concurrently, all get unique correlation IDs."""
        cids: list[str] = []

        async def run_one():
            orch = _make_orchestrator()
            result = await orch.run_pipeline(scenario={}, observability_data={})
            return result.correlation_id

        tasks = [run_one() for _ in range(5)]
        cids = await asyncio.gather(*tasks)
        assert len(set(cids)) == 5  # all unique
