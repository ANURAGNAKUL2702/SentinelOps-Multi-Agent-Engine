"""Performance tests for the orchestrator."""

from __future__ import annotations

import asyncio
import time
import tracemalloc
from unittest.mock import MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.orchestrator import Orchestrator


def _make_agents() -> dict[str, MagicMock]:
    defaults = {
        "log_agent": {"result": "log"},
        "metrics_agent": {"result": "metrics"},
        "dependency_agent": {"result": "dependency"},
        "hypothesis_agent": {"result": "hypothesis"},
        "root_cause_agent": {"result": "root_cause"},
        "validation_agent": {"result": "validation"},
        "incident_commander_agent": {"result": "commander"},
    }
    agents: dict[str, MagicMock] = {}
    for name, out in defaults.items():
        m = MagicMock()
        m.analyze.return_value = out
        m.validate.return_value = out
        m.command.return_value = out
        agents[name] = m
    return agents


def _make_orchestrator(config: OrchestratorConfig | None = None) -> Orchestrator:
    cfg = config or OrchestratorConfig(
        max_retries=0,
        retry_backoff_base=0.001,
        retry_jitter=0.0,
        pipeline_timeout=30.0,
        enable_prometheus_metrics=False,
    )
    return Orchestrator(config=cfg, agents=_make_agents())


class TestPerformance:
    @pytest.mark.asyncio
    async def test_full_pipeline_latency_under_3s(self) -> None:
        """With mock agents, pipeline should complete in well under 3s."""
        orch = _make_orchestrator()
        t0 = time.perf_counter()
        result = await orch.run_pipeline(scenario={}, observability_data={})
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0
        assert result.execution_time < 3.0

    @pytest.mark.asyncio
    async def test_parallel_speedup_ratio(self) -> None:
        """Stage 1 (3 agents sleeping 0.1s) should finish in ~0.1s, not 0.3s."""
        agents = _make_agents()
        for name in ["log_agent", "metrics_agent", "dependency_agent"]:
            async def delay(*a, _n=name, **kw):
                await asyncio.sleep(0.1)
                return {"result": _n}
            agents[name].analyze.side_effect = delay

        orch = Orchestrator(
            config=OrchestratorConfig(
                max_retries=0,
                retry_backoff_base=0.001,
                pipeline_timeout=10.0,
                enable_prometheus_metrics=False,
            ),
            agents=agents,
        )
        result = await orch.run_pipeline(scenario={}, observability_data={})
        sr1 = result.stage_results.get("stage_1")
        assert sr1 is not None
        # Parallel: should be ~0.1s, definitely < 0.25s
        assert sr1.duration < 0.25
        # Speedup ratio
        latencies = result.telemetry.agent_latencies
        if latencies:
            stage1_agents = ["log_agent", "metrics_agent", "dependency_agent"]
            sum_individual = sum(latencies.get(a, 0) for a in stage1_agents)
            if sr1.duration > 0 and sum_individual > 0:
                speedup = sum_individual / sr1.duration
                assert speedup > 2.0

    @pytest.mark.asyncio
    async def test_memory_usage_under_500mb(self) -> None:
        tracemalloc.start()
        orch = _make_orchestrator()
        await orch.run_pipeline(scenario={}, observability_data={})
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500

    @pytest.mark.asyncio
    async def test_concurrent_pipelines_no_deadlock(self) -> None:
        """10 concurrent pipelines should all complete."""

        async def run_one():
            orch = _make_orchestrator()
            return await orch.run_pipeline(scenario={}, observability_data={})

        results = await asyncio.gather(*[run_one() for _ in range(10)])
        assert len(results) == 10
        cids = [r.correlation_id for r in results]
        assert len(set(cids)) == 10  # all unique

    @pytest.mark.asyncio
    async def test_prometheus_overhead_minimal(self) -> None:
        """Prometheus metrics add < 50% overhead (generous for CI)."""
        # Without metrics
        orch_no = _make_orchestrator(
            config=OrchestratorConfig(
                max_retries=0,
                retry_backoff_base=0.001,
                enable_prometheus_metrics=False,
            )
        )
        t0 = time.perf_counter()
        for _ in range(3):
            await orch_no.run_pipeline(scenario={}, observability_data={})
        time_no = time.perf_counter() - t0

        # With metrics
        orch_yes = _make_orchestrator(
            config=OrchestratorConfig(
                max_retries=0,
                retry_backoff_base=0.001,
                enable_prometheus_metrics=True,
            )
        )
        t0 = time.perf_counter()
        for _ in range(3):
            await orch_yes.run_pipeline(scenario={}, observability_data={})
        time_yes = time.perf_counter() - t0

        # Prometheus overhead < 50%
        if time_no > 0:
            overhead = (time_yes - time_no) / time_no
            assert overhead < 0.5
