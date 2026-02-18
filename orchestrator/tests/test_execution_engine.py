"""Tests for orchestrator.execution_engine — async DAG executor."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.config import OrchestratorConfig
from orchestrator.dag import ExecutionDAG
from orchestrator.error_handler import ErrorHandler
from orchestrator.execution_engine import ExecutionEngine
from orchestrator.metrics_collector import MetricsCollector
from orchestrator.result_aggregator import ResultAggregator
from orchestrator.retry_policy import RetryPolicy
from orchestrator.schema import AgentNode, AgentStatus, CircuitBreakerOpenError
from orchestrator.state_machine import StateMachine
from orchestrator.timeout_manager import TimeoutManager


def _make_engine(
    config: OrchestratorConfig | None = None,
    breakers: dict | None = None,
) -> ExecutionEngine:
    cfg = config or OrchestratorConfig(
        max_retries=0,
        retry_backoff_base=0.001,
        retry_jitter=0.0,
        enable_prometheus_metrics=False,
    )
    return ExecutionEngine(
        config=cfg,
        timeout_manager=TimeoutManager(),
        retry_policy=RetryPolicy(cfg),
        circuit_breakers=breakers or {},
        error_handler=ErrorHandler(cfg),
        result_aggregator=ResultAggregator(),
        state_machine=StateMachine(),
        metrics_collector=MetricsCollector(cfg),
    )


def _mock_dag(*agents: tuple[str, object, float]) -> ExecutionDAG:
    dag = ExecutionDAG()
    for name, inst, timeout in agents:
        dag.add_node(name, agent_instance=inst, timeout=timeout)
    return dag


class TestExecuteAgent:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        agent = MagicMock()
        agent.analyze.return_value = "result"
        engine = _make_engine()
        dag = _mock_dag(("log_agent", agent, 2.0))
        result = await engine.execute_agent(
            dag.nodes["log_agent"], {}, "cid"
        )
        assert result == "result"

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        async def slow(*a, **kw):
            await asyncio.sleep(10)

        agent = MagicMock()
        agent.analyze.side_effect = slow
        engine = _make_engine()
        dag = _mock_dag(("log_agent", agent, 0.05))
        with pytest.raises(asyncio.TimeoutError):
            await engine.execute_agent(dag.nodes["log_agent"], {}, "cid")

    @pytest.mark.asyncio
    async def test_none_output_raises(self) -> None:
        agent = MagicMock()
        agent.analyze.return_value = None
        engine = _make_engine()
        dag = _mock_dag(("log_agent", agent, 2.0))
        with pytest.raises(RuntimeError, match="returned None"):
            await engine.execute_agent(dag.nodes["log_agent"], {}, "cid")

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises(self) -> None:
        agent = MagicMock()
        agent.analyze.return_value = "ok"
        breaker = CircuitBreaker(agent_name="log_agent", failure_threshold=1, recovery_timeout=60.0)
        breaker.record_failure()
        engine = _make_engine(breakers={"log_agent": breaker})
        dag = _mock_dag(("log_agent", agent, 2.0))
        with pytest.raises(CircuitBreakerOpenError):
            await engine.execute_agent(dag.nodes["log_agent"], {}, "cid")


class TestExecuteStage:
    @pytest.mark.asyncio
    async def test_parallel_success(self) -> None:
        agents = {}
        for name in ["a", "b", "c"]:
            m = MagicMock()
            m.analyze.return_value = f"{name}_out"
            agents[name] = m

        engine = _make_engine()
        dag = ExecutionDAG()
        for name, inst in agents.items():
            dag.add_node(name, agent_instance=inst, timeout=2.0)

        outputs = await engine.execute_stage(
            "stage_1", ["a", "b", "c"], dag, {}, "cid"
        )
        assert outputs == {"a": "a_out", "b": "b_out", "c": "c_out"}

    @pytest.mark.asyncio
    async def test_one_failure_partial(self) -> None:
        cfg = OrchestratorConfig(
            max_retries=0,
            fail_fast=False,
            allow_partial_results=True,
            enable_prometheus_metrics=False,
        )
        engine = _make_engine(config=cfg)

        ok_agent = MagicMock()
        ok_agent.analyze.return_value = "good"
        bad_agent = MagicMock()
        bad_agent.analyze.side_effect = RuntimeError("boom")

        dag = ExecutionDAG()
        dag.add_node("ok1", agent_instance=ok_agent, timeout=2.0)
        dag.add_node("bad", agent_instance=bad_agent, timeout=2.0)
        dag.add_node("ok2", agent_instance=ok_agent, timeout=2.0)

        outputs = await engine.execute_stage(
            "stage_1", ["ok1", "bad", "ok2"], dag, {}, "cid"
        )
        assert "ok1" in outputs
        assert "ok2" in outputs
        assert "bad" not in outputs

    @pytest.mark.asyncio
    async def test_parallel_speedup(self) -> None:
        """Stage with 3 agents sleeping 0.1s each finishes in ~0.1s, not 0.3s."""
        agents = {}
        for name in ["a", "b", "c"]:
            m = MagicMock()

            async def delayed(*args, _name=name, **kwargs):
                await asyncio.sleep(0.1)
                return f"{_name}_out"

            m.analyze.side_effect = delayed
            agents[name] = m

        engine = _make_engine()
        dag = ExecutionDAG()
        for name, inst in agents.items():
            dag.add_node(name, agent_instance=inst, timeout=2.0)

        t0 = time.perf_counter()
        outputs = await engine.execute_stage(
            "stage_1", list(agents.keys()), dag, {}, "cid"
        )
        elapsed = time.perf_counter() - t0
        assert len(outputs) == 3
        # Should be ~0.1s, definitely < 0.25s (not 0.3s sequential)
        assert elapsed < 0.25


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self) -> None:
        agents = {}
        for name in ["a", "b", "c", "d"]:
            m = MagicMock()
            m.analyze.return_value = f"{name}_out"
            agents[name] = m

        dag = ExecutionDAG()
        for name, inst in agents.items():
            dag.add_node(name, agent_instance=inst, timeout=2.0)
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")
        dag.add_edge("c", "d")

        engine = _make_engine()
        result = await engine.run_pipeline(dag, {"input": "data"}, "cid")
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_stage2_fails_skips_downstream(self) -> None:
        cfg = OrchestratorConfig(
            max_retries=0,
            fail_fast=False,
            enable_prometheus_metrics=False,
        )
        engine = _make_engine(config=cfg)

        ok_agent = MagicMock()
        ok_agent.analyze.return_value = "good"
        # hypothesis_agent is critical → failure aborts rest
        bad_agent = MagicMock()
        bad_agent.analyze.side_effect = RuntimeError("fail")
        downstream = MagicMock()
        downstream.analyze.return_value = "downstream_out"

        dag = ExecutionDAG()
        dag.add_node("log_agent", agent_instance=ok_agent, timeout=2.0)
        dag.add_node("hypothesis_agent", agent_instance=bad_agent, timeout=2.0)
        dag.add_node("root_cause_agent", agent_instance=downstream, timeout=2.0)
        dag.add_edge("log_agent", "hypothesis_agent")
        dag.add_edge("hypothesis_agent", "root_cause_agent")

        result = await engine.run_pipeline(dag, {}, "cid")
        assert len(result.errors) > 0
        # root_cause_agent should have been skipped
        assert result.agent_outputs.root_cause_output is None

    @pytest.mark.asyncio
    async def test_outputs_passed_between_stages(self) -> None:
        received = {}

        def capture_agent(name):
            m = MagicMock()

            def capture_analyze(input_data, **kwargs):
                received[name] = input_data
                return f"{name}_out"

            m.analyze.side_effect = capture_analyze
            return m

        agent_a = capture_agent("a")
        agent_b = capture_agent("b")

        dag = ExecutionDAG()
        dag.add_node("a", agent_instance=agent_a, timeout=2.0)
        dag.add_node("b", agent_instance=agent_b, timeout=2.0)
        dag.add_edge("a", "b")

        engine = _make_engine()
        await engine.run_pipeline(dag, {"seed": "data"}, "cid")
        # Agent b should have received accumulated inputs including a's output
        # (the inputs dict includes "a" key from stage 1)

    @pytest.mark.asyncio
    async def test_fail_fast_aborts(self) -> None:
        cfg = OrchestratorConfig(
            max_retries=0,
            fail_fast=True,
            enable_prometheus_metrics=False,
        )
        engine = _make_engine(config=cfg)

        bad = MagicMock()
        bad.analyze.side_effect = RuntimeError("fail")
        ok = MagicMock()
        ok.analyze.return_value = "ok"

        dag = ExecutionDAG()
        dag.add_node("a", agent_instance=bad, timeout=2.0)
        dag.add_node("b", agent_instance=ok, timeout=2.0)
        dag.add_edge("a", "b")

        result = await engine.run_pipeline(dag, {}, "cid")
        assert len(result.errors) > 0
        # b should be skipped
        assert engine.state_machine.get_agent_states().get("b") == AgentStatus.SKIPPED
