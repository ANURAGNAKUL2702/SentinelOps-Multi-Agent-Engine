"""Tests for orchestrator.result_aggregator."""

from __future__ import annotations

import pytest

from orchestrator.result_aggregator import ResultAggregator
from orchestrator.schema import AgentOutputs, PipelineTelemetry, StageResult
from datetime import datetime, timezone


class TestAddOutput:
    def test_add_agent_output(self) -> None:
        agg = ResultAggregator()
        agg.add_agent_output("log_agent", {"data": "logs"}, 0.5)
        outputs = agg.aggregate()
        assert outputs.log_output == {"data": "logs"}

    def test_all_agents(self) -> None:
        agg = ResultAggregator()
        names = [
            "log_agent", "metrics_agent", "dependency_agent",
            "hypothesis_agent", "root_cause_agent",
            "validation_agent", "incident_commander_agent",
        ]
        for name in names:
            agg.add_agent_output(name, f"{name}_output", 1.0)
        outputs = agg.aggregate()
        assert outputs.log_output == "log_agent_output"
        assert outputs.incident_response == "incident_commander_agent_output"

    def test_missing_agent_is_none(self) -> None:
        agg = ResultAggregator()
        agg.add_agent_output("log_agent", "data", 0.5)
        outputs = agg.aggregate()
        assert outputs.metrics_output is None

    def test_empty_aggregate_all_none(self) -> None:
        agg = ResultAggregator()
        outputs = agg.aggregate()
        assert outputs.log_output is None
        assert outputs.incident_response is None

    def test_duplicate_overwrites(self) -> None:
        agg = ResultAggregator()
        agg.add_agent_output("log_agent", "first", 0.5)
        agg.add_agent_output("log_agent", "second", 0.6)
        outputs = agg.aggregate()
        assert outputs.log_output == "second"


class TestAddError:
    def test_add_error(self) -> None:
        agg = ResultAggregator()
        agg.add_agent_error("log_agent", RuntimeError("fail"), "UNKNOWN")
        errors = agg.get_errors()
        assert len(errors) == 1
        assert errors[0].agent_name == "log_agent"

    def test_multiple_errors(self) -> None:
        agg = ResultAggregator()
        agg.add_agent_error("a", RuntimeError("1"), "TIMEOUT")
        agg.add_agent_error("b", RuntimeError("2"), "LLM_ERROR")
        assert len(agg.get_errors()) == 2


class TestTelemetry:
    def test_calculate_telemetry(self) -> None:
        agg = ResultAggregator()
        agg.add_agent_output("log_agent", "data", 0.5)
        agg.add_agent_output("metrics_agent", "data", 1.0)
        agg.record_llm_cost("log_agent", 0.001, calls=1)
        t = agg.calculate_telemetry()
        assert t.total_llm_cost == pytest.approx(0.001)
        assert t.total_llm_calls == 1
        assert t.agent_latencies["log_agent"] == 0.5

    def test_parallel_speedup(self) -> None:
        agg = ResultAggregator()
        now = datetime.now(timezone.utc)
        agg.add_agent_output("log_agent", "d", 0.5)
        agg.add_agent_output("metrics_agent", "d", 0.8)
        agg.add_agent_output("dependency_agent", "d", 0.6)
        agg.add_stage_result(StageResult(
            stage_name="stage_1",
            agents=["log_agent", "metrics_agent", "dependency_agent"],
            duration=0.8,
            status="SUCCESS",
            start_time=now,
            end_time=now,
        ))
        t = agg.calculate_telemetry()
        # (0.5 + 0.8 + 0.6) / 0.8 = 2.375
        assert t.parallel_speedup > 2.0


class TestReset:
    def test_reset_clears(self) -> None:
        agg = ResultAggregator()
        agg.add_agent_output("log_agent", "data", 0.5)
        agg.add_agent_error("x", RuntimeError("e"), "UNKNOWN")
        agg.reset()
        assert agg.aggregate().log_output is None
        assert agg.get_errors() == []
