"""Tests for orchestrator.schema — Pydantic v2 models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from orchestrator.schema import (
    AgentError,
    AgentOutputs,
    AgentStatus,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    CycleDetectedError,
    HealthStatus,
    HealthStatusValue,
    PipelineMetadata,
    PipelineResult,
    PipelineStatus,
    PipelineTelemetry,
    StageResult,
    StateMachineError,
    AgentNode,
)


class TestEnums:
    def test_pipeline_status_values(self) -> None:
        assert PipelineStatus.SUCCESS.value == "success"
        assert PipelineStatus.PARTIAL_SUCCESS.value == "partial_success"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.TIMEOUT.value == "timeout"

    def test_circuit_breaker_state_values(self) -> None:
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"

    def test_agent_status_values(self) -> None:
        assert len(AgentStatus) == 7

    def test_health_status_value(self) -> None:
        assert HealthStatusValue.HEALTHY.value == "healthy"


class TestStageResult:
    def test_frozen(self) -> None:
        now = datetime.now(timezone.utc)
        sr = StageResult(
            stage_name="stage_1",
            agents=["log_agent"],
            duration=1.5,
            status="SUCCESS",
            start_time=now,
            end_time=now,
        )
        assert sr.stage_name == "stage_1"
        with pytest.raises(Exception):
            sr.stage_name = "nope"  # type: ignore[misc]

    def test_duration_ge_zero(self) -> None:
        with pytest.raises(Exception):
            StageResult(
                stage_name="s",
                agents=[],
                duration=-1.0,
                status="FAILED",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
            )


class TestAgentError:
    def test_default_timestamp(self) -> None:
        err = AgentError(
            agent_name="log_agent",
            error_type="TIMEOUT",
            error_message="timed out",
        )
        assert err.timestamp is not None
        assert err.retries_attempted == 0


class TestPipelineTelemetry:
    def test_defaults(self) -> None:
        t = PipelineTelemetry()
        assert t.total_llm_cost == 0.0
        assert t.total_llm_calls == 0
        assert t.agent_latencies == {}
        assert t.parallel_speedup == 1.0


class TestAgentOutputs:
    def test_all_none(self) -> None:
        ao = AgentOutputs()
        assert ao.log_output is None
        assert ao.incident_response is None

    def test_partial(self) -> None:
        ao = AgentOutputs(log_output="log_data")
        assert ao.log_output == "log_data"
        assert ao.metrics_output is None


class TestPipelineResult:
    def test_defaults(self) -> None:
        pr = PipelineResult(status=PipelineStatus.FAILED, execution_time=0.0)
        assert pr.status == PipelineStatus.FAILED
        uuid.UUID(pr.correlation_id)  # valid UUID

    def test_frozen(self) -> None:
        pr = PipelineResult(status=PipelineStatus.SUCCESS, execution_time=1.0)
        with pytest.raises(Exception):
            pr.status = PipelineStatus.FAILED  # type: ignore[misc]


class TestPipelineMetadata:
    def test_creation(self) -> None:
        now = datetime.now(timezone.utc)
        meta = PipelineMetadata(start_time=now, end_time=now)
        assert meta.pipeline_version == "1.0.0"


class TestHealthStatus:
    def test_defaults(self) -> None:
        hs = HealthStatus(agent_name="test", is_healthy=True)
        assert hs.status == HealthStatusValue.UNKNOWN
        assert hs.error_message is None


class TestExceptions:
    def test_circuit_breaker_open_error(self) -> None:
        err = CircuitBreakerOpenError("hypothesis_agent")
        assert "hypothesis_agent" in str(err)
        assert err.agent_name == "hypothesis_agent"

    def test_cycle_detected_error(self) -> None:
        err = CycleDetectedError(["A", "B", "A"])
        assert err.cycle == ["A", "B", "A"]

    def test_state_machine_error(self) -> None:
        err = StateMachineError("COMPLETED", "RUNNING")
        assert "COMPLETED" in str(err)


class TestAgentNode:
    def test_creation(self) -> None:
        node = AgentNode(name="test", timeout=5.0)
        assert node.name == "test"
        assert node.agent_instance is None
