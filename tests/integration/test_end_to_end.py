"""End-to-end integration tests: simulation → observability → orchestrator → reports."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from integration.config_manager import ConfigManager, SystemConfig
from integration.logger import (
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
    setup_logging,
    get_logger,
)
from integration.pipeline import PipelineRunResult, WarRoomPipeline


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def config() -> SystemConfig:
    """Default SystemConfig with all defaults."""
    return SystemConfig()


@pytest.fixture()
def pipeline(config: SystemConfig) -> WarRoomPipeline:
    return WarRoomPipeline(config)


def _make_mock_pipeline_result(**overrides: Any) -> MagicMock:
    """Create a MagicMock that quacks like PipelineResult."""
    from orchestrator.schema import (
        PipelineStatus,
        AgentOutputs,
        PipelineTelemetry,
    )

    pr = MagicMock()
    pr.status = overrides.get("status", PipelineStatus.SUCCESS)
    pr.execution_time = overrides.get("execution_time", 4.2)
    pr.agent_outputs = overrides.get("agent_outputs", AgentOutputs())
    pr.telemetry = overrides.get("telemetry", PipelineTelemetry())
    # model_dump for reporting stage
    pr.model_dump.return_value = {
        "status": pr.status.value if hasattr(pr.status, "value") else str(pr.status),
        "correlation_id": overrides.get("correlation_id", "e2e-test"),
        "execution_time": pr.execution_time,
        "agent_outputs": {},
        "errors": [],
        "telemetry": {},
    }
    return pr


# ── Correlation IDs ───────────────────────────────────────────────


class TestCorrelationIDs:
    def test_new_correlation_id(self) -> None:
        cid = new_correlation_id()
        assert len(cid) == 36  # UUID4

    def test_get_and_set(self) -> None:
        set_correlation_id("abc-123")
        assert get_correlation_id() == "abc-123"

    def test_logger_creation(self) -> None:
        log = get_logger("test")
        assert log is not None


# ── Logging setup ─────────────────────────────────────────────────


class TestLogging:
    def test_setup_logging_idempotent(self) -> None:
        setup_logging("DEBUG")
        setup_logging("DEBUG")  # should not raise

    def test_setup_logging_invalid_still_works(self) -> None:
        # structlog accepts any level
        setup_logging("INFO")


# ── Stage 1: Simulation ───────────────────────────────────────────


class TestSimulationStage:
    def test_create_scenario_database_timeout(self, pipeline: WarRoomPipeline) -> None:
        result = pipeline.create_scenario("database_timeout")
        assert result["scenario"] == "database_timeout"
        assert "root_cause" in result
        assert "severity" in result

    def test_create_scenario_memory_leak(self, pipeline: WarRoomPipeline) -> None:
        result = pipeline.create_scenario("memory_leak")
        assert result["scenario"] == "memory_leak"

    def test_create_scenario_cpu_spike(self, pipeline: WarRoomPipeline) -> None:
        result = pipeline.create_scenario("cpu_spike")
        assert "metrics" in result
        assert len(result["metrics"]) > 0

    def test_create_scenario_network_latency(self, pipeline: WarRoomPipeline) -> None:
        result = pipeline.create_scenario("network_latency")
        assert "logs" in result
        assert len(result["logs"]) > 0


# ── Stage 2: Observability ────────────────────────────────────────


class TestObservabilityStage:
    def test_generate_observability(self, pipeline: WarRoomPipeline) -> None:
        sim = pipeline.create_scenario("database_timeout")
        obs = pipeline.generate_observability(sim)
        assert "metrics_store" in obs
        assert "log_store" in obs
        assert "query_engine" in obs


# ── Stage 3: Orchestrator (mocked async) ──────────────────────────


class TestOrchestratorStage:
    @patch("integration.pipeline.WarRoomPipeline._build_ground_truth")
    @patch("integration.pipeline.WarRoomPipeline._build_stage1_inputs")
    @patch("integration.pipeline.WarRoomPipeline._instantiate_agents")
    @patch("orchestrator.orchestrator.Orchestrator")
    def test_run_orchestrator(
        self,
        MockOrch: MagicMock,
        mock_agents: MagicMock,
        mock_inputs: MagicMock,
        mock_gt: MagicMock,
        pipeline: WarRoomPipeline,
    ) -> None:
        mock = _make_mock_pipeline_result()
        instance = MockOrch.return_value
        instance.run_pipeline = AsyncMock(return_value=mock)

        mock_agents.return_value = {}
        mock_inputs.return_value = {}
        mock_gt.return_value = MagicMock()

        sim = {"scenario": "cpu_spike", "root_cause": "cpu", "severity": "P1"}
        obs = {"metrics_store": MagicMock(), "log_store": MagicMock()}

        result = pipeline.run_orchestrator(sim, obs, correlation_id="test-id")
        assert result is mock
        instance.run_pipeline.assert_awaited_once()


# ── Full pipeline (mocked heavy stages) ───────────────────────────


class TestFullPipeline:
    @patch("integration.pipeline.WarRoomPipeline.save_to_database")
    @patch("integration.pipeline.WarRoomPipeline.generate_reports")
    @patch("integration.pipeline.WarRoomPipeline.run_orchestrator")
    @patch("integration.pipeline.WarRoomPipeline.generate_observability")
    @patch("integration.pipeline.WarRoomPipeline.create_scenario")
    def test_run_scenario_success(
        self,
        mock_sim: MagicMock,
        mock_obs: MagicMock,
        mock_orch: MagicMock,
        mock_report: MagicMock,
        mock_db: MagicMock,
        config: SystemConfig,
    ) -> None:
        mock_sim.return_value = {
            "scenario": "database_timeout",
            "severity": "P0",
            "root_cause": "pool exhaustion",
            "metrics": [1, 2],
            "logs": [1],
        }
        mock_obs.return_value = {"metrics_store": None, "log_store": None, "query_engine": None}
        mock_orch.return_value = _make_mock_pipeline_result()
        mock_report.return_value = {"html": "reports/out.html"}
        mock_db.return_value = 42

        pipe = WarRoomPipeline(config)
        result = pipe.run_scenario("database_timeout")

        assert result.status == "SUCCESS"
        assert result.scenario == "database_timeout"
        assert result.execution_time > 0
        mock_sim.assert_called_once()

    @patch("integration.pipeline.WarRoomPipeline.create_scenario")
    def test_run_scenario_failure(
        self,
        mock_sim: MagicMock,
        config: SystemConfig,
    ) -> None:
        mock_sim.side_effect = RuntimeError("sim exploded")

        pipe = WarRoomPipeline(config)
        result = pipe.run_scenario("database_timeout")
        assert result.status == "FAILED"
        assert "sim exploded" in result.errors[0]

    @patch("integration.pipeline.WarRoomPipeline.save_to_database")
    @patch("integration.pipeline.WarRoomPipeline.generate_reports")
    @patch("integration.pipeline.WarRoomPipeline.run_orchestrator")
    @patch("integration.pipeline.WarRoomPipeline.generate_observability")
    @patch("integration.pipeline.WarRoomPipeline.create_scenario")
    def test_stage_callback_called(
        self,
        mock_sim: MagicMock,
        mock_obs: MagicMock,
        mock_orch: MagicMock,
        mock_report: MagicMock,
        mock_db: MagicMock,
        config: SystemConfig,
    ) -> None:
        mock_sim.return_value = {
            "scenario": "cpu_spike",
            "severity": "P1",
            "root_cause": "cpu",
            "metrics": [],
            "logs": [],
        }
        mock_obs.return_value = {}
        mock_orch.return_value = _make_mock_pipeline_result()
        mock_report.return_value = {}
        mock_db.return_value = 1

        stages_seen: list[str] = []
        pipe = WarRoomPipeline(config)
        pipe.run_scenario("cpu_spike", on_stage=stages_seen.append)

        assert "simulation" in stages_seen
        assert "observability" in stages_seen
        assert "analysis" in stages_seen
        assert "reporting" in stages_seen


# ── Database-skipping ──────────────────────────────────────────────


class TestDatabaseSkipping:
    @patch("integration.pipeline.WarRoomPipeline.generate_reports")
    @patch("integration.pipeline.WarRoomPipeline.run_orchestrator")
    @patch("integration.pipeline.WarRoomPipeline.generate_observability")
    @patch("integration.pipeline.WarRoomPipeline.create_scenario")
    def test_no_db_when_disabled(
        self,
        mock_sim: MagicMock,
        mock_obs: MagicMock,
        mock_orch: MagicMock,
        mock_report: MagicMock,
        config: SystemConfig,
    ) -> None:
        mock_sim.return_value = {
            "scenario": "x", "severity": "P2", "root_cause": "x",
            "metrics": [], "logs": [],
        }
        mock_obs.return_value = {}
        mock_orch.return_value = _make_mock_pipeline_result()
        mock_report.return_value = {}

        pipe = WarRoomPipeline(config)
        result = pipe.run_scenario("cpu_spike", save_to_db=False)

        assert result.incident_id is None


# ── ConfigManager integration ─────────────────────────────────────


class TestConfigIntegration:
    def test_load_default_config_creates_pipeline(self, config: SystemConfig) -> None:
        pipe = WarRoomPipeline(config)
        assert pipe.config is config

    def test_config_scenarios_match_simulation(self) -> None:
        from simulation.failure_injector import get_available_scenarios

        available = get_available_scenarios()
        cfg = SystemConfig()
        for s in cfg.simulation.available_scenarios:
            assert s in available, f"Config scenario '{s}' not in simulation engine"
