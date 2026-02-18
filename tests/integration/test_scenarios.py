"""Tests for individual simulation scenarios through the integration pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from integration.config_manager import SystemConfig
from integration.pipeline import WarRoomPipeline, PipelineRunResult
from integration.cli import (
    validate_scenario,
    validate_format,
    SCENARIO_META,
    format_duration,
    format_confidence,
    format_file_size,
)


# ── Scenario validation ───────────────────────────────────────────


class TestScenarioValidation:
    ALL = ["memory_leak", "cpu_spike", "database_timeout", "network_latency"]

    @pytest.mark.parametrize("name", ALL)
    def test_valid_scenario(self, name: str) -> None:
        assert validate_scenario(name, self.ALL) is True

    def test_invalid_scenario(self) -> None:
        assert validate_scenario("unknown", self.ALL) is False

    def test_empty_string(self) -> None:
        assert validate_scenario("", self.ALL) is False

    def test_case_sensitive(self) -> None:
        assert validate_scenario("Memory_Leak", self.ALL) is False


# ── Format validation ─────────────────────────────────────────────


class TestFormatValidation:
    @pytest.mark.parametrize("fmt", ["html", "markdown", "json", "pdf"])
    def test_valid_format(self, fmt: str) -> None:
        assert validate_format(fmt) is True

    def test_invalid_format(self) -> None:
        assert validate_format("xlsx") is False

    def test_case_insensitive(self) -> None:
        assert validate_format("HTML") is True
        assert validate_format("Json") is True


# ── SCENARIO_META ──────────────────────────────────────────────────


class TestScenarioMeta:
    def test_all_scenarios_have_metadata(self) -> None:
        for s in ("memory_leak", "cpu_spike", "database_timeout", "network_latency"):
            assert s in SCENARIO_META

    def test_metadata_has_description(self) -> None:
        for meta in SCENARIO_META.values():
            assert "description" in meta
            assert len(meta["description"]) > 5

    def test_metadata_has_severity(self) -> None:
        for meta in SCENARIO_META.values():
            assert "severity" in meta

    def test_metadata_has_duration(self) -> None:
        for meta in SCENARIO_META.values():
            assert "duration" in meta


# ── Per-scenario simulation (integration) ─────────────────────────

@pytest.fixture()
def config() -> SystemConfig:
    return SystemConfig()


class TestScenarioSimulation:
    """Run each scenario through the simulation engine directly."""

    @pytest.mark.parametrize("scenario", [
        "memory_leak",
        "cpu_spike",
        "database_timeout",
        "network_latency",
    ])
    def test_simulation_produces_output(self, scenario: str) -> None:
        from simulation import run_simulation

        output = run_simulation(
            scenario=scenario,
            duration_minutes=5,
            metrics_interval_seconds=60,
            log_interval_seconds=60,
        )
        assert isinstance(output, dict)
        assert "scenario" in output
        assert "severity" in output
        assert "root_cause" in output
        assert "metrics" in output
        assert "logs" in output
        assert len(output["metrics"]) > 0
        assert len(output["logs"]) > 0

    @pytest.mark.parametrize("scenario", [
        "memory_leak",
        "cpu_spike",
        "database_timeout",
        "network_latency",
    ])
    def test_simulation_blast_radius(self, scenario: str) -> None:
        from simulation import run_simulation

        output = run_simulation(scenario=scenario, duration_minutes=5)
        br = output.get("blast_radius")
        assert br is not None
        total = br["total_affected"]
        direct = br["directly_affected"]
        # directly_affected may be a list of names or an int count
        direct_count = len(direct) if isinstance(direct, list) else direct
        total_count = len(total) if isinstance(total, list) else total
        assert total_count >= direct_count

    @pytest.mark.parametrize("scenario", [
        "memory_leak",
        "cpu_spike",
        "database_timeout",
        "network_latency",
    ])
    def test_observability_from_scenario(self, scenario: str) -> None:
        from simulation import run_simulation
        from observability import build_observability_from_simulation

        sim = run_simulation(scenario=scenario, duration_minutes=5)
        obs = build_observability_from_simulation(sim)

        assert "metrics_store" in obs
        assert "log_store" in obs
        assert "query_engine" in obs


# ── Pipeline stage 1 (via WarRoomPipeline.create_scenario) ────────


class TestPipelineCreateScenario:
    def test_create_scenario_returns_dict(self, config: SystemConfig) -> None:
        pipeline = WarRoomPipeline(config)
        result = pipeline.create_scenario("database_timeout")
        assert isinstance(result, dict)
        assert "scenario" in result

    def test_create_scenario_has_services(self, config: SystemConfig) -> None:
        pipeline = WarRoomPipeline(config)
        result = pipeline.create_scenario("memory_leak")
        assert "services" in result
        assert len(result["services"]) > 0


# ── Formatting helpers ─────────────────────────────────────────────


class TestFormattingHelpers:
    def test_format_duration_seconds(self) -> None:
        assert "s" in format_duration(5.3)
        assert "5.3" in format_duration(5.3)

    def test_format_duration_minutes(self) -> None:
        result = format_duration(125.0)
        assert "m" in result
        assert "2" in result

    def test_format_confidence_high(self) -> None:
        result = format_confidence(0.92)
        assert "92" in result
        assert "green" in result

    def test_format_confidence_medium(self) -> None:
        result = format_confidence(0.65)
        assert "65" in result
        assert "yellow" in result

    def test_format_confidence_low(self) -> None:
        result = format_confidence(0.3)
        assert "30" in result
        assert "red" in result

    def test_format_file_size_bytes(self) -> None:
        assert "B" in format_file_size(500)

    def test_format_file_size_kb(self) -> None:
        assert "KB" in format_file_size(5120)

    def test_format_file_size_mb(self) -> None:
        assert "MB" in format_file_size(2 * 1024 * 1024)


# ── PipelineRunResult ──────────────────────────────────────────────


class TestPipelineRunResult:
    def test_default_values(self) -> None:
        r = PipelineRunResult()
        assert r.status == "SUCCESS"
        assert r.errors == []
        assert r.report_paths == {}
        assert r.incident_id is None

    def test_custom_values(self) -> None:
        r = PipelineRunResult(
            status="FAILED",
            scenario="cpu_spike",
            root_cause="high load",
            confidence=0.99,
        )
        assert r.status == "FAILED"
        assert r.scenario == "cpu_spike"
        assert r.confidence == 0.99
