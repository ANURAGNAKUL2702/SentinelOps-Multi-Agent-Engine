"""Tests for orchestrator.metrics_collector."""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.metrics_collector import MetricsCollector
from orchestrator.schema import PipelineStatus


class TestRecording:
    def test_record_agent_execution(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_agent_execution("log_agent", 0.5, "success")
        assert mc._agent_durations["log_agent"] == 0.5

    def test_record_pipeline_result(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_pipeline_result(PipelineStatus.SUCCESS, 2.0)
        assert mc._pipeline_count["success"] == 1

    def test_record_timeout(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_timeout("agent_x")
        mc.record_timeout("agent_x")
        assert mc._timeout_counts["agent_x"] == 2

    def test_record_retry(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_retry("agent_y")
        assert mc._retry_counts["agent_y"] == 1

    def test_record_llm_cost(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_llm_cost("log_agent", 0.001)
        mc.record_llm_cost("log_agent", 0.002)
        assert mc._llm_costs["log_agent"] == pytest.approx(0.003)
        assert mc._llm_calls["log_agent"] == 2

    def test_record_circuit_breaker_state_change(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_circuit_breaker_state_change("a", "closed", "open")
        assert mc._cb_trips == 1


class TestTelemetry:
    def test_get_telemetry_populated(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_agent_execution("log_agent", 0.5, "success")
        mc.record_llm_cost("log_agent", 0.01)
        mc.record_timeout("metrics_agent")
        t = mc.get_telemetry()
        assert t.total_llm_cost == pytest.approx(0.01)
        assert t.total_llm_calls == 1
        assert t.agent_latencies["log_agent"] == 0.5
        assert t.timeout_violations == 1


class TestExportMetrics:
    def test_disabled_returns_empty(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        assert mc.export_metrics() == ""

    def test_enabled_returns_prometheus_format(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=True))
        mc.record_agent_execution("log_agent", 0.5, "success")
        mc.record_pipeline_result(PipelineStatus.SUCCESS, 2.0)
        text = mc.export_metrics()
        assert "agent_execution_seconds" in text
        assert "pipeline_executions_total" in text

    def test_prometheus_7_metric_types(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=True))
        mc.record_agent_execution("a", 1.0, "success")
        mc.record_pipeline_result(PipelineStatus.SUCCESS, 1.0)
        mc.record_agent_failure("a", "TIMEOUT")
        mc.record_circuit_breaker_state_change("a", "closed", "open")
        mc.record_timeout("a")
        mc.record_retry("a")
        mc.record_llm_cost("a", 0.01)
        text = mc.export_metrics()
        expected_metrics = [
            "agent_execution_seconds",
            "pipeline_executions_total",
            "agent_failures_total",
            "circuit_breaker_state",
            "timeout_violations_total",
            "retry_attempts_total",
            "llm_cost_dollars_total",
        ]
        for metric in expected_metrics:
            assert metric in text, f"Missing metric: {metric}"


class TestReset:
    def test_reset_clears(self) -> None:
        mc = MetricsCollector(OrchestratorConfig(enable_prometheus_metrics=False))
        mc.record_agent_execution("a", 1.0, "success")
        mc.record_timeout("a")
        mc.reset()
        assert mc._agent_durations == {}
        assert mc._timeout_counts == {}
