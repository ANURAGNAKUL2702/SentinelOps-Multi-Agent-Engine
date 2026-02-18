"""Tests for reporting.report_builder."""

from __future__ import annotations

import json

import pytest

from reporting.config import ReportingConfig
from reporting.report_builder import ReportBuilder
from reporting.schema import IncidentReport, SeverityLevel


@pytest.fixture
def config(tmp_path) -> ReportingConfig:
    return ReportingConfig(
        output_dir=str(tmp_path / "output"),
        include_visualizations=False,
    )


@pytest.fixture
def builder(config: ReportingConfig) -> ReportBuilder:
    return ReportBuilder(config)


@pytest.fixture
def pipeline_result() -> dict:
    return {
        "correlation_id": "test-corr-id",
        "status": "success",
        "execution_time": 5.0,
        "agent_outputs": {
            "root_cause_output": {
                "root_cause": "Connection pool exhaustion",
                "confidence": 0.85,
                "severity": "high",
                "evidence": [
                    {"description": "Error logs show pool timeout"},
                ],
                "causal_chain": ["High traffic", "Pool exhaustion", "Timeouts"],
            },
            "validation_output": {
                "accuracy": 0.9,
                "hallucinations": [],
            },
            "incident_response": {
                "runbook": {
                    "title": "Fix connection pool",
                    "steps": ["Increase pool size", "Restart service"],
                },
                "action_items": [
                    {"priority": "P1", "description": "Increase pool"},
                ],
                "rollback_plan": "Revert config",
            },
            "dependency_output": {
                "affected_services": ["api-gateway", "auth-service"],
            },
            "log_output": {
                "timeline": [
                    {"source": "api", "event": "First error", "severity": "critical"},
                ],
            },
        },
        "telemetry": {
            "total_llm_cost": 0.05,
            "total_tokens": 5000,
            "total_llm_calls": 10,
            "agent_latencies": {"log_agent": 1.0, "rca_agent": 2.5},
            "parallel_speedup": 1.5,
            "timeout_violations": 0,
            "circuit_breaker_trips": 0,
        },
        "metadata": {
            "start_time": "2025-01-01T10:00:00Z",
            "end_time": "2025-01-01T10:00:05Z",
        },
    }


class TestBuildReport:
    def test_returns_incident_report(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        assert isinstance(report, IncidentReport)

    def test_correlation_id(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        assert report.metadata.correlation_id == "test-corr-id"

    def test_executive_summary(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        assert "Connection pool exhaustion" in report.executive_summary.summary
        assert report.executive_summary.severity == SeverityLevel.P1_HIGH

    def test_root_cause(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        assert report.root_cause_analysis.root_cause == "Connection pool exhaustion"
        assert report.root_cause_analysis.confidence == 0.85

    def test_remediation(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        assert len(report.remediation_plan.runbook_steps) == 2
        assert len(report.remediation_plan.action_items) == 1

    def test_cost_report(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        assert report.cost_report.total_cost == 0.05
        assert report.cost_report.total_tokens == 5000

    def test_performance_metrics(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        assert report.performance_metrics.total_pipeline_time == 5.0
        assert report.performance_metrics.parallel_speedup == 1.5


class TestRender:
    def test_render_json(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        output = builder.render(report, "json")
        data = json.loads(output)
        assert "metadata" in data

    def test_render_html(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        output = builder.render(report, "html")
        assert isinstance(output, str)
        assert len(output) > 100

    def test_render_markdown(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        output = builder.render(report, "markdown")
        assert isinstance(output, str)
        assert "#" in output


class TestSave:
    def test_save_json(
        self, builder: ReportBuilder, pipeline_result: dict, tmp_path,
    ) -> None:
        report = builder.build_report(pipeline_result)
        path = builder.save(report, "json", str(tmp_path))
        from pathlib import Path
        assert Path(path).exists()

    def test_save_html(
        self, builder: ReportBuilder, pipeline_result: dict, tmp_path,
    ) -> None:
        report = builder.build_report(pipeline_result)
        path = builder.save(report, "html", str(tmp_path))
        from pathlib import Path
        assert Path(path).exists()


class TestReportMetadata:
    def test_get_report_metadata(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        meta = builder.get_report_metadata(report.metadata.report_id)
        assert meta is not None
        assert meta["correlation_id"] == "test-corr-id"

    def test_get_nonexistent_metadata(
        self, builder: ReportBuilder,
    ) -> None:
        assert builder.get_report_metadata("nope") is None

    def test_get_report_file(
        self, builder: ReportBuilder, pipeline_result: dict, tmp_path,
    ) -> None:
        report = builder.build_report(pipeline_result)
        builder.save(report, "json", str(tmp_path))
        path = builder.get_report_file(report.metadata.report_id, "json")
        assert path is not None
