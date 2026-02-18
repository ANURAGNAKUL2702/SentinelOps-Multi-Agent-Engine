"""Tests for reporting.performance — latency quality gates."""

from __future__ import annotations

import time

import pytest

from reporting.config import ReportingConfig
from reporting.report_builder import ReportBuilder
from reporting.schema import IncidentReport


@pytest.fixture
def config(tmp_path) -> ReportingConfig:
    return ReportingConfig(
        output_dir=str(tmp_path),
        include_visualizations=False,
    )


@pytest.fixture
def builder(config: ReportingConfig) -> ReportBuilder:
    return ReportBuilder(config)


@pytest.fixture
def pipeline_result() -> dict:
    return {
        "correlation_id": "perf-test",
        "status": "success",
        "execution_time": 3.0,
        "agent_outputs": {
            "root_cause_output": {"root_cause": "test", "confidence": 0.5},
        },
        "telemetry": {"total_llm_cost": 0.01, "total_tokens": 100},
    }


class TestPerformanceGates:
    def test_build_report_under_500ms(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        start = time.perf_counter()
        builder.build_report(pipeline_result)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"build_report took {elapsed:.3f}s"

    def test_render_html_under_2s(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        start = time.perf_counter()
        builder.render(report, "html")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"HTML render took {elapsed:.3f}s"

    def test_render_json_under_100ms(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        start = time.perf_counter()
        builder.render(report, "json")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"JSON render took {elapsed:.3f}s"

    def test_render_markdown_under_2s(
        self, builder: ReportBuilder, pipeline_result: dict,
    ) -> None:
        report = builder.build_report(pipeline_result)
        start = time.perf_counter()
        builder.render(report, "markdown")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Markdown render took {elapsed:.3f}s"
