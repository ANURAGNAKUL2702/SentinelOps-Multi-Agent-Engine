"""Tests for reporting.generators.markdown_generator."""

from __future__ import annotations

import pytest

from reporting.schema import (
    CostReport,
    ExecutiveSummary,
    IncidentReport,
    ReportMetadata,
    RootCauseAnalysis,
    SeverityLevel,
)
from reporting.generators.markdown_generator import MarkdownGenerator


@pytest.fixture
def generator() -> MarkdownGenerator:
    return MarkdownGenerator()


@pytest.fixture
def sample_report() -> IncidentReport:
    return IncidentReport(
        metadata=ReportMetadata(report_id="test-123", correlation_id="corr-456"),
        executive_summary=ExecutiveSummary(
            summary="Test incident resolved",
            severity=SeverityLevel.P1_HIGH,
            confidence=0.85,
        ),
        root_cause_analysis=RootCauseAnalysis(
            root_cause="Connection pool exhaustion",
            confidence=0.85,
        ),
        cost_report=CostReport(total_cost=0.05, total_tokens=5000),
    )


class TestMarkdownGenerator:
    def test_generate_returns_string(
        self, generator: MarkdownGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_report_id(
        self, generator: MarkdownGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        assert "corr-456" in result or "test-123" in result

    def test_contains_root_cause(
        self, generator: MarkdownGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        assert "Connection pool exhaustion" in result

    def test_contains_markdown_headers(
        self, generator: MarkdownGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        assert "#" in result

    def test_save_creates_file(
        self,
        generator: MarkdownGenerator,
        sample_report: IncidentReport,
        tmp_path,
    ) -> None:
        path = tmp_path / "report.md"
        generator.save(sample_report, str(path))
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0
