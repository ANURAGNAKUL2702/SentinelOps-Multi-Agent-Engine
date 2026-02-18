"""Tests for reporting.generators.html_generator."""

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
from reporting.generators.html_generator import HTMLGenerator


@pytest.fixture
def generator() -> HTMLGenerator:
    return HTMLGenerator()


@pytest.fixture
def sample_report() -> IncidentReport:
    return IncidentReport(
        metadata=ReportMetadata(report_id="html-test", correlation_id="corr-html"),
        executive_summary=ExecutiveSummary(
            summary="HTML test incident",
            severity=SeverityLevel.P0_CRITICAL,
            confidence=0.92,
        ),
        root_cause_analysis=RootCauseAnalysis(
            root_cause="Database deadlock",
            confidence=0.92,
        ),
        cost_report=CostReport(total_cost=0.1, total_tokens=10000),
    )


class TestHTMLGenerator:
    def test_generate_returns_html(
        self, generator: HTMLGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        assert isinstance(result, str)
        assert "<html" in result.lower() or "<div" in result.lower() or "<!doctype" in result.lower()

    def test_contains_root_cause(
        self, generator: HTMLGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        assert "Database deadlock" in result

    def test_html_escaping(
        self, generator: HTMLGenerator,
    ) -> None:
        """Ensure XSS payloads are escaped."""
        report = IncidentReport(
            executive_summary=ExecutiveSummary(
                summary="<script>alert(1)</script>",
            ),
        )
        result = generator.generate(report)
        # The XSS payload must be escaped; the literal injection must not appear
        assert "<script>alert(1)</script>" not in result
        # The escaped version should be present
        assert "&lt;script&gt;" in result or "alert(1)" in result

    def test_save_creates_file(
        self,
        generator: HTMLGenerator,
        sample_report: IncidentReport,
        tmp_path,
    ) -> None:
        path = tmp_path / "report.html"
        generator.save(sample_report, str(path))
        assert path.exists()
        assert path.stat().st_size > 0
