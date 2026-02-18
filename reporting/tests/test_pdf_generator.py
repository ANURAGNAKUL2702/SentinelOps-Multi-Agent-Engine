"""Tests for reporting.generators.pdf_generator."""

from __future__ import annotations

import pytest

from reporting.schema import (
    CostReport,
    ExecutiveSummary,
    IncidentReport,
    ReportMetadata,
    RootCauseAnalysis,
)
from reporting.generators.pdf_generator import PDFGenerator


@pytest.fixture
def generator() -> PDFGenerator:
    return PDFGenerator()


@pytest.fixture
def sample_report() -> IncidentReport:
    return IncidentReport(
        metadata=ReportMetadata(report_id="pdf-test", correlation_id="corr-pdf"),
        executive_summary=ExecutiveSummary(summary="PDF test incident"),
        root_cause_analysis=RootCauseAnalysis(
            root_cause="Disk full",
            confidence=0.8,
        ),
        cost_report=CostReport(total_cost=0.03),
    )


class TestPDFGenerator:
    def test_available_flag(self, generator: PDFGenerator) -> None:
        # Should be a boolean regardless of whether reportlab is installed
        assert isinstance(generator.available, bool)

    def test_generate_returns_bytes_or_fallback(
        self, generator: PDFGenerator, sample_report: IncidentReport,
    ) -> None:
        if generator.available:
            result = generator.generate(sample_report)
            assert isinstance(result, bytes)
            assert result[:4] == b"%PDF"
        else:
            with pytest.raises(ImportError):
                generator.generate(sample_report)

    def test_save_creates_file_if_available(
        self,
        generator: PDFGenerator,
        sample_report: IncidentReport,
        tmp_path,
    ) -> None:
        if not generator.available:
            pytest.skip("reportlab not installed")
        path = tmp_path / "report.pdf"
        generator.save(sample_report, str(path))
        assert path.exists()
        assert path.stat().st_size > 0
