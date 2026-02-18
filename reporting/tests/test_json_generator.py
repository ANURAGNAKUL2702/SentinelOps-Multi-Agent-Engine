"""Tests for reporting.generators.json_generator."""

from __future__ import annotations

import json

import pytest

from reporting.schema import (
    CostReport,
    ExecutiveSummary,
    IncidentReport,
    ReportMetadata,
    RootCauseAnalysis,
)
from reporting.generators.json_generator import JSONGenerator


@pytest.fixture
def generator() -> JSONGenerator:
    return JSONGenerator()


@pytest.fixture
def sample_report() -> IncidentReport:
    return IncidentReport(
        metadata=ReportMetadata(report_id="json-test", correlation_id="corr-json"),
        executive_summary=ExecutiveSummary(summary="JSON test"),
        root_cause_analysis=RootCauseAnalysis(
            root_cause="Memory leak",
            confidence=0.75,
        ),
        cost_report=CostReport(total_cost=0.02),
    )


class TestJSONGenerator:
    def test_generate_valid_json(
        self, generator: JSONGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_fields(
        self, generator: JSONGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        data = json.loads(result)
        assert "metadata" in data
        assert "executive_summary" in data
        assert "root_cause_analysis" in data

    def test_root_cause_value(
        self, generator: JSONGenerator, sample_report: IncidentReport,
    ) -> None:
        data = json.loads(generator.generate(sample_report))
        assert data["root_cause_analysis"]["root_cause"] == "Memory leak"

    def test_save_creates_file(
        self,
        generator: JSONGenerator,
        sample_report: IncidentReport,
        tmp_path,
    ) -> None:
        path = tmp_path / "report.json"
        generator.save(sample_report, str(path))
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["metadata"]["report_id"] == "json-test"

    def test_datetime_serialization(
        self, generator: JSONGenerator, sample_report: IncidentReport,
    ) -> None:
        result = generator.generate(sample_report)
        # Must not crash — datetimes should be serialised
        data = json.loads(result)
        assert "generated_at" in data["metadata"]
