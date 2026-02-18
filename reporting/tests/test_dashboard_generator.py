"""Tests for reporting.generators.dashboard_generator."""

from __future__ import annotations

import pytest

from reporting.schema import DashboardData, KPICard, TrendDirection
from reporting.generators.dashboard_generator import DashboardGenerator


@pytest.fixture
def generator() -> DashboardGenerator:
    return DashboardGenerator()


@pytest.fixture
def sample_dashboard() -> DashboardData:
    return DashboardData(
        kpis=[
            KPICard(label="MTTR", value="5m", trend=TrendDirection.DECREASING),
            KPICard(label="Incidents", value="42", trend=TrendDirection.STABLE),
        ],
        recent_incidents=[
            {"id": "inc-1", "root_cause": "DB pool", "severity": "P1_HIGH"},
        ],
    )


class TestDashboardGenerator:
    def test_generate_returns_html(
        self, generator: DashboardGenerator, sample_dashboard: DashboardData,
    ) -> None:
        result = generator.generate(sample_dashboard)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_kpi_labels(
        self, generator: DashboardGenerator, sample_dashboard: DashboardData,
    ) -> None:
        result = generator.generate(sample_dashboard)
        assert "MTTR" in result
        assert "42" in result

    def test_empty_dashboard(
        self, generator: DashboardGenerator,
    ) -> None:
        d = DashboardData()
        result = generator.generate(d)
        assert isinstance(result, str)
