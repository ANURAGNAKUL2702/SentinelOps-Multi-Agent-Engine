"""Tests for reporting.schema — Pydantic v2 models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from reporting.schema import (
    ActionItem,
    CostReport,
    DashboardData,
    ExecutiveSummary,
    HistoricalAnalytics,
    IncidentDetails,
    IncidentReport,
    IncidentStatus,
    IncidentTimeline,
    Insight,
    KPICard,
    PerformanceMetrics,
    Recommendation,
    RemediationPlan,
    ReportFormat,
    ReportMetadata,
    RootCauseAnalysis,
    SeverityLevel,
    TimelineEvent,
    TrendDirection,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_report_format_values(self) -> None:
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.PDF.value == "pdf"

    def test_severity_level_values(self) -> None:
        assert SeverityLevel.P0_CRITICAL.value == "P0_CRITICAL"
        assert SeverityLevel.P3_LOW.value == "P3_LOW"

    def test_incident_status_values(self) -> None:
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert len(IncidentStatus) == 5

    def test_trend_direction(self) -> None:
        assert TrendDirection.STABLE.value == "stable"

    def test_report_format_count(self) -> None:
        assert len(ReportFormat) == 4


# ---------------------------------------------------------------------------
# Sub-model tests
# ---------------------------------------------------------------------------

class TestExecutiveSummary:
    def test_defaults(self) -> None:
        es = ExecutiveSummary()
        assert es.summary == ""
        assert es.severity == SeverityLevel.P2_MEDIUM
        assert es.confidence == 0.0

    def test_frozen(self) -> None:
        es = ExecutiveSummary(summary="test")
        with pytest.raises(Exception):
            es.summary = "changed"  # type: ignore[misc]


class TestIncidentDetails:
    def test_defaults(self) -> None:
        d = IncidentDetails()
        assert d.incident_id == ""
        assert d.affected_services == []

    def test_with_values(self) -> None:
        d = IncidentDetails(
            incident_id="abc",
            duration_seconds=5.0,
            affected_services=["svc-a"],
        )
        assert d.duration_seconds == 5.0
        assert len(d.affected_services) == 1


class TestRootCauseAnalysis:
    def test_defaults(self) -> None:
        rca = RootCauseAnalysis()
        assert rca.root_cause == ""
        assert rca.evidence_trail == []

    def test_confidence_bounds(self) -> None:
        rca = RootCauseAnalysis(confidence=0.95)
        assert rca.confidence == 0.95


class TestActionItem:
    def test_defaults(self) -> None:
        ai = ActionItem()
        assert ai.priority == "P2"
        assert ai.owner == "SRE Team"


class TestRemediationPlan:
    def test_with_items(self) -> None:
        rp = RemediationPlan(
            runbook_title="Fix DB",
            runbook_steps=["Step 1", "Step 2"],
            action_items=[ActionItem(description="Restart DB")],
        )
        assert len(rp.action_items) == 1


class TestTimelineEvent:
    def test_defaults(self) -> None:
        te = TimelineEvent()
        assert te.source == ""
        assert te.severity == "info"


class TestCostReport:
    def test_defaults(self) -> None:
        cr = CostReport()
        assert cr.total_cost == 0.0

    def test_with_breakdown(self) -> None:
        cr = CostReport(
            total_cost=0.05,
            cost_by_agent={"log_agent": 0.01, "rca_agent": 0.04},
            total_tokens=1000,
        )
        assert sum(cr.cost_by_agent.values()) == 0.05


class TestPerformanceMetrics:
    def test_defaults(self) -> None:
        pm = PerformanceMetrics()
        assert pm.parallel_speedup == 1.0

    def test_with_latencies(self) -> None:
        pm = PerformanceMetrics(agent_latencies={"a": 1.0, "b": 2.0})
        assert len(pm.agent_latencies) == 2


class TestReportMetadata:
    def test_auto_uuid(self) -> None:
        m = ReportMetadata()
        uuid.UUID(m.report_id)  # Should not raise

    def test_auto_timestamp(self) -> None:
        m = ReportMetadata()
        assert m.generated_at is not None


class TestHistoricalAnalytics:
    def test_defaults(self) -> None:
        ha = HistoricalAnalytics()
        assert ha.total_incidents == 0
        assert ha.slo_compliance == 1.0


# ---------------------------------------------------------------------------
# Top-level model tests
# ---------------------------------------------------------------------------

class TestIncidentReport:
    def test_default_construction(self) -> None:
        r = IncidentReport()
        assert r.metadata is not None
        assert r.executive_summary.summary == ""

    def test_frozen(self) -> None:
        r = IncidentReport()
        with pytest.raises(Exception):
            r.visualizations = {}  # type: ignore[misc]

    def test_full_construction(self) -> None:
        r = IncidentReport(
            executive_summary=ExecutiveSummary(summary="outage"),
            cost_report=CostReport(total_cost=0.1),
            recommendations=[Recommendation(description="fix it")],
        )
        assert r.executive_summary.summary == "outage"
        assert len(r.recommendations) == 1


class TestDashboardData:
    def test_defaults(self) -> None:
        d = DashboardData()
        assert d.kpis == []

    def test_with_kpis(self) -> None:
        d = DashboardData(
            kpis=[KPICard(label="MTTR", value="5m")],
        )
        assert len(d.kpis) == 1


class TestInsight:
    def test_defaults(self) -> None:
        i = Insight()
        assert i.category == ""

    def test_with_data(self) -> None:
        i = Insight(
            category="cost",
            title="High cost",
            data={"amount": 0.1},
        )
        assert i.data["amount"] == 0.1
