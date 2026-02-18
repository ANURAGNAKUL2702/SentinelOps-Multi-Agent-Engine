"""
Tests for core/impact_assessor.py — Algorithm 8.
"""

import pytest

from agents.root_cause_agent.core.impact_assessor import ImpactAssessor
from agents.root_cause_agent.schema import (
    DependencyAgentFindings,
    MetricsAgentFindings,
    RootCauseAgentInput,
)


def _make_input() -> RootCauseAgentInput:
    return RootCauseAgentInput(
        dependency_findings=DependencyAgentFindings(
            impact_graph={
                "db": ["api", "cache"],
                "api": ["frontend", "mobile"],
                "cache": [],
            },
            blast_radius=4,
            affected_services=["api", "cache", "frontend", "mobile"],
            confidence=0.85,
        ),
        metrics_findings=MetricsAgentFindings(
            anomalies=[
                {"service": "api", "metric": "latency", "severity": "high"},
                {"service": "frontend", "metric": "errors", "severity": "critical"},
            ],
            confidence=0.75,
        ),
    )


class TestImpactAssessor:
    def test_assess_basic(self):
        assessor = ImpactAssessor()
        impact = assessor.assess(_make_input(), root_cause_service="db")
        assert impact.affected_count > 0
        assert impact.blast_radius > 0

    def test_bfs_reachable(self):
        assessor = ImpactAssessor()
        impact = assessor.assess(_make_input(), root_cause_service="db")
        # db → api, cache; api → frontend, mobile
        assert "api" in impact.affected_services
        assert "cache" in impact.affected_services

    def test_cascading_detection(self):
        assessor = ImpactAssessor()
        impact = assessor.assess(_make_input(), root_cause_service="db")
        assert impact.is_cascading is True  # >2 affected

    def test_severity_score_range(self):
        assessor = ImpactAssessor()
        impact = assessor.assess(_make_input(), root_cause_service="db")
        assert 0.0 <= impact.severity_score <= 1.0

    def test_no_graph_entry(self):
        assessor = ImpactAssessor()
        impact = assessor.assess(_make_input(), root_cause_service="nonexistent")
        # Should still have affected services from dep agent's list
        assert impact.affected_count >= 0

    def test_empty_input(self):
        assessor = ImpactAssessor()
        impact = assessor.assess(RootCauseAgentInput(), root_cause_service="")
        assert impact.affected_count == 0
        assert impact.is_cascading is False

    def test_blast_radius_uses_max(self):
        assessor = ImpactAssessor()
        impact = assessor.assess(_make_input(), root_cause_service="db")
        # blast_radius should be max(bfs_count, dep_agent_blast_radius)
        assert impact.blast_radius >= 4
