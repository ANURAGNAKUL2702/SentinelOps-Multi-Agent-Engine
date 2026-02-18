"""Tests for schema.py — Pydantic model validation."""

from __future__ import annotations

import pytest

from agents.incident_commander_agent.schema import (
    ActionItem,
    ActionPriority,
    BlastRadius,
    CommandMetadata,
    CommandValidatorResult,
    CommunicationPlan,
    EscalationDecision,
    IncidentCommanderInput,
    IncidentResponse,
    IncidentSeverity,
    Prevention,
    PreventionCategory,
    RemediationStep,
    RollbackCheckpoint,
    RollbackPlan,
    RollbackStrategy,
    Runbook,
)
from agents.incident_commander_agent.tests.conftest import (
    make_verdict,
    make_validation_report,
)


class TestBlastRadius:
    def test_defaults(self):
        br = BlastRadius()
        assert br.affected_services == []
        assert br.affected_service_count == 0
        assert br.estimated_users == 0
        assert br.availability_impact == 0.0
        assert br.is_customer_facing is False

    def test_custom_values(self):
        br = BlastRadius(
            affected_services=["svc-a", "svc-b"],
            affected_service_count=2,
            estimated_users=10_000,
            revenue_impact_per_minute=500.0,
            availability_impact=0.3,
            is_customer_facing=True,
        )
        assert br.affected_service_count == 2
        assert br.is_customer_facing is True


class TestRunbook:
    def test_empty_runbook(self):
        rb = Runbook()
        assert rb.steps == []
        assert rb.title == ""

    def test_with_steps(self):
        step = RemediationStep(step_number=1, description="Check logs")
        rb = Runbook(title="Test", steps=[step])
        assert len(rb.steps) == 1


class TestActionItem:
    def test_defaults(self):
        ai = ActionItem()
        assert ai.priority == ActionPriority.P2
        assert ai.dependencies == []

    def test_with_priority(self):
        ai = ActionItem(priority=ActionPriority.P0, action_id="a1")
        assert ai.priority == ActionPriority.P0


class TestRollbackPlan:
    def test_defaults(self):
        rp = RollbackPlan()
        assert rp.strategy == RollbackStrategy.NO_ROLLBACK
        assert rp.is_safe is True

    def test_with_strategy(self):
        rp = RollbackPlan(strategy=RollbackStrategy.DEPLOYMENT_ROLLBACK)
        assert rp.strategy == RollbackStrategy.DEPLOYMENT_ROLLBACK


class TestIncidentResponse:
    def test_defaults(self):
        ir = IncidentResponse()
        assert ir.agent == "incident_commander_agent"
        assert ir.classification_source == "deterministic"
        assert ir.pipeline_latency_ms == 0.0

    def test_severity_enum(self):
        ir = IncidentResponse(severity=IncidentSeverity.P0_CRITICAL)
        assert ir.severity == IncidentSeverity.P0_CRITICAL

    def test_latency_clamped(self):
        ir = IncidentResponse(pipeline_latency_ms=0.0)


class TestIncidentCommanderInput:
    def test_creation(self):
        inp = IncidentCommanderInput(
            verdict=make_verdict(),
            validation_report=make_validation_report(),
            correlation_id="test-123",
        )
        assert inp.correlation_id == "test-123"
        assert inp.verdict.root_cause == "database_connection_pool_exhaustion"
