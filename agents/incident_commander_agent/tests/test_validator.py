"""Tests for validator.py — 28+ validation checks."""

from __future__ import annotations

import pytest

from agents.incident_commander_agent.schema import (
    ActionItem,
    ActionPriority,
    BlastRadius,
    CommandMetadata,
    CommunicationPlan,
    EscalationDecision,
    IncidentResponse,
    IncidentSeverity,
    Prevention,
    PreventionCategory,
    RollbackPlan,
    RollbackStrategy,
    Runbook,
    RemediationStep,
)
from agents.incident_commander_agent.validator import CommandValidator


def _valid_response() -> IncidentResponse:
    """Build a minimally valid IncidentResponse."""
    return IncidentResponse(
        incident_id="inc-001",
        root_cause_summary="database_connection_pool_exhaustion",
        severity=IncidentSeverity.P1_HIGH,
        blast_radius=BlastRadius(
            affected_services=["svc-a", "svc-b"],
            affected_service_count=2,
            estimated_users=10_000,
            availability_impact=0.2,
        ),
        runbook=Runbook(
            title="DB Pool Remediation",
            steps=[
                RemediationStep(step_number=1, description="Check", command="kubectl get pods"),
                RemediationStep(step_number=2, description="Fix", command="kubectl rollout restart"),
            ],
        ),
        action_items=[
            ActionItem(action_id="a1", priority=ActionPriority.P0, description="Investigate"),
            ActionItem(action_id="a2", priority=ActionPriority.P1, dependencies=["a1"]),
        ],
        rollback_plan=RollbackPlan(
            strategy=RollbackStrategy.DEPLOYMENT_ROLLBACK,
            is_safe=True,
        ),
        communication_plan=CommunicationPlan(
            status_update="Incident in progress",
            notification_channels=["#ops"],
        ),
        prevention_recommendations=[
            Prevention(
                category=PreventionCategory.MONITORING,
                title="Add alerts",
            ),
        ],
        escalation_decision=EscalationDecision(
            should_escalate=False,
            reason="All good",
            auto_resolve_confidence=0.85,
        ),
        metadata=CommandMetadata(total_pipeline_ms=50.0),
        correlation_id="corr-001",
        classification_source="deterministic",
        pipeline_latency_ms=50.0,
    )


class TestCommandValidator:
    def test_valid_response_passes(self):
        validator = CommandValidator()
        result = validator.validate(_valid_response())
        assert result.validation_passed is True
        assert result.total_checks == 28
        assert len(result.errors) == 0

    def test_empty_root_cause_fails_critical(self):
        resp = _valid_response()
        resp.root_cause_summary = ""
        result = CommandValidator().validate(resp)
        assert result.validation_passed is False
        crit_names = [e.check_name for e in result.errors]
        assert "root_cause_summary" in crit_names

    def test_blast_radius_count_mismatch_fails(self):
        resp = _valid_response()
        resp.blast_radius.affected_service_count = 999
        result = CommandValidator().validate(resp)
        assert result.validation_passed is False

    def test_no_runbook_steps_fails(self):
        resp = _valid_response()
        resp.runbook.steps = []
        result = CommandValidator().validate(resp)
        assert result.validation_passed is False

    def test_duplicate_action_ids_fails(self):
        resp = _valid_response()
        resp.action_items = [
            ActionItem(action_id="dup"),
            ActionItem(action_id="dup"),
        ]
        result = CommandValidator().validate(resp)
        assert result.validation_passed is False

    def test_self_dependency_fails(self):
        resp = _valid_response()
        resp.action_items = [
            ActionItem(action_id="a1", dependencies=["a1"]),
        ]
        result = CommandValidator().validate(resp)
        assert result.validation_passed is False

    def test_invalid_classification_source_fails(self):
        resp = _valid_response()
        resp.classification_source = "magic"
        result = CommandValidator().validate(resp)
        assert result.validation_passed is False

    def test_validation_latency_recorded(self):
        result = CommandValidator().validate(_valid_response())
        assert result.validation_latency_ms >= 0
