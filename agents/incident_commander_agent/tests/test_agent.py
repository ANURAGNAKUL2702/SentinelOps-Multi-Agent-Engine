"""Tests for agent.py — IncidentCommanderAgent end-to-end."""

from __future__ import annotations

from agents.incident_commander_agent.agent import IncidentCommanderAgent
from agents.incident_commander_agent.config import IncidentCommanderConfig
from agents.incident_commander_agent.tests.conftest import make_input


class TestIncidentCommanderAgent:
    def test_command_returns_response(self):
        agent = IncidentCommanderAgent()
        response = agent.command(make_input())

        assert response.agent == "incident_commander_agent"
        assert response.root_cause_summary != ""
        assert response.severity is not None
        assert len(response.runbook.steps) > 0
        assert len(response.action_items) > 0
        assert response.blast_radius.affected_service_count > 0

    def test_output_validation_runs(self):
        agent = IncidentCommanderAgent()
        response = agent.command(make_input())
        assert response.output_validation is not None
        assert response.output_validation.total_checks >= 28

    def test_correlation_id_propagated(self):
        agent = IncidentCommanderAgent()
        response = agent.command(make_input(correlation_id="corr-xyz"))
        assert response.correlation_id == "corr-xyz"
        assert response.incident_id == "corr-xyz"

    def test_classification_deterministic(self):
        agent = IncidentCommanderAgent()
        response = agent.command(make_input())
        assert response.classification_source == "deterministic"

    def test_escalation_for_low_confidence(self):
        agent = IncidentCommanderAgent()
        response = agent.command(make_input(confidence=0.2))
        assert response.escalation_decision.should_escalate is True
