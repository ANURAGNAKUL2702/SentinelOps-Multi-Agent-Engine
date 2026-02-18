"""Tests for core/communication_builder.py — Algorithm 6."""

from __future__ import annotations

from agents.incident_commander_agent.core.communication_builder import (
    build_communications,
)
from agents.incident_commander_agent.schema import (
    BlastRadius,
    IncidentSeverity,
)


class TestCommunicationBuilder:
    def test_p0_has_war_room_channel(self):
        comms = build_communications(
            "db outage",
            severity=IncidentSeverity.P0_CRITICAL,
        )
        assert "#incident-war-room" in comms.notification_channels
        assert comms.update_frequency_minutes == 5

    def test_p3_low_frequency(self):
        comms = build_communications(
            "minor issue",
            severity=IncidentSeverity.P3_LOW,
        )
        assert comms.update_frequency_minutes == 60

    def test_status_update_contains_root_cause(self):
        comms = build_communications("memory_leak", severity=IncidentSeverity.P1_HIGH)
        assert "memory_leak" in comms.status_update

    def test_stakeholder_has_revenue(self):
        br = BlastRadius(
            affected_services=["svc-a"],
            revenue_impact_per_minute=100.0,
            estimated_users=5000,
        )
        comms = build_communications("db issue", blast_radius=br)
        assert "$" in comms.stakeholder_message or "100" in comms.stakeholder_message

    def test_external_comms_no_internal_details(self):
        comms = build_communications(
            "database_connection_pool_exhaustion",
            severity=IncidentSeverity.P0_CRITICAL,
        )
        assert "database_connection_pool" not in comms.external_comms
