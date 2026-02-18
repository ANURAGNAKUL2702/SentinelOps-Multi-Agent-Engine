"""
File: core/communication_builder.py
Purpose: Algorithm 6 – Build incident communication templates.
Dependencies: Standard library only + schema.
Performance: <1ms per call, pure function.
"""

from __future__ import annotations

from typing import Dict, List

from agents.incident_commander_agent.schema import (
    BlastRadius,
    CommunicationPlan,
    IncidentSeverity,
)


_SEVERITY_LABEL: Dict[IncidentSeverity, str] = {
    IncidentSeverity.P0_CRITICAL: "Critical",
    IncidentSeverity.P1_HIGH: "High",
    IncidentSeverity.P2_MEDIUM: "Medium",
    IncidentSeverity.P3_LOW: "Low",
}

_CHANNELS: Dict[IncidentSeverity, List[str]] = {
    IncidentSeverity.P0_CRITICAL: [
        "#incident-war-room", "#engineering-all", "#exec-alerts",
    ],
    IncidentSeverity.P1_HIGH: ["#incident-war-room", "#engineering-on-call"],
    IncidentSeverity.P2_MEDIUM: ["#engineering-on-call"],
    IncidentSeverity.P3_LOW: ["#engineering-on-call"],
}

_FREQ: Dict[IncidentSeverity, int] = {
    IncidentSeverity.P0_CRITICAL: 5,
    IncidentSeverity.P1_HIGH: 15,
    IncidentSeverity.P2_MEDIUM: 30,
    IncidentSeverity.P3_LOW: 60,
}


def _status_update(
    root_cause: str,
    severity: IncidentSeverity,
    blast_radius: BlastRadius,
    eta_minutes: float,
) -> str:
    """Build internal status update."""
    label = _SEVERITY_LABEL.get(severity, "Unknown")
    services = ", ".join(blast_radius.affected_services[:5]) or "unknown"
    return (
        f"[{label} Incident] Root cause: {root_cause}. "
        f"Affected services: {services}. "
        f"Estimated users impacted: {blast_radius.estimated_users:,}. "
        f"ETA to resolution: {eta_minutes:.0f} min."
    )


def _stakeholder_message(
    root_cause: str,
    severity: IncidentSeverity,
    blast_radius: BlastRadius,
    eta_minutes: float,
) -> str:
    """Build executive summary."""
    label = _SEVERITY_LABEL.get(severity, "Unknown")
    revenue = blast_radius.revenue_impact_per_minute
    users = blast_radius.estimated_users
    return (
        f"Severity: {label}\n"
        f"Root cause: {root_cause}\n"
        f"Impact: ~{users:,} users affected, "
        f"~${revenue:,.2f}/min revenue exposure.\n"
        f"ETA: {eta_minutes:.0f} minutes to resolution.\n"
        f"Status: Remediation in progress."
    )


def _external_comms(
    severity: IncidentSeverity,
    blast_radius: BlastRadius,
) -> str:
    """Build customer-facing status page message."""
    if severity in (IncidentSeverity.P0_CRITICAL, IncidentSeverity.P1_HIGH):
        return (
            "We are currently experiencing degraded performance "
            "affecting some of our services. Our engineering team "
            "is actively investigating and working to resolve the issue. "
            "We will provide updates as more information becomes available."
        )
    return (
        "A minor issue has been identified and is being addressed. "
        "No significant impact to services is expected."
    )


def build_communications(
    root_cause: str,
    blast_radius: BlastRadius | None = None,
    severity: IncidentSeverity = IncidentSeverity.P2_MEDIUM,
    eta_minutes: float = 30.0,
) -> CommunicationPlan:
    """Build communication templates for the incident.

    Args:
        root_cause: Root cause description.
        blast_radius: Quantified impact (defaults to empty).
        severity: Incident severity.
        eta_minutes: Estimated time to resolution.

    Returns:
        CommunicationPlan with status update, stakeholder message,
        external comms, channels, and update frequency.
    """
    br = blast_radius or BlastRadius()

    return CommunicationPlan(
        status_update=_status_update(root_cause, severity, br, eta_minutes),
        stakeholder_message=_stakeholder_message(
            root_cause, severity, br, eta_minutes
        ),
        external_comms=_external_comms(severity, br),
        notification_channels=_CHANNELS.get(severity, ["#engineering-on-call"]),
        update_frequency_minutes=_FREQ.get(severity, 30),
    )
