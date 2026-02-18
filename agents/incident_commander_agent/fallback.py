"""
File: fallback.py
Purpose: Deterministic fallback pipeline (<100ms).
Dependencies: Core algorithms only (no LLM).
Performance: <100ms end-to-end guaranteed.

Called when LLM is unavailable or circuit breaker is open.
Uses the 8 core deterministic algorithms to produce a
complete IncidentResponse without any LLM calls.
"""

from __future__ import annotations

import time
from typing import List

from agents.incident_commander_agent.config import IncidentCommanderConfig
from agents.incident_commander_agent.core.blast_radius_reporter import (
    calculate_blast_radius,
)
from agents.incident_commander_agent.core.communication_builder import (
    build_communications,
)
from agents.incident_commander_agent.core.escalation_calculator import (
    calculate_escalation,
)
from agents.incident_commander_agent.core.prevention_advisor import (
    generate_prevention_recommendations,
)
from agents.incident_commander_agent.core.priority_ranker import (
    rank_actions,
)
from agents.incident_commander_agent.core.action_sequencer import (
    sequence_actions,
)
from agents.incident_commander_agent.core.rollback_planner import (
    plan_rollback,
)
from agents.incident_commander_agent.core.runbook_generator import (
    generate_runbook,
)
from agents.incident_commander_agent.schema import (
    ActionItem,
    ActionPriority,
    BlastRadius,
    CommandMetadata,
    IncidentResponse,
    IncidentSeverity,
    RootCauseVerdict,
    ValidationReport,
)
from agents.incident_commander_agent.telemetry import TelemetryCollector


def _determine_severity(
    blast_radius: BlastRadius,
    confidence: float,
) -> IncidentSeverity:
    """Map blast radius + confidence to incident severity."""
    if blast_radius.estimated_users > 50_000 or blast_radius.availability_impact > 0.5:
        return IncidentSeverity.P0_CRITICAL
    if blast_radius.estimated_users > 10_000 or blast_radius.is_customer_facing:
        return IncidentSeverity.P1_HIGH
    if blast_radius.estimated_users > 1_000:
        return IncidentSeverity.P2_MEDIUM
    return IncidentSeverity.P3_LOW


def _build_action_items(
    root_cause: str,
    affected_services: List[str],
) -> List[ActionItem]:
    """Build default action items from root cause."""
    items = [
        ActionItem(
            action_id="act-001",
            priority=ActionPriority.P0,
            description=f"Investigate and mitigate: {root_cause}",
            owner="on-call SRE",
            estimated_minutes=15.0,
            category="investigation",
        ),
        ActionItem(
            action_id="act-002",
            priority=ActionPriority.P1,
            description="Verify service health post-remediation",
            owner="on-call SRE",
            dependencies=["act-001"],
            estimated_minutes=5.0,
            category="verification",
        ),
        ActionItem(
            action_id="act-003",
            priority=ActionPriority.P2,
            description="Update runbooks with lessons learned",
            owner="engineering lead",
            dependencies=["act-002"],
            estimated_minutes=30.0,
            category="documentation",
        ),
    ]

    if len(affected_services) > 2:
        items.append(
            ActionItem(
                action_id="act-004",
                priority=ActionPriority.P1,
                description="Check downstream service dependencies",
                owner="on-call SRE",
                dependencies=["act-001"],
                estimated_minutes=10.0,
                category="investigation",
            )
        )

    return items


class DeterministicFallback:
    """Deterministic fallback pipeline.

    Produces a complete IncidentResponse using only
    the 8 core algorithms — no LLM calls. Guaranteed
    to complete within 100ms.

    Args:
        config: Agent configuration.
        telemetry: Telemetry collector.
    """

    def __init__(
        self,
        config: IncidentCommanderConfig | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self._config = config or IncidentCommanderConfig()
        self._telemetry = telemetry or TelemetryCollector()

    def execute(
        self,
        verdict: RootCauseVerdict,
        validation_report: ValidationReport,
        correlation_id: str = "",
    ) -> IncidentResponse:
        """Execute the deterministic fallback pipeline.

        Args:
            verdict: Root cause verdict.
            validation_report: Validation report.
            correlation_id: Request correlation ID.

        Returns:
            Complete IncidentResponse.
        """
        pipeline_start = time.perf_counter()

        root_cause = verdict.root_cause or "unknown"
        affected = list(verdict.affected_services) if verdict.affected_services else []
        causal_chain = list(verdict.causal_chain) if verdict.causal_chain else []

        # Phase 1: Runbook
        t0 = time.perf_counter()
        runbook = generate_runbook(root_cause, affected, causal_chain)
        runbook_ms = (time.perf_counter() - t0) * 1000

        # Phase 2: Rollback
        t0 = time.perf_counter()
        rollback = plan_rollback(
            root_cause,
            timeline=list(verdict.timeline) if verdict.timeline else [],
            causal_chain=causal_chain,
        )
        rollback_ms = (time.perf_counter() - t0) * 1000

        # Phase 3: Blast radius
        t0 = time.perf_counter()
        total_known = len(self._config.known_services) or 10
        blast_radius = calculate_blast_radius(
            affected, causal_chain, total_known, self._config.blast_radius,
        )
        blast_ms = (time.perf_counter() - t0) * 1000

        # Phase 4: Severity
        severity = _determine_severity(blast_radius, verdict.confidence)

        # Phase 5: Action items
        t0 = time.perf_counter()
        actions = _build_action_items(root_cause, affected)
        ranking_ms_start = time.perf_counter()
        actions = rank_actions(actions, blast_radius, verdict.confidence)
        ranking_ms = (time.perf_counter() - ranking_ms_start) * 1000

        # Phase 6: Sequence
        t0_seq = time.perf_counter()
        actions = sequence_actions(actions)
        sequencing_ms = (time.perf_counter() - t0_seq) * 1000

        # Phase 7: Communications
        t0 = time.perf_counter()
        eta = runbook.estimated_total_minutes or 30.0
        comms = build_communications(root_cause, blast_radius, severity, eta)
        comms_ms = (time.perf_counter() - t0) * 1000

        # Phase 8: Prevention
        t0 = time.perf_counter()
        prevention = generate_prevention_recommendations(
            root_cause, causal_chain, validation_report,
        )
        prevention_ms = (time.perf_counter() - t0) * 1000

        # Phase 9: Escalation
        t0 = time.perf_counter()
        escalation = calculate_escalation(
            verdict, validation_report, blast_radius, self._config.escalation,
        )
        escalation_ms = (time.perf_counter() - t0) * 1000

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

        # Record telemetry
        self._telemetry.measure_value("runbook_generation", runbook_ms)
        self._telemetry.measure_value("rollback_planning", rollback_ms)
        self._telemetry.measure_value("blast_radius_calculation", blast_ms)
        self._telemetry.measure_value("priority_ranking", ranking_ms)
        self._telemetry.measure_value("action_sequencing", sequencing_ms)
        self._telemetry.measure_value("communication_building", comms_ms)
        self._telemetry.measure_value("prevention_advising", prevention_ms)
        self._telemetry.measure_value("escalation_calculation", escalation_ms)
        self._telemetry.measure_value("pipeline_total", pipeline_ms)
        self._telemetry.fallback_triggers.inc()

        if escalation.should_escalate:
            self._telemetry.escalations_triggered.inc()

        metadata = CommandMetadata(
            correlation_id=correlation_id,
            runbook_generation_ms=round(runbook_ms, 3),
            blast_radius_ms=round(blast_ms, 3),
            priority_ranking_ms=round(ranking_ms, 3),
            action_sequencing_ms=round(sequencing_ms, 3),
            rollback_planning_ms=round(rollback_ms, 3),
            communication_ms=round(comms_ms, 3),
            prevention_ms=round(prevention_ms, 3),
            escalation_ms=round(escalation_ms, 3),
            total_pipeline_ms=round(pipeline_ms, 3),
            used_llm=False,
            used_fallback=True,
            cache_hit=False,
            confidence_in_recommendations=escalation.auto_resolve_confidence,
        )

        return IncidentResponse(
            incident_id=correlation_id,
            root_cause_summary=root_cause,
            severity=severity,
            blast_radius=blast_radius,
            runbook=runbook,
            action_items=actions,
            rollback_plan=rollback,
            communication_plan=comms,
            prevention_recommendations=prevention,
            escalation_decision=escalation,
            metadata=metadata,
            correlation_id=correlation_id,
            classification_source="fallback",
            pipeline_latency_ms=round(pipeline_ms, 3),
        )
