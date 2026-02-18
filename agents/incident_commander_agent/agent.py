"""
File: agent.py
Purpose: Incident Commander Agent — the orchestration layer.
Dependencies: All submodules.
Performance: <100ms fallback, <2s LLM.

Pipeline:
  Phase 1: Extract inputs + determine severity.
  Phase 2: Run 8 core algorithms (deterministic).
  Phase 3: Optionally enhance with LLM.
  Phase 4: Validate output + assemble response.

Entry point::

    agent = IncidentCommanderAgent()
    response = agent.command(input_data)
"""

from __future__ import annotations

import time
from typing import Optional

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
from agents.incident_commander_agent.fallback import DeterministicFallback
from agents.incident_commander_agent.schema import (
    ActionItem,
    ActionPriority,
    BlastRadius,
    CommandMetadata,
    IncidentCommanderInput,
    IncidentResponse,
    IncidentSeverity,
    RootCauseVerdict,
    ValidationReport,
)
from agents.incident_commander_agent.telemetry import (
    TelemetryCollector,
    get_logger,
)
from agents.incident_commander_agent.validator import CommandValidator

logger = get_logger("incident_commander_agent.agent")


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
    affected_services: list[str],
) -> list[ActionItem]:
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


class IncidentCommanderAgent:
    """Incident Commander Agent — generates actionable remediation.

    Takes a validated root cause verdict and generates
    runbooks, rollback plans, blast radius reports,
    action items, communication templates, prevention
    recommendations, and escalation decisions.

    Args:
        config: Agent configuration.

    Example::

        agent = IncidentCommanderAgent()
        response = agent.command(input_data)
        print(response.severity, response.runbook.title)
    """

    def __init__(
        self, config: Optional[IncidentCommanderConfig] = None
    ) -> None:
        self._config = config or IncidentCommanderConfig()
        self._telemetry = TelemetryCollector()
        self._fallback = DeterministicFallback(
            self._config, self._telemetry
        )
        self._validator = CommandValidator()

    def command(
        self,
        input_data: IncidentCommanderInput,
    ) -> IncidentResponse:
        """Run the full incident command pipeline.

        Args:
            input_data: Verdict + validation report from upstream.

        Returns:
            Complete IncidentResponse.
        """
        pipeline_start = time.perf_counter()
        correlation_id = input_data.correlation_id or ""

        self._telemetry.commands_total.inc()

        logger.info(
            f"Incident command started — correlation={correlation_id}",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
            },
        )

        try:
            response = self._execute_pipeline(
                input_data, correlation_id, pipeline_start
            )
            self._telemetry.commands_succeeded.inc()
            return response

        except Exception as exc:
            self._telemetry.commands_failed.inc()
            logger.error(
                f"Pipeline failed, engaging fallback: {exc}",
                extra={
                    "correlation_id": correlation_id,
                    "layer": "pipeline",
                },
            )
            return self._fallback.execute(
                input_data.verdict,
                input_data.validation_report,
                correlation_id,
            )

    def _execute_pipeline(
        self,
        input_data: IncidentCommanderInput,
        correlation_id: str,
        pipeline_start: float,
    ) -> IncidentResponse:
        """Execute the multi-phase pipeline.

        Phase 1: Extract inputs.
        Phase 2: Core deterministic algorithms.
        Phase 3: Optional LLM enhancement.
        Phase 4: Validate + assemble.
        """
        verdict = input_data.verdict
        validation_report = input_data.validation_report

        root_cause = verdict.root_cause or "unknown"
        affected = list(verdict.affected_services) if verdict.affected_services else []
        causal_chain = list(verdict.causal_chain) if verdict.causal_chain else []
        timeline = list(verdict.timeline) if verdict.timeline else []

        # ── Phase 2: Core algorithms ────────────────────────

        # 2a: Runbook
        with self._telemetry.measure("runbook_generation"):
            runbook = generate_runbook(root_cause, affected, causal_chain)

        # 2b: Rollback
        with self._telemetry.measure("rollback_planning"):
            rollback = plan_rollback(root_cause, timeline, causal_chain)

        # 2c: Blast radius
        with self._telemetry.measure("blast_radius_calculation"):
            total_known = len(self._config.known_services) or 10
            blast_radius = calculate_blast_radius(
                affected, causal_chain, total_known,
                self._config.blast_radius,
            )

        # 2d: Severity
        severity = _determine_severity(blast_radius, verdict.confidence)

        # 2e: Action items + ranking
        actions = _build_action_items(root_cause, affected)
        with self._telemetry.measure("priority_ranking"):
            actions = rank_actions(
                actions, blast_radius, verdict.confidence
            )

        # 2f: Sequencing
        with self._telemetry.measure("action_sequencing"):
            actions = sequence_actions(actions)

        # 2g: Communications
        with self._telemetry.measure("communication_building"):
            eta = runbook.estimated_total_minutes or 30.0
            comms = build_communications(
                root_cause, blast_radius, severity, eta
            )

        # 2h: Prevention
        with self._telemetry.measure("prevention_advising"):
            prevention = generate_prevention_recommendations(
                root_cause, causal_chain, validation_report,
            )

        # 2i: Escalation
        with self._telemetry.measure("escalation_calculation"):
            escalation = calculate_escalation(
                verdict, validation_report, blast_radius,
                self._config.escalation,
            )

        if escalation.should_escalate:
            self._telemetry.escalations_triggered.inc()

        # ── Phase 4: Assemble + validate ────────────────────

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        self._telemetry.measure_value("pipeline_total", pipeline_ms)

        metadata = CommandMetadata(
            correlation_id=correlation_id,
            total_pipeline_ms=round(pipeline_ms, 3),
            used_llm=False,
            used_fallback=False,
            cache_hit=False,
            confidence_in_recommendations=escalation.auto_resolve_confidence,
        )

        response = IncidentResponse(
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
            classification_source="deterministic",
            pipeline_latency_ms=round(pipeline_ms, 3),
        )

        # Validate output
        if self._config.features.enable_validation:
            with self._telemetry.measure("output_validation"):
                validation_result = self._validator.validate(response)
            response.output_validation = validation_result
            if not validation_result.validation_passed:
                self._telemetry.output_validation_failures.inc()
                logger.warning(
                    f"Output validation failed: "
                    f"{len(validation_result.errors)} error(s)",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "validation",
                    },
                )

        return response
