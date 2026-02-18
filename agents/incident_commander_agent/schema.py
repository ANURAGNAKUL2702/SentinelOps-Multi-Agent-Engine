"""
File: schema.py
Purpose: Type-safe Pydantic v2 schemas for the Incident Commander Agent.
Dependencies: pydantic >=2.0
Performance: Schema validation <1ms per object.

Defines input/output contracts — incident response, runbook,
rollback plan, blast radius, action items, communication plan,
prevention recommendations, and escalation decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ═══════════════════════════════════════════════════════════════
#  RE-EXPORTS from upstream agents
# ═══════════════════════════════════════════════════════════════

from agents.root_cause_agent.schema import (  # noqa: E402
    AlternativeVerdict,
    CausalLink,
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    IncidentCategory,
    RootCauseVerdict,
    Severity,
    TimelineEvent,
)
from agents.validation_agent.schema import (  # noqa: E402
    Hallucination,
    HallucinationType,
    ValidationReport,
)


# ═══════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════


class IncidentSeverity(str, Enum):
    """Incident priority level."""
    P0_CRITICAL = "P0_CRITICAL"
    P1_HIGH = "P1_HIGH"
    P2_MEDIUM = "P2_MEDIUM"
    P3_LOW = "P3_LOW"


class ActionPriority(str, Enum):
    """Priority level for an action item."""
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class RollbackStrategy(str, Enum):
    """Rollback strategy type."""
    CONFIG_ROLLBACK = "config_rollback"
    DEPLOYMENT_ROLLBACK = "deployment_rollback"
    FEATURE_FLAG_TOGGLE = "feature_flag_toggle"
    TRAFFIC_REROUTE = "traffic_reroute"
    DATABASE_RESTORE = "database_restore"
    NO_ROLLBACK = "no_rollback"


class PreventionCategory(str, Enum):
    """Category of prevention recommendation."""
    MONITORING = "monitoring"
    ARCHITECTURE = "architecture"
    PROCESS = "process"
    CONFIGURATION = "configuration"


class CommandValidationSeverity(str, Enum):
    """Severity of a validation check failure."""
    CRITICAL = "critical"
    WARNING = "warning"


# ═══════════════════════════════════════════════════════════════
#  BLAST RADIUS
# ═══════════════════════════════════════════════════════════════


class BlastRadius(BaseModel):
    """Quantified impact of the incident.

    Attributes:
        affected_services: List of affected service names.
        affected_service_count: Number of affected services.
        estimated_users: Estimated number of impacted users.
        revenue_impact_per_minute: Estimated revenue loss per minute (USD).
        availability_impact: Fraction of system capacity affected 0.0-1.0.
        is_customer_facing: Whether customer-facing services are impacted.
    """
    affected_services: List[str] = Field(default_factory=list)
    affected_service_count: int = Field(default=0, ge=0)
    estimated_users: int = Field(default=0, ge=0)
    revenue_impact_per_minute: float = Field(default=0.0, ge=0.0)
    availability_impact: float = Field(default=0.0, ge=0.0, le=1.0)
    is_customer_facing: bool = False


# ═══════════════════════════════════════════════════════════════
#  RUNBOOK
# ═══════════════════════════════════════════════════════════════


class RemediationStep(BaseModel):
    """A single step in a remediation runbook.

    Attributes:
        step_number: Ordinal step number.
        description: Human-readable step description.
        command: Shell/kubectl command to execute.
        expected_outcome: What success looks like.
        validation_check: Command to verify step succeeded.
        estimated_minutes: Estimated time for this step.
        is_destructive: Whether this step is irreversible.
    """
    step_number: int = Field(ge=1)
    description: str = ""
    command: str = ""
    expected_outcome: str = ""
    validation_check: str = ""
    estimated_minutes: float = Field(default=1.0, ge=0.0)
    is_destructive: bool = False


class Runbook(BaseModel):
    """Step-by-step remediation procedure.

    Attributes:
        title: Runbook title.
        root_cause_category: Root cause this runbook addresses.
        steps: Ordered list of remediation steps.
        estimated_total_minutes: Total estimated resolution time.
        requires_approval: Whether changes need approval.
    """
    title: str = ""
    root_cause_category: str = ""
    steps: List[RemediationStep] = Field(default_factory=list)
    estimated_total_minutes: float = Field(default=0.0, ge=0.0)
    requires_approval: bool = False


# ═══════════════════════════════════════════════════════════════
#  ACTION ITEMS
# ═══════════════════════════════════════════════════════════════


class ActionItem(BaseModel):
    """A single prioritized action item.

    Attributes:
        action_id: Unique action identifier.
        priority: Priority level (P0-P3).
        description: What needs to be done.
        owner: Responsible team or role.
        dependencies: IDs of actions that must complete first.
        estimated_minutes: Estimated completion time.
        is_automated: Whether this can be automated.
        category: Action category.
    """
    action_id: str = ""
    priority: ActionPriority = ActionPriority.P2
    description: str = ""
    owner: str = ""
    dependencies: List[str] = Field(default_factory=list)
    estimated_minutes: float = Field(default=5.0, ge=0.0)
    is_automated: bool = False
    category: str = ""


# ═══════════════════════════════════════════════════════════════
#  ROLLBACK PLAN
# ═══════════════════════════════════════════════════════════════


class RollbackCheckpoint(BaseModel):
    """A checkpoint before/after a rollback step.

    Attributes:
        checkpoint_name: Name of the checkpoint.
        verification_command: Command to verify state.
        expected_result: What the verify command should return.
    """
    checkpoint_name: str = ""
    verification_command: str = ""
    expected_result: str = ""


class RollbackPlan(BaseModel):
    """Safe rollback strategy with checkpoints.

    Attributes:
        strategy: Rollback strategy type.
        is_safe: Whether rollback is considered safe.
        checkpoints: Verification checkpoints.
        rollback_steps: Steps to execute for rollback.
        risks: Identified risks of rollback.
        estimated_minutes: Estimated rollback time.
        requires_data_backup: Whether data backup is needed first.
    """
    strategy: RollbackStrategy = RollbackStrategy.NO_ROLLBACK
    is_safe: bool = True
    checkpoints: List[RollbackCheckpoint] = Field(default_factory=list)
    rollback_steps: List[RemediationStep] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    estimated_minutes: float = Field(default=0.0, ge=0.0)
    requires_data_backup: bool = False


# ═══════════════════════════════════════════════════════════════
#  COMMUNICATION PLAN
# ═══════════════════════════════════════════════════════════════


class CommunicationPlan(BaseModel):
    """Incident communication templates.

    Attributes:
        status_update: Internal status update message.
        stakeholder_message: Executive summary for stakeholders.
        external_comms: Customer-facing status page message.
        notification_channels: Channels to post updates to.
        update_frequency_minutes: How often to send updates.
    """
    status_update: str = ""
    stakeholder_message: str = ""
    external_comms: str = ""
    notification_channels: List[str] = Field(default_factory=list)
    update_frequency_minutes: int = Field(default=15, ge=1)


# ═══════════════════════════════════════════════════════════════
#  PREVENTION
# ═══════════════════════════════════════════════════════════════


class Prevention(BaseModel):
    """A prevention recommendation.

    Attributes:
        category: monitoring, architecture, process, or configuration.
        title: Short recommendation title.
        description: Detailed recommendation.
        effort_estimate: Estimated implementation effort.
        priority: Priority for implementing this fix.
    """
    category: PreventionCategory = PreventionCategory.MONITORING
    title: str = ""
    description: str = ""
    effort_estimate: str = ""
    priority: ActionPriority = ActionPriority.P2


# ═══════════════════════════════════════════════════════════════
#  ESCALATION DECISION
# ═══════════════════════════════════════════════════════════════


class EscalationDecision(BaseModel):
    """Whether to escalate to a human SRE.

    Attributes:
        should_escalate: Whether human escalation is needed.
        reason: Explanation for the decision.
        suggested_escalation_path: Who to escalate to.
        auto_resolve_confidence: Confidence in auto-resolution.
    """
    should_escalate: bool = False
    reason: str = ""
    suggested_escalation_path: str = ""
    auto_resolve_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════
#  COMMAND METADATA
# ═══════════════════════════════════════════════════════════════


class CommandMetadata(BaseModel):
    """Metadata for the incident command response.

    Attributes:
        correlation_id: Request correlation ID.
        command_start: When command generation began.
        command_end: When command generation completed.
        runbook_generation_ms: Time for runbook generation.
        blast_radius_ms: Time for blast radius calculation.
        priority_ranking_ms: Time for priority ranking.
        action_sequencing_ms: Time for action sequencing.
        rollback_planning_ms: Time for rollback planning.
        communication_ms: Time for communication building.
        prevention_ms: Time for prevention advising.
        escalation_ms: Time for escalation calculation.
        total_pipeline_ms: End-to-end latency.
        used_llm: Whether LLM was used.
        used_fallback: Whether fallback was used.
        cache_hit: Whether LLM cache was hit.
        confidence_in_recommendations: Overall confidence.
    """
    correlation_id: str = ""
    command_start: str = ""
    command_end: str = ""
    runbook_generation_ms: float = Field(default=0.0, ge=0.0)
    blast_radius_ms: float = Field(default=0.0, ge=0.0)
    priority_ranking_ms: float = Field(default=0.0, ge=0.0)
    action_sequencing_ms: float = Field(default=0.0, ge=0.0)
    rollback_planning_ms: float = Field(default=0.0, ge=0.0)
    communication_ms: float = Field(default=0.0, ge=0.0)
    prevention_ms: float = Field(default=0.0, ge=0.0)
    escalation_ms: float = Field(default=0.0, ge=0.0)
    total_pipeline_ms: float = Field(default=0.0, ge=0.0)
    used_llm: bool = False
    used_fallback: bool = False
    cache_hit: bool = False
    confidence_in_recommendations: float = Field(
        default=0.0, ge=0.0, le=1.0
    )


# ═══════════════════════════════════════════════════════════════
#  VALIDATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class CommandValidatorError(BaseModel):
    """A single validation error or warning.

    Attributes:
        check_number: Validation check number (1-30).
        check_name: Symbolic check name.
        error_description: What failed.
        expected: Expected value or condition.
        actual: Actual value.
        severity: CRITICAL or WARNING.
    """
    check_number: int = Field(ge=1, le=35)
    check_name: str
    error_description: str
    expected: str = ""
    actual: str = ""
    severity: CommandValidationSeverity = CommandValidationSeverity.WARNING


class CommandValidatorResult(BaseModel):
    """Result of all output validation checks.

    Attributes:
        validation_passed: True if no CRITICAL errors.
        total_checks: Number of checks run.
        errors: CRITICAL failures.
        warnings: Non-critical warnings.
        validation_latency_ms: Time to validate.
    """
    validation_passed: bool = True
    total_checks: int = 0
    errors: List[CommandValidatorError] = Field(default_factory=list)
    warnings: List[CommandValidatorError] = Field(default_factory=list)
    validation_latency_ms: float = Field(default=0.0, ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  INCIDENT COMMANDER INPUT
# ═══════════════════════════════════════════════════════════════


class IncidentCommanderInput(BaseModel):
    """Input to the incident commander agent.

    Attributes:
        verdict: The root cause verdict.
        validation_report: Validation agent's report.
        correlation_id: Request correlation ID.
    """
    verdict: RootCauseVerdict
    validation_report: ValidationReport
    correlation_id: str = ""


# ═══════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA — IncidentResponse
# ═══════════════════════════════════════════════════════════════


class IncidentResponse(BaseModel):
    """Complete output from the Incident Commander Agent.

    Attributes:
        agent: Agent identifier.
        analysis_timestamp: When response was generated.
        incident_id: Incident/correlation ID.
        root_cause_summary: One-line root cause summary.
        severity: Incident severity (P0-P3).
        blast_radius: Quantified impact.
        runbook: Step-by-step remediation procedure.
        action_items: Prioritized action items.
        rollback_plan: Rollback strategy and steps.
        communication_plan: Status updates and messages.
        prevention_recommendations: Long-term prevention measures.
        escalation_decision: Whether to escalate.
        metadata: Pipeline telemetry.
        correlation_id: Request correlation ID.
        classification_source: 'llm', 'fallback', 'deterministic'.
        pipeline_latency_ms: End-to-end latency.
        output_validation: Output validation results.
    """
    agent: str = Field(default="incident_commander_agent", frozen=True)
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    incident_id: str = ""
    root_cause_summary: str = ""
    severity: IncidentSeverity = IncidentSeverity.P2_MEDIUM
    blast_radius: BlastRadius = Field(default_factory=BlastRadius)
    runbook: Runbook = Field(default_factory=Runbook)
    action_items: List[ActionItem] = Field(default_factory=list)
    rollback_plan: RollbackPlan = Field(default_factory=RollbackPlan)
    communication_plan: CommunicationPlan = Field(
        default_factory=CommunicationPlan
    )
    prevention_recommendations: List[Prevention] = Field(
        default_factory=list
    )
    escalation_decision: EscalationDecision = Field(
        default_factory=EscalationDecision
    )
    metadata: Optional[CommandMetadata] = None
    correlation_id: str = ""
    classification_source: str = "deterministic"
    pipeline_latency_ms: float = Field(default=0.0, ge=0.0)
    output_validation: Optional[CommandValidatorResult] = None

    @field_validator(
        "pipeline_latency_ms",
    )
    @classmethod
    def clamp_latency(cls, v: float) -> float:
        """Ensure latency is non-negative."""
        return max(0.0, v)
