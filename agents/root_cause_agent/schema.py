"""
File: schema.py
Purpose: Type-safe Pydantic v2 schemas for the root cause agent pipeline.
Dependencies: pydantic >=2.0
Performance: Schema validation <1ms per object

Defines input/output contracts for every layer — evidence synthesis,
confidence scoring, causal chain construction, verdict ranking,
timeline reconstruction, impact assessment, and validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


# ═══════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════


class Severity(str, Enum):
    """Severity classification for findings and verdicts."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentCategory(str, Enum):
    """Category of incident being investigated."""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    """Type of evidence in the trail."""
    DIRECT = "direct"
    CORRELATED = "correlated"
    CIRCUMSTANTIAL = "circumstantial"


class EvidenceSourceAgent(str, Enum):
    """Source agent that produced the evidence."""
    LOG_AGENT = "log_agent"
    METRICS_AGENT = "metrics_agent"
    DEPENDENCY_AGENT = "dependency_agent"
    HYPOTHESIS_AGENT = "hypothesis_agent"


class CausalRelationship(str, Enum):
    """Type of causal relationship in a chain."""
    CAUSES = "causes"
    CONTRIBUTES_TO = "contributes_to"
    CORRELATES_WITH = "correlates_with"


class ContradictionStrategy(str, Enum):
    """How a contradiction was resolved."""
    CONFIDENCE_WINS = "confidence_wins"
    TIMESTAMP_PRIORITY = "timestamp_priority"
    GRAPH_CENTRALITY = "graph_centrality"
    UNRESOLVED = "unresolved"


class ValidationSeverity(str, Enum):
    """Severity of a validation failure."""
    CRITICAL = "critical"
    WARNING = "warning"


# ═══════════════════════════════════════════════════════════════
#  INPUT SCHEMAS — upstream agent outputs (summarised)
# ═══════════════════════════════════════════════════════════════


class LogAgentFindings(BaseModel):
    """Summarised output from the log agent.

    Attributes:
        suspicious_services: Service names with elevated errors.
        error_patterns: Detected error patterns / keywords.
        log_anomalies: Anomalous log signals.
        confidence: Log agent confidence 0.0-1.0.
        timestamp: When the log analysis completed.
    """
    suspicious_services: List[str] = Field(default_factory=list)
    error_patterns: List[str] = Field(default_factory=list)
    log_anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: str = ""


class MetricsAgentFindings(BaseModel):
    """Summarised output from the metrics agent.

    Attributes:
        anomalies: Anomalous metrics detected.
        correlations: Metric correlations found.
        confidence: Metrics agent confidence 0.0-1.0.
        timestamp: When the metrics analysis completed.
    """
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    correlations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: str = ""


class DependencyAgentFindings(BaseModel):
    """Summarised output from the dependency agent.

    Attributes:
        impact_graph: Dependency graph edges.
        critical_paths: Critical path service lists.
        bottlenecks: Bottleneck services.
        blast_radius: Count of affected services.
        affected_services: Names of affected services.
        confidence: Dependency agent confidence 0.0-1.0.
        timestamp: When the dependency analysis completed.
    """
    impact_graph: Dict[str, List[str]] = Field(default_factory=dict)
    critical_paths: List[List[str]] = Field(default_factory=list)
    bottlenecks: List[str] = Field(default_factory=list)
    blast_radius: int = 0
    affected_services: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: str = ""


class HypothesisFindings(BaseModel):
    """Summarised output from the hypothesis agent.

    Attributes:
        ranked_hypotheses: Hypotheses sorted by confidence.
        top_hypothesis: Top-ranked theory string.
        top_confidence: Top hypothesis confidence.
        causal_chains: Causal chain descriptions.
        mttr_estimate: Estimated MTTR in minutes.
        category: Best-guess incident category.
        confidence: Hypothesis agent confidence 0.0-1.0.
        timestamp: When the hypothesis analysis completed.
    """
    ranked_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    top_hypothesis: str = ""
    top_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    causal_chains: List[Dict[str, Any]] = Field(default_factory=list)
    mttr_estimate: float = 30.0
    category: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: str = ""


class RootCauseAgentInput(BaseModel):
    """Input to the root cause agent from all 4 upstream agents.

    Attributes:
        log_findings: Log agent output summary.
        metrics_findings: Metrics agent output summary.
        dependency_findings: Dependency agent output summary.
        hypothesis_findings: Hypothesis agent output summary.
        incident_id: Current incident identifier.
        correlation_id: Request correlation ID.
        time_window: Analysis time window.
    """
    log_findings: LogAgentFindings = Field(
        default_factory=LogAgentFindings
    )
    metrics_findings: MetricsAgentFindings = Field(
        default_factory=MetricsAgentFindings
    )
    dependency_findings: DependencyAgentFindings = Field(
        default_factory=DependencyAgentFindings
    )
    hypothesis_findings: HypothesisFindings = Field(
        default_factory=HypothesisFindings
    )
    incident_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    correlation_id: str = ""
    time_window: str = "5m"


# ═══════════════════════════════════════════════════════════════
#  EVIDENCE SCHEMAS
# ═══════════════════════════════════════════════════════════════


class Evidence(BaseModel):
    """A single piece of evidence in the trail.

    Attributes:
        source: Which agent produced this.
        evidence_type: DIRECT, CORRELATED, or CIRCUMSTANTIAL.
        description: Human-readable description.
        confidence: Confidence in this evidence 0.0-1.0.
        timestamp: When the underlying signal was observed.
        raw_data: Original data from the source agent.
        score: Computed evidence score after weighting.
    """
    source: EvidenceSourceAgent
    evidence_type: EvidenceType = EvidenceType.CORRELATED
    description: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: str = ""
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(default=0.0, ge=0.0)


class SynthesisResult(BaseModel):
    """Result of fusing evidence from all 4 agents.

    Attributes:
        evidence_trail: All evidence items collected.
        sources_present: Which agents contributed.
        agreement_score: How much agents agree 0.0-1.0.
        primary_service: Most-blamed service.
        synthesis_latency_ms: Time to synthesise.
    """
    evidence_trail: List[Evidence] = Field(default_factory=list)
    sources_present: List[EvidenceSourceAgent] = Field(default_factory=list)
    agreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_service: str = ""
    synthesis_latency_ms: float = Field(default=0.0, ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  CAUSAL CHAIN SCHEMAS
# ═══════════════════════════════════════════════════════════════


class CausalLink(BaseModel):
    """A single link in the causal chain.

    Attributes:
        cause: The upstream event or service.
        effect: The downstream effect.
        relationship: Type of causal relationship.
        confidence: Confidence in this link 0.0-1.0.
        service: Service involved.
    """
    cause: str
    effect: str
    relationship: CausalRelationship = CausalRelationship.CAUSES
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    service: str = ""


# ═══════════════════════════════════════════════════════════════
#  TIMELINE SCHEMAS
# ═══════════════════════════════════════════════════════════════


class TimelineEvent(BaseModel):
    """A single event in the chronological timeline.

    Attributes:
        timestamp: ISO-8601 timestamp of the event.
        source: Which agent reported this.
        event: Description of what happened.
        service: Service involved.
        severity: Severity of the event.
    """
    timestamp: str
    source: EvidenceSourceAgent
    event: str = ""
    service: str = ""
    severity: Severity = Severity.MEDIUM


# ═══════════════════════════════════════════════════════════════
#  IMPACT SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ImpactAssessment(BaseModel):
    """Result of quantifying the blast radius and severity.

    Attributes:
        affected_services: List of affected service names.
        affected_count: Number of affected services.
        severity_score: Aggregate severity 0.0-1.0.
        blast_radius: Graph-traversal blast radius.
        is_cascading: Whether cascading failure detected.
    """
    affected_services: List[str] = Field(default_factory=list)
    affected_count: int = 0
    severity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    blast_radius: int = 0
    is_cascading: bool = False


# ═══════════════════════════════════════════════════════════════
#  CONTRADICTION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class Contradiction(BaseModel):
    """A detected contradiction between agents.

    Attributes:
        agent_a: First agent.
        agent_b: Second agent.
        claim_a: What agent A says.
        claim_b: What agent B says.
        resolved: Whether the contradiction was resolved.
        resolution_strategy: How it was resolved.
        winner: Which claim was accepted.
    """
    agent_a: EvidenceSourceAgent
    agent_b: EvidenceSourceAgent
    claim_a: str = ""
    claim_b: str = ""
    resolved: bool = False
    resolution_strategy: ContradictionStrategy = ContradictionStrategy.UNRESOLVED
    winner: str = ""


# ═══════════════════════════════════════════════════════════════
#  ALTERNATIVE VERDICT SCHEMAS
# ═══════════════════════════════════════════════════════════════


class AlternativeVerdict(BaseModel):
    """An alternative root cause candidate.

    Attributes:
        root_cause: Alternative theory.
        confidence: Confidence in this alternative 0.0-1.0.
        evidence_count: Number of supporting evidence items.
        category: Incident category.
    """
    root_cause: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = 0
    category: IncidentCategory = IncidentCategory.UNKNOWN


# ═══════════════════════════════════════════════════════════════
#  VERDICT METADATA
# ═══════════════════════════════════════════════════════════════


class VerdictMetadata(BaseModel):
    """Metadata for the root cause verdict.

    Attributes:
        correlation_id: Request correlation ID.
        analysis_start: When analysis began.
        analysis_end: When analysis completed.
        evidence_synthesis_ms: Time for evidence synthesis.
        confidence_calculation_ms: Time for confidence calculation.
        causal_chain_ms: Time for causal chain building.
        verdict_ranking_ms: Time for verdict ranking.
        timeline_reconstruction_ms: Time for timeline reconstruction.
        impact_assessment_ms: Time for impact assessment.
        contradiction_resolution_ms: Time for contradiction resolution.
        validation_ms: Time for validation.
        total_pipeline_ms: End-to-end latency.
        used_llm: Whether LLM was used.
        used_fallback: Whether fallback was used.
        cache_hit: Whether LLM cache was hit.
    """
    correlation_id: str = ""
    analysis_start: str = ""
    analysis_end: str = ""
    evidence_synthesis_ms: float = Field(default=0.0, ge=0.0)
    confidence_calculation_ms: float = Field(default=0.0, ge=0.0)
    causal_chain_ms: float = Field(default=0.0, ge=0.0)
    verdict_ranking_ms: float = Field(default=0.0, ge=0.0)
    timeline_reconstruction_ms: float = Field(default=0.0, ge=0.0)
    impact_assessment_ms: float = Field(default=0.0, ge=0.0)
    contradiction_resolution_ms: float = Field(default=0.0, ge=0.0)
    validation_ms: float = Field(default=0.0, ge=0.0)
    total_pipeline_ms: float = Field(default=0.0, ge=0.0)
    used_llm: bool = False
    used_fallback: bool = False
    cache_hit: bool = False


# ═══════════════════════════════════════════════════════════════
#  VALIDATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ValidatorError(BaseModel):
    """A single validation error or warning.

    Attributes:
        check_number: Validation check number (1-30).
        check_name: Symbolic check name.
        error_description: What failed.
        expected: Expected value or condition.
        actual: Actual value.
        severity: CRITICAL or WARNING.
    """
    check_number: int = Field(ge=1, le=30)
    check_name: str
    error_description: str
    expected: str = ""
    actual: str = ""
    severity: ValidationSeverity = ValidationSeverity.WARNING


class ValidationResult(BaseModel):
    """Result of all validation checks.

    Attributes:
        validation_passed: True if no CRITICAL errors.
        total_checks: Number of checks run.
        errors: CRITICAL failures.
        warnings: Non-critical warnings.
        validation_latency_ms: Time to validate.
    """
    validation_passed: bool = True
    total_checks: int = 0
    errors: List[ValidatorError] = Field(default_factory=list)
    warnings: List[ValidatorError] = Field(default_factory=list)
    validation_latency_ms: float = Field(default=0.0, ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA — RootCauseVerdict
# ═══════════════════════════════════════════════════════════════


class RootCauseVerdict(BaseModel):
    """Complete output from the root cause agent.

    Attributes:
        agent: Agent identifier (frozen to 'root_cause_agent').
        analysis_timestamp: When analysis was run.
        root_cause: Final verdict string.
        confidence: Bayesian confidence 0.0-1.0.
        evidence_trail: All supporting evidence from 4 agents.
        causal_chain: Root cause → intermediates → symptoms.
        affected_services: Services affected by the incident.
        timeline: Chronological incident reconstruction.
        impact: Quantified blast radius and severity.
        contradictions: Detected and resolved contradictions.
        alternative_causes: Ranked alternatives with confidence.
        reasoning: Human-readable explanation (≥50 chars).
        category: Best-guess incident category.
        severity: Assessed severity.
        estimated_mttr_minutes: Estimated mean-time-to-repair.
        correlation_id: Request correlation ID.
        classification_source: 'llm', 'fallback', 'cached', 'deterministic'.
        pipeline_latency_ms: End-to-end latency.
        metadata: Pipeline telemetry.
        validation: Validation check results.
    """
    agent: str = Field(default="root_cause_agent", frozen=True)
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    root_cause: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_trail: List[Evidence] = Field(default_factory=list)
    causal_chain: List[CausalLink] = Field(default_factory=list)
    affected_services: List[str] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    impact: Optional[ImpactAssessment] = None
    contradictions: List[Contradiction] = Field(default_factory=list)
    alternative_causes: List[AlternativeVerdict] = Field(default_factory=list)
    reasoning: str = ""
    category: IncidentCategory = IncidentCategory.UNKNOWN
    severity: Severity = Severity.MEDIUM
    estimated_mttr_minutes: float = Field(default=30.0, ge=0.0)
    correlation_id: str = ""
    classification_source: str = "deterministic"
    pipeline_latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: Optional[VerdictMetadata] = None
    validation: Optional[ValidationResult] = None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Clamp confidence to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
