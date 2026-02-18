"""
File: schema.py
Purpose: Type-safe Pydantic v2 schemas for the hypothesis agent pipeline.
Dependencies: pydantic >=2.0
Performance: Schema validation <1ms per object

Defines input/output contracts for every layer — evidence aggregation,
pattern matching, hypothesis generation, causal reasoning, ranking,
and validation.
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
    """Severity classification for hypotheses and evidence."""
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


class PatternName(str, Enum):
    """Known failure pattern names from the pattern library."""
    DATABASE_CONNECTION_POOL_EXHAUSTION = "database_connection_pool_exhaustion"
    MEMORY_LEAK = "memory_leak"
    NETWORK_PARTITION = "network_partition"
    CPU_SPIKE = "cpu_spike"
    DEPLOYMENT_ISSUE = "deployment_issue"
    CONFIGURATION_ERROR = "configuration_error"


class EvidenceSource(str, Enum):
    """Source agent that produced the evidence."""
    LOG_AGENT = "log_agent"
    METRICS_AGENT = "metrics_agent"
    DEPENDENCY_AGENT = "dependency_agent"


class EvidenceStrength(str, Enum):
    """Strength classification of a piece of evidence."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class CausalRelationship(str, Enum):
    """Type of causal relationship in a chain."""
    CAUSES = "causes"
    CONTRIBUTES_TO = "contributes_to"
    CORRELATES_WITH = "correlates_with"


class HypothesisStatus(str, Enum):
    """Status of a hypothesis after ranking/pruning."""
    ACTIVE = "active"
    PRUNED = "pruned"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ValidationSeverity(str, Enum):
    """Severity of a validation failure."""
    CRITICAL = "critical"
    WARNING = "warning"


# ═══════════════════════════════════════════════════════════════
#  EVIDENCE SCHEMAS
# ═══════════════════════════════════════════════════════════════


class EvidenceItem(BaseModel):
    """A single piece of evidence extracted from an agent's findings.

    Attributes:
        source: Agent that produced this evidence.
        description: Human-readable description.
        severity: Severity of the underlying finding.
        strength: How strongly this evidence supports a hypothesis.
        timestamp: When the evidence was observed.
        raw_data: Original data from the source agent.
    """
    source: EvidenceSource
    description: str
    severity: Severity = Severity.MEDIUM
    strength: EvidenceStrength = EvidenceStrength.MODERATE
    timestamp: str = ""
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class CrossAgentCorrelation(BaseModel):
    """Correlation detected across multiple agent findings.

    Attributes:
        sources: Which agents' data correlated.
        description: What was correlated.
        correlation_score: Strength of correlation (0.0-1.0).
        evidence_ids: Indices of related evidence items.
    """
    sources: List[EvidenceSource]
    description: str
    correlation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_ids: List[int] = Field(default_factory=list)


class AggregatedEvidence(BaseModel):
    """All evidence collected and aggregated from agent findings.

    Attributes:
        evidence_items: Individual evidence pieces.
        correlations: Cross-agent correlations.
        total_evidence_count: Total evidence items.
        strong_evidence_count: Count of strong evidence.
        sources_represented: Which agents contributed.
        dominant_severity: Most common severity.
        aggregation_latency_ms: Time to aggregate.
    """
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    correlations: List[CrossAgentCorrelation] = Field(default_factory=list)
    total_evidence_count: int = 0
    strong_evidence_count: int = 0
    sources_represented: List[EvidenceSource] = Field(default_factory=list)
    dominant_severity: Severity = Severity.MEDIUM
    aggregation_latency_ms: float = Field(default=0.0, ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  PATTERN SCHEMAS
# ═══════════════════════════════════════════════════════════════


class PatternIndicator(BaseModel):
    """An indicator that matches a known failure pattern.

    Attributes:
        indicator_name: Name of the thing being checked.
        matched: Whether this indicator was found.
        weight: How important this indicator is (0.0-1.0).
        evidence_description: What specifically matched.
    """
    indicator_name: str
    matched: bool = False
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_description: str = ""


class KnownPattern(BaseModel):
    """A known failure pattern from the pattern library.

    Attributes:
        pattern_name: Name from PatternName enum.
        description: Human-readable description.
        category: Incident category this pattern belongs to.
        indicators: Indicators to check.
        typical_mttr_minutes: Typical mean-time-to-repair.
        validation_tests: Suggested tests to validate.
    """
    pattern_name: PatternName
    description: str = ""
    category: IncidentCategory = IncidentCategory.UNKNOWN
    indicators: List[PatternIndicator] = Field(default_factory=list)
    typical_mttr_minutes: float = 30.0
    validation_tests: List[str] = Field(default_factory=list)


class PatternMatch(BaseModel):
    """Result of matching evidence against a known pattern.

    Attributes:
        pattern_name: Which pattern matched.
        match_score: Score 0.0-1.0 (weighted indicator match).
        matched_indicators: Indicators that matched.
        total_indicators: Total indicators checked.
        category: Incident category from pattern.
        description: Human-readable match description.
    """
    pattern_name: PatternName
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_indicators: int = 0
    total_indicators: int = 0
    category: IncidentCategory = IncidentCategory.UNKNOWN
    description: str = ""


# ═══════════════════════════════════════════════════════════════
#  HYPOTHESIS SCHEMAS
# ═══════════════════════════════════════════════════════════════


class CausalChainLink(BaseModel):
    """A single link in a causal chain.

    Attributes:
        step: Order in the chain (1-based).
        service: Service involved.
        event: What happened.
        relationship: Causal relationship to next step.
    """
    step: int = Field(ge=1)
    service: str
    event: str
    relationship: CausalRelationship = CausalRelationship.CAUSES


class CausalChain(BaseModel):
    """A causal chain explaining how an incident propagated.

    Attributes:
        chain: Ordered list of causal links.
        root_cause_service: Service at the start.
        terminal_effect: Final symptom observed.
        chain_confidence: Confidence in this chain 0.0-1.0.
    """
    chain: List[CausalChainLink] = Field(default_factory=list)
    root_cause_service: str = ""
    terminal_effect: str = ""
    chain_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ValidationTest(BaseModel):
    """A suggested test to validate or invalidate a hypothesis.

    Attributes:
        test_name: Name of the test.
        description: What to check.
        expected_outcome: What a positive result looks like.
        priority: Priority ordering (lower = more important).
    """
    test_name: str
    description: str = ""
    expected_outcome: str = ""
    priority: int = Field(default=1, ge=1, le=10)


class Hypothesis(BaseModel):
    """A root cause hypothesis with supporting evidence and scoring.

    Attributes:
        hypothesis_id: Unique identifier.
        theory: Human-readable theory statement.
        category: Incident category.
        severity: Assessed severity.
        likelihood_score: Probability score 0.0-1.0.
        evidence_supporting: Evidence items that support this.
        evidence_contradicting: Evidence items that contradict this.
        causal_chain: How the incident propagated.
        validation_tests: Tests to verify this hypothesis.
        pattern_match: Matched known pattern (if any).
        estimated_mttr_minutes: Estimated time to resolve.
        status: Active, pruned, etc.
        reasoning: Why this hypothesis was generated.
    """
    hypothesis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    theory: str
    category: IncidentCategory = IncidentCategory.UNKNOWN
    severity: Severity = Severity.MEDIUM
    likelihood_score: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_supporting: List[str] = Field(default_factory=list)
    evidence_contradicting: List[str] = Field(default_factory=list)
    causal_chain: Optional[CausalChain] = None
    validation_tests: List[ValidationTest] = Field(default_factory=list)
    pattern_match: Optional[PatternMatch] = None
    estimated_mttr_minutes: float = Field(default=30.0, ge=0.0)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    reasoning: str = ""

    @field_validator("likelihood_score")
    @classmethod
    def clamp_likelihood(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ═══════════════════════════════════════════════════════════════
#  INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════


class LogFindings(BaseModel):
    """Summarized findings from the log agent.

    Attributes:
        suspicious_services: Services with error patterns.
        total_error_logs: Total error logs observed.
        dominant_service: Most error-heavy service.
        system_wide_spike: Whether a system-wide error spike exists.
        potential_upstream_failure: Upstream failure signal.
        database_errors_detected: DB-specific errors found.
        error_keywords: Keywords detected across logs.
        confidence_score: Log agent confidence.
    """
    suspicious_services: List[Dict[str, Any]] = Field(default_factory=list)
    total_error_logs: int = 0
    dominant_service: Optional[str] = None
    system_wide_spike: bool = False
    potential_upstream_failure: bool = False
    database_errors_detected: bool = False
    error_keywords: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MetricFindings(BaseModel):
    """Summarized findings from the metrics agent.

    Attributes:
        anomalous_metrics: List of anomalous metric summaries.
        correlations: Metric correlations found.
        total_anomalies: Count of anomalies.
        critical_anomalies: Count of critical anomalies.
        resource_saturation: Whether saturation detected.
        cascading_degradation: Whether cascading degradation.
        confidence_score: Metrics agent confidence.
    """
    anomalous_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    correlations: List[Dict[str, Any]] = Field(default_factory=list)
    total_anomalies: int = 0
    critical_anomalies: int = 0
    resource_saturation: bool = False
    cascading_degradation: bool = False
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class DependencyFindings(BaseModel):
    """Summarized findings from the dependency agent.

    Attributes:
        failed_service: The primary failed service name.
        blast_radius_count: Total affected services.
        is_cascading: Whether cascading failure detected.
        cascade_pattern: Pattern of cascade.
        single_points_of_failure: SPOF service names.
        bottleneck_services: Bottleneck service names.
        critical_path: Critical path services.
        graph_has_cycles: Whether graph contains cycles.
        confidence_score: Dependency agent confidence.
    """
    failed_service: Optional[str] = None
    blast_radius_count: int = 0
    is_cascading: bool = False
    cascade_pattern: str = "isolated"
    single_points_of_failure: List[str] = Field(default_factory=list)
    bottleneck_services: List[str] = Field(default_factory=list)
    critical_path: List[str] = Field(default_factory=list)
    graph_has_cycles: bool = False
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class HistoricalIncident(BaseModel):
    """A past incident for historical similarity comparison.

    Attributes:
        incident_id: Unique identifier.
        title: Short incident title.
        root_cause: What the root cause was.
        category: Incident category.
        resolution: How it was resolved.
        mttr_minutes: How long it took to resolve.
        indicators: What indicators were present.
    """
    incident_id: str = ""
    title: str = ""
    root_cause: str = ""
    category: IncidentCategory = IncidentCategory.UNKNOWN
    resolution: str = ""
    mttr_minutes: float = 30.0
    indicators: List[str] = Field(default_factory=list)


class HypothesisAgentInput(BaseModel):
    """Input to the hypothesis agent from upstream agents.

    Attributes:
        log_findings: Findings from log agent.
        metric_findings: Findings from metrics agent.
        dependency_findings: Findings from dependency agent.
        historical_context: Past incidents for comparison.
        incident_id: Current incident identifier.
        correlation_id: Request correlation ID.
        time_window: Analysis time window.
    """
    log_findings: LogFindings = Field(default_factory=LogFindings)
    metric_findings: MetricFindings = Field(default_factory=MetricFindings)
    dependency_findings: DependencyFindings = Field(
        default_factory=DependencyFindings
    )
    historical_context: List[HistoricalIncident] = Field(default_factory=list)
    incident_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    correlation_id: str = ""
    time_window: str = "5m"


# ═══════════════════════════════════════════════════════════════
#  VALIDATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ValidatorError(BaseModel):
    """A single validation error or warning.

    Attributes:
        check_number: Validation check number (1-27).
        check_name: Symbolic check name.
        error_description: What failed.
        expected: Expected value or condition.
        actual: Actual value.
        severity: CRITICAL or WARNING.
    """
    check_number: int = Field(ge=1, le=27)
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
#  PIPELINE METADATA
# ═══════════════════════════════════════════════════════════════


class PipelineMetadata(BaseModel):
    """Telemetry metadata for the hypothesis pipeline.

    Attributes:
        evidence_aggregation_time_ms: Phase 1a time.
        pattern_matching_time_ms: Phase 1b time.
        hypothesis_generation_time_ms: Phase 2a time.
        causal_reasoning_time_ms: Phase 2b time.
        ranking_time_ms: Phase 3a time.
        validation_time_ms: Phase 3b time.
        total_time_ms: End-to-end latency.
        used_llm: Whether LLM was used.
        used_fallback: Whether fallback was used.
        cache_hit: Whether LLM cache was hit.
        correlation_id: Request correlation ID.
    """
    evidence_aggregation_time_ms: float = Field(default=0.0, ge=0.0)
    pattern_matching_time_ms: float = Field(default=0.0, ge=0.0)
    hypothesis_generation_time_ms: float = Field(default=0.0, ge=0.0)
    causal_reasoning_time_ms: float = Field(default=0.0, ge=0.0)
    ranking_time_ms: float = Field(default=0.0, ge=0.0)
    validation_time_ms: float = Field(default=0.0, ge=0.0)
    total_time_ms: float = Field(default=0.0, ge=0.0)
    used_llm: bool = False
    used_fallback: bool = False
    cache_hit: bool = False
    correlation_id: str = ""


# ═══════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════


class HypothesisAgentOutput(BaseModel):
    """Complete output from the hypothesis agent.

    Attributes:
        agent: Agent identifer (frozen to "hypothesis_agent").
        analysis_timestamp: When analysis was run.
        time_window: Analysis time window.
        incident_id: Current incident identifier.
        hypotheses: Ranked list of hypotheses.
        confidence_score: Overall confidence (0.0-1.0).
        pattern_matches: Known patterns matched.
        recommended_hypothesis: Top-ranked hypothesis ID.
        hypothesis_summary: Human-readable summary.
        estimated_mttr_minutes: MTTR of recommended hypothesis.
        category: Best-guess incident category.
        severity: Assessed severity of incident.
        correlation_id: Request correlation ID.
        classification_source: "llm", "fallback", "cached".
        pipeline_latency_ms: End-to-end latency.
        metadata: Pipeline telemetry.
        validation: Validation check results.
    """
    agent: str = Field(default="hypothesis_agent", frozen=True)
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    time_window: str = "5m"
    incident_id: str = ""
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    pattern_matches: List[PatternMatch] = Field(default_factory=list)
    recommended_hypothesis: str = ""
    hypothesis_summary: str = ""
    estimated_mttr_minutes: float = Field(default=30.0, ge=0.0)
    category: IncidentCategory = IncidentCategory.UNKNOWN
    severity: Severity = Severity.MEDIUM
    correlation_id: str = ""
    classification_source: str = "deterministic"
    pipeline_latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: Optional[PipelineMetadata] = None
    validation: Optional[ValidationResult] = None

    @field_validator("confidence_score")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))
