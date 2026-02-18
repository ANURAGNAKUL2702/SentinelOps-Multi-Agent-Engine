"""
File: schema.py
Purpose: Type-safe Pydantic v2 schemas for the validation agent pipeline.
Dependencies: pydantic >=2.0
Performance: Schema validation <1ms per object

Defines input/output contracts — ground truth, validation report,
discrepancy tracking, hallucination detection, confusion matrix,
calibration bins, and all supporting types.
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


class HallucinationType(str, Enum):
    """Type of hallucination detected."""
    SERVICE = "service"
    DEPENDENCY = "dependency"
    METRIC = "metric"
    EVENT = "event"


class DiscrepancyType(str, Enum):
    """Type of discrepancy between verdict and ground truth."""
    ROOT_CAUSE_MISMATCH = "root_cause_mismatch"
    SERVICE_MISMATCH = "service_mismatch"
    TIMELINE_ERROR = "timeline_error"
    EVIDENCE_FABRICATION = "evidence_fabrication"
    CONFIDENCE_MISCALIBRATION = "confidence_miscalibration"
    CAUSAL_CHAIN_ERROR = "causal_chain_error"
    SEVERITY_MISMATCH = "severity_mismatch"


class DiscrepancySeverity(str, Enum):
    """Severity of a discrepancy."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationCheckSeverity(str, Enum):
    """Severity of a validation check failure."""
    CRITICAL = "critical"
    WARNING = "warning"


# ═══════════════════════════════════════════════════════════════
#  GROUND TRUTH INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════


class PropagationStep(BaseModel):
    """A single step in the failure propagation chain.

    Attributes:
        from_service: Source service of the propagation.
        to_service: Target service of the propagation.
        delay_seconds: Time delay between source and target failure.
        mechanism: How the failure propagated.
    """
    from_service: str
    to_service: str
    delay_seconds: float = Field(default=0.0, ge=0.0)
    mechanism: str = ""


class GroundTruth(BaseModel):
    """Ground truth from the simulation engine.

    Attributes:
        actual_root_cause: True root cause string.
        failure_type: Type of failure injected.
        injected_at: When the failure was injected (ISO-8601).
        affected_services_ground_truth: Actual impacted services.
        failure_propagation_chain: Actual cascade path.
        expected_symptoms: Observable symptoms expected.
        simulation_metadata: Scenario parameters.
    """
    actual_root_cause: str
    failure_type: str = ""
    injected_at: str = ""
    affected_services_ground_truth: List[str] = Field(default_factory=list)
    failure_propagation_chain: List[PropagationStep] = Field(
        default_factory=list
    )
    expected_symptoms: List[str] = Field(default_factory=list)
    simulation_metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
#  RE-EXPORT ROOT CAUSE AGENT TYPES (for convenience)
# ═══════════════════════════════════════════════════════════════

# We import from root_cause_agent to avoid redefining
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


# ═══════════════════════════════════════════════════════════════
#  VALIDATION AGENT INPUT
# ═══════════════════════════════════════════════════════════════


class ValidationAgentInput(BaseModel):
    """Input to the validation agent.

    Attributes:
        verdict: The root cause verdict to validate.
        ground_truth: Simulation ground truth data.
        incident_id: Current incident identifier.
        correlation_id: Request correlation ID.
        history: Historical predictions for calibration.
    """
    verdict: RootCauseVerdict
    ground_truth: GroundTruth
    incident_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    correlation_id: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  DISCREPANCY & HALLUCINATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class Discrepancy(BaseModel):
    """A specific mismatch between verdict and ground truth.

    Attributes:
        discrepancy_type: Category of the mismatch.
        description: Human-readable description.
        expected: What ground truth says.
        actual: What the verdict says.
        severity: How critical the mismatch is.
    """
    discrepancy_type: DiscrepancyType
    description: str = ""
    expected: str = ""
    actual: str = ""
    severity: DiscrepancySeverity = DiscrepancySeverity.MEDIUM


class Hallucination(BaseModel):
    """A fabricated element not in simulation data.

    Attributes:
        hallucination_type: SERVICE, DEPENDENCY, METRIC, or EVENT.
        description: What was fabricated.
        fabricated_value: The non-existent element referenced.
        context: Where in the verdict it appeared.
    """
    hallucination_type: HallucinationType
    description: str = ""
    fabricated_value: str = ""
    context: str = ""


# ═══════════════════════════════════════════════════════════════
#  CONFUSION MATRIX SCHEMA
# ═══════════════════════════════════════════════════════════════


class ConfusionMatrixResult(BaseModel):
    """Confusion matrix for classification evaluation.

    Attributes:
        tp: True positives.
        fp: False positives.
        tn: True negatives.
        fn: False negatives.
        matrix: Full multi-class matrix (rows=actual, cols=predicted).
        classes: Class labels used.
    """
    tp: int = Field(default=0, ge=0)
    fp: int = Field(default=0, ge=0)
    tn: int = Field(default=0, ge=0)
    fn: int = Field(default=0, ge=0)
    matrix: List[List[int]] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)

    @property
    def total(self) -> int:
        """Total count of all classifications."""
        return self.tp + self.fp + self.tn + self.fn


# ═══════════════════════════════════════════════════════════════
#  CALIBRATION SCHEMA
# ═══════════════════════════════════════════════════════════════


class CalibrationBin(BaseModel):
    """A single bin in the calibration curve.

    Attributes:
        bin_start: Lower bound of confidence bin.
        bin_end: Upper bound of confidence bin.
        avg_confidence: Average predicted confidence in this bin.
        avg_accuracy: Actual accuracy in this bin.
        count: Number of samples in this bin.
    """
    bin_start: float = Field(ge=0.0, le=1.0)
    bin_end: float = Field(ge=0.0, le=1.0)
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    count: int = Field(default=0, ge=0)


# ═══════════════════════════════════════════════════════════════
#  VALIDATION METADATA
# ═══════════════════════════════════════════════════════════════


class ValidationMetadata(BaseModel):
    """Metadata for the validation report.

    Attributes:
        correlation_id: Request correlation ID.
        validation_start: When validation began (ISO-8601).
        validation_end: When validation completed (ISO-8601).
        accuracy_calculation_ms: Time for accuracy calculation.
        precision_recall_ms: Time for precision/recall calculation.
        evidence_validation_ms: Time for evidence validation.
        timeline_validation_ms: Time for timeline validation.
        hallucination_detection_ms: Time for hallucination detection.
        confusion_matrix_ms: Time for confusion matrix building.
        calibration_ms: Time for calibration calculation.
        total_pipeline_ms: End-to-end latency.
        used_llm: Whether LLM was used for analysis.
        used_fallback: Whether fallback was used.
        cache_hit: Whether LLM cache was hit.
        agent_version: Validation agent version string.
    """
    correlation_id: str = ""
    validation_start: str = ""
    validation_end: str = ""
    accuracy_calculation_ms: float = Field(default=0.0, ge=0.0)
    precision_recall_ms: float = Field(default=0.0, ge=0.0)
    evidence_validation_ms: float = Field(default=0.0, ge=0.0)
    timeline_validation_ms: float = Field(default=0.0, ge=0.0)
    hallucination_detection_ms: float = Field(default=0.0, ge=0.0)
    confusion_matrix_ms: float = Field(default=0.0, ge=0.0)
    calibration_ms: float = Field(default=0.0, ge=0.0)
    total_pipeline_ms: float = Field(default=0.0, ge=0.0)
    used_llm: bool = False
    used_fallback: bool = False
    cache_hit: bool = False
    agent_version: str = "1.0.0"


# ═══════════════════════════════════════════════════════════════
#  VALIDATOR CHECK SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ValidatorError(BaseModel):
    """A single validation error or warning.

    Attributes:
        check_number: Validation check number (1-25).
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
    severity: ValidationCheckSeverity = ValidationCheckSeverity.WARNING


class ValidatorResult(BaseModel):
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
    errors: List[ValidatorError] = Field(default_factory=list)
    warnings: List[ValidatorError] = Field(default_factory=list)
    validation_latency_ms: float = Field(default=0.0, ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA — ValidationReport
# ═══════════════════════════════════════════════════════════════


class ValidationReport(BaseModel):
    """Complete output from the validation agent.

    Attributes:
        agent: Agent identifier (frozen to 'validation_agent').
        analysis_timestamp: When validation was run.
        verdict_correct: Whether the verdict matches ground truth.
        accuracy_score: Fuzzy match score 0.0-1.0.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1_score: Harmonic mean of precision and recall.
        confidence_calibration_error: MAE of predicted vs actual.
        evidence_accuracy: Fraction of evidence matching ground truth.
        timeline_accuracy: Fraction of timeline events correct.
        affected_services_accuracy: Jaccard similarity of service sets.
        discrepancies: Specific mismatches found.
        hallucinations: Fabricated evidence/services.
        recommendations: Suggestions to improve pipeline.
        confusion_matrix: TP/FP/TN/FN breakdown.
        calibration_curve: Calibration bins.
        metadata: Correlation ID, timestamps, latencies.
        correlation_id: Request correlation ID.
        classification_source: 'llm', 'fallback', 'deterministic'.
        pipeline_latency_ms: End-to-end latency.
        output_validation: Output validation check results.
    """
    agent: str = Field(default="validation_agent", frozen=True)
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    verdict_correct: bool = False
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_calibration_error: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    evidence_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    timeline_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    affected_services_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    discrepancies: List[Discrepancy] = Field(default_factory=list)
    hallucinations: List[Hallucination] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confusion_matrix: Optional[ConfusionMatrixResult] = None
    calibration_curve: List[CalibrationBin] = Field(default_factory=list)
    metadata: Optional[ValidationMetadata] = None
    correlation_id: str = ""
    classification_source: str = "deterministic"
    pipeline_latency_ms: float = Field(default=0.0, ge=0.0)
    output_validation: Optional[ValidatorResult] = None

    @field_validator("accuracy_score", "precision", "recall",
                     "f1_score", "evidence_accuracy",
                     "timeline_accuracy", "affected_services_accuracy")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp scores to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
