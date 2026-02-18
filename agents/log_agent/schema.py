"""
File: schema.py
Purpose: Type-safe Pydantic schemas for the entire log agent pipeline.
Dependencies: pydantic >=2.0
Performance: Schema validation <1ms per object

Defines input/output contracts for every layer so data flows
are type-checked at runtime with descriptive validation errors.
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
    model_validator,
)


# ═══════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════


class TrendType(str, Enum):
    """Classified direction of error trend over time."""
    SUDDEN_SPIKE = "sudden_spike"
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"


class SeverityHint(str, Enum):
    """Severity classification for a suspicious service."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationSeverity(str, Enum):
    """Severity of a validation failure."""
    CRITICAL = "critical"
    WARNING = "warning"


# ═══════════════════════════════════════════════════════════════
#  INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════


class LogAnalysisInput(BaseModel):
    """Raw input to the log agent pipeline.

    Constructed from observability layer data (LogStore queries).

    Example::

        LogAnalysisInput(
            error_summary={"payment-service": 340, "auth-service": 12},
            total_error_logs=352,
            error_trends={"payment-service": [0, 0, 5, 30, 340]},
            keyword_matches={"payment-service": ["database timeout"]},
            time_window="2026-02-13T10:00:00Z to 2026-02-13T10:15:00Z",
            correlation_id="abc-123",
        )
    """
    error_summary: Dict[str, int] = Field(
        ..., description="Service name → total error count"
    )
    total_error_logs: int = Field(
        ..., ge=0, description="Sum of all error logs across services"
    )
    error_trends: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Service → error counts per time bucket (chronological)",
    )
    keyword_matches: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Service → list of matched error keywords",
    )
    time_window: str = Field(
        ..., min_length=1, description="Human-readable time window string"
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID propagated through all layers",
    )

    @field_validator("error_summary")
    @classmethod
    def validate_error_counts_non_negative(
        cls, v: Dict[str, int]
    ) -> Dict[str, int]:
        for svc, count in v.items():
            if count < 0:
                raise ValueError(
                    f"Negative error count for {svc}: {count}"
                )
        return v

    @model_validator(mode="after")
    def validate_total_matches_summary(self) -> "LogAnalysisInput":
        summary_total = sum(self.error_summary.values())
        if summary_total > 0 and self.total_error_logs <= 0:
            raise ValueError(
                f"total_error_logs is {self.total_error_logs} but "
                f"error_summary sums to {summary_total}"
            )
        return self


# ═══════════════════════════════════════════════════════════════
#  SIGNAL EXTRACTION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ServiceSignal(BaseModel):
    """Extracted signals for a single service.

    Produced by the deterministic signal extractor (replaces scanner.txt).

    Example::

        ServiceSignal(
            service="payment-service",
            error_count=340,
            error_percentage=96.6,
            first_non_zero_trend_index=2,
            growth_rate_last_period=2166.7,
            trend_type=TrendType.SUDDEN_SPIKE,
            critical_keyword=True,
            moderate_keyword=False,
            log_flooding_signal=False,
            dominant_service_signal=True,
        )
    """
    service: str
    error_count: int = Field(ge=0)
    error_percentage: float = Field(ge=0.0, le=100.0)
    first_non_zero_trend_index: int = Field(ge=-1)
    growth_rate_last_period: float = Field(default=0.0)
    trend_type: TrendType = Field(default=TrendType.STABLE)
    critical_keyword: bool = False
    moderate_keyword: bool = False
    log_flooding_signal: bool = False
    dominant_service_signal: bool = False


class SystemSignal(BaseModel):
    """System-wide aggregate signals.

    Example::

        SystemSignal(
            total_error_logs=352,
            affected_service_count=1,
            earliest_error_service="payment-service",
            cascading_candidate=False,
        )
    """
    total_error_logs: int = Field(ge=0)
    affected_service_count: int = Field(ge=0)
    earliest_error_service: Optional[str] = None
    cascading_candidate: bool = False


class SignalExtractionResult(BaseModel):
    """Complete output of the deterministic signal extraction layer.

    Example::

        SignalExtractionResult(
            service_signals=[ServiceSignal(...)],
            system_signals=SystemSignal(...),
            extraction_latency_ms=12.4,
        )
    """
    service_signals: List[ServiceSignal]
    system_signals: SystemSignal
    extraction_latency_ms: float = Field(ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  ANOMALY DETECTION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class AnomalyResult(BaseModel):
    """Statistical anomaly detected on a metric."""
    service: str
    metric: str
    value: float
    z_score: float
    threshold: float
    is_anomaly: bool


class AnomalyDetectionResult(BaseModel):
    """Output of the anomaly detection layer."""
    anomalies: List[AnomalyResult]
    detection_latency_ms: float = Field(ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class SuspiciousService(BaseModel):
    """A service flagged as suspicious with classification metadata."""
    service: str
    error_count: int = Field(ge=0)
    error_percentage: float = Field(ge=0.0, le=100.0)
    error_keywords_detected: List[str] = Field(default_factory=list)
    error_trend: TrendType = TrendType.STABLE
    severity_hint: SeverityHint = SeverityHint.LOW
    log_flooding: bool = False


class SystemErrorSummary(BaseModel):
    """System-level error summary for the final output."""
    total_error_logs: int = Field(ge=0)
    dominant_service: Optional[str] = None
    system_wide_spike: bool = False
    potential_upstream_failure: bool = False


class ClassificationResult(BaseModel):
    """Output of the classification layer (LLM or fallback).

    Example::

        ClassificationResult(
            suspicious_services=[SuspiciousService(...)],
            system_error_summary=SystemErrorSummary(...),
            database_related_errors_detected=True,
            confidence_score=0.9,
            classification_source="deterministic",
            classification_latency_ms=5.2,
        )
    """
    suspicious_services: List[SuspiciousService]
    system_error_summary: SystemErrorSummary
    database_related_errors_detected: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0)
    classification_source: str = Field(
        default="deterministic",
        description="'deterministic' | 'llm' | 'fallback' | 'cached'",
    )
    classification_latency_ms: float = Field(ge=0.0, default=0.0)

    @model_validator(mode="after")
    def validate_confidence_vs_services(self) -> "ClassificationResult":
        if not self.suspicious_services and self.confidence_score >= 0.3:
            # Auto-correct: clamp confidence when no suspicious services
            object.__setattr__(self, "confidence_score", 0.0)
        return self


# ═══════════════════════════════════════════════════════════════
#  VALIDATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ValidatorError(BaseModel):
    """A single validation failure."""
    check_number: int
    check_name: str
    error_description: str
    expected: str
    actual: str
    severity: ValidationSeverity = ValidationSeverity.WARNING


class ValidationResult(BaseModel):
    """Output of the validation layer."""
    validation_passed: bool
    checks_executed: int = Field(ge=0)
    errors: List[ValidatorError] = Field(default_factory=list)
    warnings: List[ValidatorError] = Field(default_factory=list)
    validation_latency_ms: float = Field(ge=0.0, default=0.0)


# ═══════════════════════════════════════════════════════════════
#  FINAL OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════


class LogAgentOutput(BaseModel):
    """Final output of the log agent — passed to downstream consumers.

    Matches the output schema defined in synthesizer.txt.

    Example::

        LogAgentOutput(
            agent="log_agent",
            analysis_timestamp="2026-02-13T10:15:23Z",
            time_window="2026-02-13T10:00:00Z to 2026-02-13T10:15:00Z",
            suspicious_services=[...],
            system_error_summary=SystemErrorSummary(...),
            database_related_errors_detected=True,
            confidence_score=0.9,
            correlation_id="abc-123",
            classification_source="deterministic",
            pipeline_latency_ms=45.2,
            validation=ValidationResult(...),
        )
    """
    agent: str = Field(default="log_agent", frozen=True)
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )
    time_window: str
    suspicious_services: List[SuspiciousService]
    system_error_summary: SystemErrorSummary
    database_related_errors_detected: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0)
    correlation_id: str = ""
    classification_source: str = "deterministic"
    pipeline_latency_ms: float = Field(ge=0.0, default=0.0)
    validation: Optional[ValidationResult] = None

    @field_validator("agent")
    @classmethod
    def agent_must_be_log_agent(cls, v: str) -> str:
        if v != "log_agent":
            raise ValueError(f"agent must be 'log_agent', got '{v}'")
        return v
