"""
File: schema.py
Purpose: Type-safe Pydantic v2 schemas for the metrics agent pipeline.
Dependencies: pydantic >=2.0
Performance: Schema validation <1ms per object

Defines input/output contracts for every layer — metric aggregation,
anomaly detection, correlation, classification, and validation.
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
    """Classified direction of a metric's recent movement."""
    SUDDEN_SPIKE = "sudden_spike"
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"


class AnomalyType(str, Enum):
    """Type of statistical anomaly detected in a timeseries."""
    SPIKE = "spike"
    SUSTAINED = "sustained"
    OSCILLATING = "oscillating"
    NONE = "none"


class Severity(str, Enum):
    """Severity classification for an anomalous metric."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CorrelationRelationship(str, Enum):
    """Direction of Pearson correlation between two metrics."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    WEAK = "weak"


class ValidationSeverity(str, Enum):
    """Severity of a validation failure."""
    CRITICAL = "critical"
    WARNING = "warning"


# ═══════════════════════════════════════════════════════════════
#  INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════


class BaselineStats(BaseModel):
    """Baseline statistics for a single metric.

    Example::

        BaselineStats(mean=50.0, stddev=10.0)
    """
    mean: float = Field(..., description="Historical mean value")
    stddev: float = Field(
        ..., ge=0.0, description="Historical standard deviation"
    )


class MetricsAnalysisInput(BaseModel):
    """Raw input to the metrics agent pipeline.

    Constructed from observability layer data (MetricsStore queries).

    Example::

        MetricsAnalysisInput(
            service="payment-api",
            time_window="2026-02-13T10:00:00Z to 2026-02-13T10:15:00Z",
            metrics={"cpu_percent": [45.2, 48.1, 52.3, 87.6, 95.2]},
            baseline={"cpu_percent": BaselineStats(mean=50.0, stddev=10.0)},
        )
    """
    service: str = Field(
        ..., min_length=1, description="Service name being analyzed"
    )
    time_window: str = Field(
        ..., min_length=1, description="Human-readable time window string"
    )
    metrics: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Metric name → timeseries values (chronological)",
    )
    baseline: Dict[str, BaselineStats] = Field(
        default_factory=dict,
        description="Metric name → baseline statistics (mean, stddev)",
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID propagated through all layers",
    )

    @field_validator("metrics")
    @classmethod
    def validate_timeseries_non_empty_values(
        cls, v: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Ensure timeseries lists contain at least one value."""
        for metric, ts in v.items():
            if len(ts) == 0:
                raise ValueError(
                    f"Empty timeseries for metric '{metric}'"
                )
        return v


# ═══════════════════════════════════════════════════════════════
#  METRIC AGGREGATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class MetricSignal(BaseModel):
    """Aggregated signal for a single metric.

    Produced by the deterministic metric aggregator.

    Example::

        MetricSignal(
            metric_name="cpu_percent",
            current_value=95.2,
            previous_value=87.6,
            baseline_mean=50.0,
            baseline_stddev=10.0,
            zscore=4.52,
            deviation_percent=90.4,
            growth_rate=8.67,
            is_anomalous=True,
            trend=TrendType.STABLE,
            threshold_breached=True,
        )
    """
    metric_name: str
    current_value: float
    previous_value: float
    baseline_mean: float
    baseline_stddev: float = Field(ge=0.0)
    zscore: float
    deviation_percent: float
    growth_rate: float
    is_anomalous: bool
    trend: TrendType = TrendType.STABLE
    threshold_breached: bool = False


class AggregationResult(BaseModel):
    """Complete output of the deterministic metric aggregation layer.

    Example::

        AggregationResult(
            metric_signals=[MetricSignal(...)],
            total_metrics_analyzed=12,
            aggregation_latency_ms=0.82,
        )
    """
    metric_signals: List[MetricSignal]
    total_metrics_analyzed: int = Field(ge=0)
    aggregation_latency_ms: float = Field(ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  ANOMALY DETECTION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class AnomalyResult(BaseModel):
    """Anomaly classification for a single metric.

    Example::

        AnomalyResult(
            metric_name="cpu_percent",
            anomaly_type=AnomalyType.SPIKE,
            severity=Severity.HIGH,
            reasoning="CPU spike detected: growth 63.8%, zscore 4.52",
        )
    """
    metric_name: str
    anomaly_type: AnomalyType = AnomalyType.NONE
    severity: Severity = Severity.INFO
    reasoning: str = ""


class AnomalyDetectionResult(BaseModel):
    """Output of the anomaly detection layer."""
    anomalies: List[AnomalyResult]
    detection_latency_ms: float = Field(ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  CORRELATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class CorrelationResult(BaseModel):
    """Correlation between two metrics.

    Example::

        CorrelationResult(
            metric_1="cpu_percent",
            metric_2="api_latency_p99_ms",
            correlation_coefficient=0.94,
            relationship=CorrelationRelationship.POSITIVE,
            interpretation="High CPU correlates with P99 latency (r=0.94)",
        )
    """
    metric_1: str
    metric_2: str
    correlation_coefficient: float = Field(ge=-1.0, le=1.0)
    relationship: CorrelationRelationship = CorrelationRelationship.WEAK
    interpretation: str = ""


class CorrelationDetectionResult(BaseModel):
    """Output of the correlation detection layer."""
    correlations: List[CorrelationResult]
    detection_latency_ms: float = Field(ge=0.0)


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class AnomalousMetric(BaseModel):
    """A metric flagged as anomalous with full classification metadata.

    This is the primary element in the output's anomalous_metrics list.
    """
    metric_name: str
    current_value: float
    previous_value: float
    baseline_mean: float
    baseline_stddev: float = Field(ge=0.0)
    zscore: float
    deviation_percent: float
    growth_rate: float
    is_anomalous: bool = True
    anomaly_type: AnomalyType = AnomalyType.NONE
    severity: Severity = Severity.INFO
    trend: TrendType = TrendType.STABLE
    threshold_breached: bool = False
    reasoning: str = ""


class SystemSummary(BaseModel):
    """System-level summary for the final output."""
    total_metrics_analyzed: int = Field(ge=0)
    total_anomalies_detected: int = Field(ge=0)
    critical_anomalies: int = Field(ge=0, default=0)
    high_anomalies: int = Field(ge=0, default=0)
    medium_anomalies: int = Field(ge=0, default=0)
    resource_saturation: bool = False
    cascading_degradation: bool = False


class ClassificationResult(BaseModel):
    """Output of the classification layer (LLM or fallback).

    Example::

        ClassificationResult(
            anomalous_metrics=[AnomalousMetric(...)],
            correlations=[CorrelationResult(...)],
            system_summary=SystemSummary(...),
            confidence_score=0.92,
            classification_source="deterministic",
        )
    """
    anomalous_metrics: List[AnomalousMetric]
    correlations: List[CorrelationResult]
    system_summary: SystemSummary
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_reasoning: str = ""
    classification_source: str = Field(
        default="deterministic",
        description="'deterministic' | 'llm' | 'fallback' | 'cached'",
    )
    classification_latency_ms: float = Field(ge=0.0, default=0.0)


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
#  METADATA SCHEMA
# ═══════════════════════════════════════════════════════════════


class PipelineMetadata(BaseModel):
    """Pipeline execution metadata."""
    extraction_time_ms: float = Field(ge=0.0, default=0.0)
    classification_time_ms: float = Field(ge=0.0, default=0.0)
    validation_time_ms: float = Field(ge=0.0, default=0.0)
    total_time_ms: float = Field(ge=0.0, default=0.0)
    used_llm: bool = False
    used_fallback: bool = False
    cache_hit: bool = False
    correlation_id: str = ""


# ═══════════════════════════════════════════════════════════════
#  FINAL OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════


class MetricsAgentOutput(BaseModel):
    """Final output of the metrics agent — passed to downstream consumers.

    Example::

        MetricsAgentOutput(
            agent="metrics_agent",
            service="payment-api",
            anomalous_metrics=[AnomalousMetric(...)],
            correlations=[CorrelationResult(...)],
            system_summary=SystemSummary(...),
            confidence_score=0.92,
            metadata=PipelineMetadata(...),
            validation=ValidationResult(...),
        )
    """
    agent: str = Field(default="metrics_agent", frozen=True)
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )
    time_window: str
    service: str
    anomalous_metrics: List[AnomalousMetric]
    correlations: List[CorrelationResult]
    system_summary: SystemSummary
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_reasoning: str = ""
    correlation_id: str = ""
    classification_source: str = "deterministic"
    pipeline_latency_ms: float = Field(ge=0.0, default=0.0)
    metadata: Optional[PipelineMetadata] = None
    validation: Optional[ValidationResult] = None

    @field_validator("agent")
    @classmethod
    def agent_must_be_metrics_agent(cls, v: str) -> str:
        if v != "metrics_agent":
            raise ValueError(f"agent must be 'metrics_agent', got '{v}'")
        return v
