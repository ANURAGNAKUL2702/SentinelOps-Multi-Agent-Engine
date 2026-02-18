"""
File: config.py
Purpose: Frozen dataclass configuration for the validation agent.
Dependencies: Standard library only.
Performance: O(1) attribute access.

All configuration is immutable after construction.
Use from_env() for environment-based overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class AccuracyConfig:
    """Accuracy calculation thresholds.

    Attributes:
        exact_match_score: Score for exact root cause match.
        fuzzy_threshold: Minimum fuzzy score to consider correct.
        correct_threshold: Accuracy threshold for verdict_correct.
        normalize_whitespace: Normalize whitespace in comparison.
        case_insensitive: Case-insensitive comparison.
    """
    exact_match_score: float = 1.0
    fuzzy_threshold: float = 0.8
    correct_threshold: float = 0.8
    normalize_whitespace: bool = True
    case_insensitive: bool = True


@dataclass(frozen=True)
class PrecisionRecallConfig:
    """Precision/recall calculation settings.

    Attributes:
        confidence_threshold: Threshold to classify as positive.
        zero_division_default: Default value for division by zero.
    """
    confidence_threshold: float = 0.7
    zero_division_default: float = 0.0


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration curve settings.

    Attributes:
        num_bins: Number of bins for calibration curve.
        min_samples_per_bin: Minimum samples for a bin to be valid.
    """
    num_bins: int = 10
    min_samples_per_bin: int = 1


@dataclass(frozen=True)
class EvidenceValidationConfig:
    """Evidence validation settings.

    Attributes:
        timestamp_tolerance_seconds: Tolerance for timestamp matching.
        require_service_match: Require evidence services in ground truth.
    """
    timestamp_tolerance_seconds: float = 300.0
    require_service_match: bool = True


@dataclass(frozen=True)
class TimelineConfig:
    """Timeline validation settings.

    Attributes:
        timing_tolerance_seconds: Tolerance for timing delta matching.
        order_weight: Weight for ordering accuracy.
        timing_weight: Weight for timing accuracy.
    """
    timing_tolerance_seconds: float = 10.0
    order_weight: float = 0.6
    timing_weight: float = 0.4


@dataclass(frozen=True)
class HallucinationConfig:
    """Hallucination detection settings.

    Attributes:
        check_services: Detect fabricated services.
        check_dependencies: Detect phantom dependencies.
        check_metrics: Detect fake metrics.
        known_services: Pre-configured known service names.
    """
    check_services: bool = True
    check_dependencies: bool = True
    check_metrics: bool = True
    known_services: tuple = (
        "api-gateway", "auth-service", "payment-service",
        "fraud-service", "notification-service", "database",
        "cache-service", "merchant-portal",
    )


@dataclass(frozen=True)
class RecommendationThresholds:
    """Thresholds that trigger recommendations.

    Attributes:
        critical_accuracy: Below this triggers critical recommendation.
        evidence_accuracy: Below this triggers evidence recommendation.
        timeline_accuracy: Below this triggers timeline recommendation.
        calibration_error: Above this triggers calibration recommendation.
        accuracy_for_recs: Below this requires at least 1 recommendation.
    """
    critical_accuracy: float = 0.5
    evidence_accuracy: float = 0.6
    timeline_accuracy: float = 0.7
    calibration_error: float = 0.2
    accuracy_for_recs: float = 0.9


@dataclass(frozen=True)
class FeatureFlags:
    """Feature toggle flags.

    Attributes:
        use_llm: Enable LLM-based discrepancy analysis.
        enable_validation: Run output validation checks.
        fallback_to_deterministic: Fall back if LLM fails.
        enable_calibration: Calculate calibration curves.
        enable_hallucination_detection: Detect hallucinations.
    """
    use_llm: bool = False
    enable_validation: bool = True
    fallback_to_deterministic: bool = True
    enable_calibration: bool = True
    enable_hallucination_detection: bool = True


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration.

    Attributes:
        provider: LLM provider name ('mock' or 'groq').
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        api_key: API key (from env if empty).
        timeout_seconds: Request timeout.
    """
    provider: str = "mock"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1024
    api_key: str = ""
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance budgets.

    Attributes:
        max_pipeline_ms: Max total pipeline latency.
        max_fallback_ms: Max fallback latency.
        max_cost_per_call: Max USD per LLM call.
    """
    max_pipeline_ms: float = 1000.0
    max_fallback_ms: float = 50.0
    max_cost_per_call: float = 0.0005


@dataclass
class ValidationAgentConfig:
    """Master configuration for the validation agent.

    Composes all sub-configs with sensible defaults.
    """
    accuracy: AccuracyConfig = None  # type: ignore[assignment]
    precision_recall: PrecisionRecallConfig = None  # type: ignore[assignment]
    calibration: CalibrationConfig = None  # type: ignore[assignment]
    evidence: EvidenceValidationConfig = None  # type: ignore[assignment]
    timeline: TimelineConfig = None  # type: ignore[assignment]
    hallucination: HallucinationConfig = None  # type: ignore[assignment]
    recommendations: RecommendationThresholds = None  # type: ignore[assignment]
    features: FeatureFlags = None  # type: ignore[assignment]
    llm: LLMConfig = None  # type: ignore[assignment]
    performance: PerformanceConfig = None  # type: ignore[assignment]
    all_failure_types: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Populate None fields with frozen defaults."""
        if self.accuracy is None:
            object.__setattr__(self, "accuracy", AccuracyConfig())
        if self.precision_recall is None:
            object.__setattr__(
                self, "precision_recall", PrecisionRecallConfig()
            )
        if self.calibration is None:
            object.__setattr__(self, "calibration", CalibrationConfig())
        if self.evidence is None:
            object.__setattr__(
                self, "evidence", EvidenceValidationConfig()
            )
        if self.timeline is None:
            object.__setattr__(self, "timeline", TimelineConfig())
        if self.hallucination is None:
            object.__setattr__(
                self, "hallucination", HallucinationConfig()
            )
        if self.recommendations is None:
            object.__setattr__(
                self, "recommendations", RecommendationThresholds()
            )
        if self.features is None:
            object.__setattr__(self, "features", FeatureFlags())
        if self.llm is None:
            object.__setattr__(self, "llm", LLMConfig())
        if self.performance is None:
            object.__setattr__(self, "performance", PerformanceConfig())
        if self.all_failure_types is None:
            object.__setattr__(self, "all_failure_types", [
                "resource_exhaustion", "network_partition", "crash",
                "configuration_error", "deployment_failure",
                "security_breach", "database_failure", "unknown",
            ])

    @classmethod
    def from_env(cls) -> ValidationAgentConfig:
        """Create config with environment variable overrides.

        Returns:
            ValidationAgentConfig with env-based overrides.
        """
        llm_provider = os.environ.get(
            "VALIDATION_LLM_PROVIDER", "mock"
        )
        api_key = os.environ.get("GROQ_API_KEY", "")
        use_llm = os.environ.get(
            "VALIDATION_USE_LLM", "false"
        ).lower() == "true"

        return cls(
            llm=LLMConfig(provider=llm_provider, api_key=api_key),
            features=FeatureFlags(use_llm=use_llm),
        )
