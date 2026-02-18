"""
File: config.py
Purpose: Production configuration for the root cause agent.
Dependencies: Standard library (dataclasses, os)
Performance: O(1) — static config, no I/O

Provides feature flags, weights, LLM settings,
scoring parameters, and performance budgets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SourceWeights:
    """Reliability weights per source agent (Algorithm 1).

    Higher weight = more trusted. Hypothesis > Dependency > Metrics > Log.

    Attributes:
        hypothesis_agent: Weight for hypothesis agent evidence.
        dependency_agent: Weight for dependency agent evidence.
        metrics_agent: Weight for metrics agent evidence.
        log_agent: Weight for log agent evidence.
    """
    hypothesis_agent: float = 1.0
    dependency_agent: float = 0.85
    metrics_agent: float = 0.7
    log_agent: float = 0.55


@dataclass(frozen=True)
class EvidenceTypeMultipliers:
    """Multipliers for evidence types (Algorithm 5).

    Attributes:
        direct: Multiplier for direct evidence.
        correlated: Multiplier for correlated evidence.
        circumstantial: Multiplier for circumstantial evidence.
    """
    direct: float = 1.0
    correlated: float = 0.7
    circumstantial: float = 0.4


@dataclass(frozen=True)
class ConfidenceConfig:
    """Bayesian confidence calculation parameters (Algorithm 2).

    Attributes:
        prior: Base prior probability.
        agent_count_weight: How much agent count affects posterior.
        agreement_weight: Weight for inter-agent agreement.
        evidence_strength_weight: Weight for evidence strength.
        min_confidence: Floor for confidence output.
        max_confidence: Ceiling for confidence output.
    """
    prior: float = 0.47
    agent_count_weight: float = 0.25
    agreement_weight: float = 0.40
    evidence_strength_weight: float = 0.35
    min_confidence: float = 0.05
    max_confidence: float = 0.99


@dataclass(frozen=True)
class VerdictLimits:
    """Bounds on verdict generation.

    Attributes:
        max_alternatives: Maximum alternative verdicts.
        min_alternatives_low_confidence: Minimum alts if confidence < threshold.
        low_confidence_threshold: Below this, require min alts.
        min_reasoning_length: Minimum reasoning character length.
        max_causal_chain_depth: Maximum causal chain links.
        max_timeline_events: Maximum timeline events.
    """
    max_alternatives: int = 5
    min_alternatives_low_confidence: int = 2
    low_confidence_threshold: float = 0.9
    min_reasoning_length: int = 50
    max_causal_chain_depth: int = 10
    max_timeline_events: int = 50


@dataclass(frozen=True)
class RecencyDecay:
    """Exponential recency decay parameters (Algorithm 5).

    Attributes:
        half_life_hours: Hours at which weight drops to 50%.
        max_age_hours: Beyond this age, minimum weight applied.
        min_weight: Minimum weight for old evidence.
    """
    half_life_hours: float = 1.0
    max_age_hours: float = 24.0
    min_weight: float = 0.1


@dataclass(frozen=True)
class TimelineDedupConfig:
    """Timeline deduplication settings (Algorithm 7).

    Attributes:
        close_event_threshold_seconds: Events within this window
            are considered duplicates.
    """
    close_event_threshold_seconds: float = 1.0


@dataclass(frozen=True)
class FeatureFlags:
    """Feature toggles for the root cause agent.

    Attributes:
        use_llm: Enable LLM verdict synthesis.
        fallback_to_deterministic: Fall back to rules on LLM failure.
        enable_caching: Enable LLM response cache.
        enable_validation: Run 30 validation checks.
        enable_contradiction_resolution: Detect/resolve contradictions.
        enable_timeline_reconstruction: Build incident timeline.
        enable_impact_assessment: Compute blast radius.
    """
    use_llm: bool = False
    fallback_to_deterministic: bool = True
    enable_caching: bool = True
    enable_validation: bool = True
    enable_contradiction_resolution: bool = True
    enable_timeline_reconstruction: bool = True
    enable_impact_assessment: bool = True


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration for verdict synthesis.

    Attributes:
        provider: Provider name ('groq', 'mock').
        model: Model identifier.
        api_key: API key (from env).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        timeout_seconds: Request timeout.
        circuit_failure_threshold: Failures before OPEN.
        circuit_cooldown_seconds: Seconds before HALF_OPEN.
        circuit_success_threshold: Successes to CLOSE.
        max_retries: Max retry attempts.
        retry_base_delay: Base delay for exponential backoff.
        cache_ttl_seconds: Cache entry TTL.
        cost_per_1k_input_tokens: USD per 1K input tokens.
        cost_per_1k_output_tokens: USD per 1K output tokens.
    """
    provider: str = "mock"
    model: str = "llama-3.3-70b-versatile"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2000
    timeout_seconds: float = 15.0
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 60.0
    circuit_success_threshold: int = 2
    max_retries: int = 3
    retry_base_delay: float = 1.0
    cache_ttl_seconds: int = 300
    cost_per_1k_input_tokens: float = 0.00059
    cost_per_1k_output_tokens: float = 0.00079


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance budgets and limits.

    Attributes:
        max_evidence_items: Maximum evidence items to process.
        max_pipeline_ms: Maximum pipeline latency.
        max_fallback_ms: Maximum fallback latency.
    """
    max_evidence_items: int = 500
    max_pipeline_ms: float = 2000.0
    max_fallback_ms: float = 100.0


@dataclass
class RootCauseAgentConfig:
    """Master configuration for the root cause agent.

    Example::

        config = RootCauseAgentConfig.from_env()
        agent = RootCauseAgent(config)
    """
    source_weights: SourceWeights = None  # type: ignore[assignment]
    evidence_types: EvidenceTypeMultipliers = None  # type: ignore[assignment]
    confidence: ConfidenceConfig = None  # type: ignore[assignment]
    limits: VerdictLimits = None  # type: ignore[assignment]
    recency: RecencyDecay = None  # type: ignore[assignment]
    timeline_dedup: TimelineDedupConfig = None  # type: ignore[assignment]
    features: FeatureFlags = None  # type: ignore[assignment]
    llm: LLMConfig = None  # type: ignore[assignment]
    performance: PerformanceConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.source_weights is None:
            self.source_weights = SourceWeights()
        if self.evidence_types is None:
            self.evidence_types = EvidenceTypeMultipliers()
        if self.confidence is None:
            self.confidence = ConfidenceConfig()
        if self.limits is None:
            self.limits = VerdictLimits()
        if self.recency is None:
            self.recency = RecencyDecay()
        if self.timeline_dedup is None:
            self.timeline_dedup = TimelineDedupConfig()
        if self.features is None:
            self.features = FeatureFlags()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()

    @classmethod
    def from_env(cls) -> RootCauseAgentConfig:
        """Build config from environment variables.

        Environment variables:
            GROQ_API_KEY: Groq API key.
            ROOT_CAUSE_AGENT_USE_LLM: 'true' to enable LLM.
            ROOT_CAUSE_AGENT_LLM_PROVIDER: Provider name.

        Returns:
            Configured RootCauseAgentConfig.
        """
        use_llm = os.getenv(
            "ROOT_CAUSE_AGENT_USE_LLM", "false"
        ).lower() == "true"
        provider = os.getenv(
            "ROOT_CAUSE_AGENT_LLM_PROVIDER", "mock"
        )
        api_key = os.getenv("GROQ_API_KEY", "")

        return cls(
            features=FeatureFlags(use_llm=use_llm),
            llm=LLMConfig(
                provider=provider,
                api_key=api_key or None,
            ),
        )
