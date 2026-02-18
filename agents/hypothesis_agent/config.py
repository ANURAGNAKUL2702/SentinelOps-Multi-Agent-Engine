"""
File: config.py
Purpose: Production configuration for the hypothesis agent.
Dependencies: Standard library (dataclasses, os)
Performance: O(1) — static config, no I/O

Provides feature flags, threshold controls, LLM settings,
scoring weights, and performance budgets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for hypothesis likelihood scoring (Algorithm 5).

    Attributes:
        supporting_evidence: Weight for supporting evidence count.
        contradicting_evidence: Penalty weight for contradicting evidence.
        pattern_match: Weight for known pattern match score.
        cross_agent_correlation: Weight for cross-agent correlations.
        historical_similarity: Weight for historical incident match.
        causal_chain_strength: Weight for causal chain confidence.
    """
    supporting_evidence: float = 0.25
    contradicting_evidence: float = 0.3
    pattern_match: float = 0.35
    cross_agent_correlation: float = 0.2
    historical_similarity: float = 0.1
    causal_chain_strength: float = 0.1


@dataclass(frozen=True)
class HypothesisLimits:
    """Bounds on hypothesis generation and pruning.

    Attributes:
        min_hypotheses: Minimum hypotheses to generate.
        max_hypotheses: Maximum hypotheses to keep after pruning.
        pruning_threshold: Minimum likelihood to survive pruning.
        min_evidence_for_hypothesis: Minimum evidence items needed.
    """
    min_hypotheses: int = 3
    max_hypotheses: int = 5
    pruning_threshold: float = 0.15
    min_evidence_for_hypothesis: int = 1


@dataclass(frozen=True)
class PatternThresholds:
    """Thresholds for pattern matching (Algorithm 2).

    Attributes:
        min_match_score: Minimum score for a pattern to be considered.
        min_indicators_matched: Minimum indicators to match.
        high_match_threshold: Score above which match is "strong".
    """
    min_match_score: float = 0.3
    min_indicators_matched: int = 2
    high_match_threshold: float = 0.7


@dataclass(frozen=True)
class FeatureFlags:
    """Feature toggles for the hypothesis agent.

    Attributes:
        use_llm: Enable LLM hypothesis generation path.
        fallback_to_rules: Fall back to rules on LLM failure.
        enable_caching: Enable LLM response cache.
        enable_validation: Run 27 validation checks.
        enable_pattern_matching: Run pattern matching.
        enable_causal_reasoning: Run LLM causal chain construction.
        enable_historical_search: Search historical incidents.
    """
    use_llm: bool = False
    fallback_to_rules: bool = True
    enable_caching: bool = True
    enable_validation: bool = True
    enable_pattern_matching: bool = True
    enable_causal_reasoning: bool = True
    enable_historical_search: bool = True


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration for hypothesis generation.

    Higher temperature (0.3) and more tokens (1500) than other agents
    because hypothesis generation requires creativity and reasoning.

    Attributes:
        provider: Provider name ('groq', 'mock').
        model: Model identifier.
        api_key: API key (from env).
        temperature: Sampling temperature (0.3 for creativity).
        max_tokens: Max output tokens (1500 for complex reasoning).
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
    temperature: float = 0.3
    max_tokens: int = 1500
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
        max_historical_incidents: Maximum historical incidents to search.
        max_causal_chain_depth: Maximum causal chain length.
    """
    max_evidence_items: int = 500
    max_historical_incidents: int = 100
    max_causal_chain_depth: int = 10


@dataclass
class HypothesisAgentConfig:
    """Master configuration for the hypothesis agent.

    Example::

        config = HypothesisAgentConfig.from_env()
        agent = HypothesisAgent(config)
    """
    scoring: ScoringWeights = None  # type: ignore[assignment]
    limits: HypothesisLimits = None  # type: ignore[assignment]
    patterns: PatternThresholds = None  # type: ignore[assignment]
    features: FeatureFlags = None  # type: ignore[assignment]
    llm: LLMConfig = None  # type: ignore[assignment]
    performance: PerformanceConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scoring is None:
            self.scoring = ScoringWeights()
        if self.limits is None:
            self.limits = HypothesisLimits()
        if self.patterns is None:
            self.patterns = PatternThresholds()
        if self.features is None:
            self.features = FeatureFlags()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()

    @classmethod
    def from_env(cls) -> HypothesisAgentConfig:
        """Build config from environment variables.

        Environment variables:
            GROQ_API_KEY: Groq API key.
            HYPOTHESIS_AGENT_USE_LLM: "true" to enable LLM.
            HYPOTHESIS_AGENT_LLM_PROVIDER: Provider name.

        Returns:
            Configured HypothesisAgentConfig.
        """
        use_llm = os.getenv(
            "HYPOTHESIS_AGENT_USE_LLM", "false"
        ).lower() == "true"
        provider = os.getenv(
            "HYPOTHESIS_AGENT_LLM_PROVIDER", "mock"
        )
        api_key = os.getenv("GROQ_API_KEY", "")

        return cls(
            features=FeatureFlags(
                use_llm=use_llm,
            ),
            llm=LLMConfig(
                provider=provider,
                api_key=api_key or None,
            ),
        )
