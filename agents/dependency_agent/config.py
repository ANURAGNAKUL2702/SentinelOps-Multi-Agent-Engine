"""
File: config.py
Purpose: Production configuration for the dependency agent.
Dependencies: Standard library (dataclasses, os)
Performance: O(1) — static config, no I/O

Provides feature flags, threshold controls, LLM settings,
and performance budgets.  All values configurable via environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DependencyThresholds:
    """Domain-specific thresholds for dependency graph analysis.

    Attributes:
        fan_in_threshold: High fan-in if > this count.
        fan_out_threshold: High fan-out if > this count.
        bottleneck_pct_threshold: Sequential bottleneck if > this %.
        slow_span_threshold_ms: Slow span if > this duration.
        spof_min_dependents: SPOF if >= this many dependents.
        high_error_rate_threshold: Edge unhealthy if error_rate > this %.
    """
    fan_in_threshold: int = 3
    fan_out_threshold: int = 5
    bottleneck_pct_threshold: float = 50.0
    slow_span_threshold_ms: float = 1000.0
    spof_min_dependents: int = 3
    high_error_rate_threshold: float = 5.0


@dataclass(frozen=True)
class CriticalityWeights:
    """Weights for criticality score calculation (Algorithm 5).

    Attributes:
        upstream_weight: Weight for upstream dependency count (max 0.3).
        downstream_weight: Weight for downstream dependency count (max 0.2).
        blast_radius_weight: Weight for blast radius size (max 0.3).
        critical_path_weight: Weight for critical path presence (0.2).
    """
    upstream_weight: float = 0.3
    downstream_weight: float = 0.2
    blast_radius_weight: float = 0.3
    critical_path_weight: float = 0.2
    upstream_divisor: float = 10.0
    downstream_divisor: float = 10.0
    blast_radius_divisor: float = 20.0


@dataclass(frozen=True)
class FeatureFlags:
    """Feature toggles for the dependency agent.

    Attributes:
        use_llm: Enable LLM classification path.
        fallback_to_rules: Fall back to rules on LLM failure.
        enable_caching: Enable LLM response cache.
        enable_validation: Run 25 validation checks.
        enable_trace_analysis: Parse distributed traces.
        enable_bottleneck_detection: Detect bottlenecks.
    """
    use_llm: bool = False
    fallback_to_rules: bool = True
    enable_caching: bool = True
    enable_validation: bool = True
    enable_trace_analysis: bool = True
    enable_bottleneck_detection: bool = True


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration (same structure as metrics_agent).

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
    temperature: float = 0.1
    max_tokens: int = 600
    timeout_seconds: float = 10.0
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
        max_services: Maximum graph nodes.
        max_dependencies: Maximum graph edges.
        max_trace_spans: Maximum spans per trace.
        max_graph_depth: Maximum BFS/DFS depth.
    """
    max_services: int = 10000
    max_dependencies: int = 50000
    max_trace_spans: int = 1000
    max_graph_depth: int = 20


@dataclass
class DependencyAgentConfig:
    """Master configuration for the dependency agent.

    Example::

        config = DependencyAgentConfig.from_env()
        agent = DependencyAgent(config)
    """
    thresholds: DependencyThresholds = None  # type: ignore[assignment]
    criticality: CriticalityWeights = None  # type: ignore[assignment]
    features: FeatureFlags = None  # type: ignore[assignment]
    llm: LLMConfig = None  # type: ignore[assignment]
    performance: PerformanceConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = DependencyThresholds()
        if self.criticality is None:
            self.criticality = CriticalityWeights()
        if self.features is None:
            self.features = FeatureFlags()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()

    @classmethod
    def from_env(cls) -> DependencyAgentConfig:
        """Build config from environment variables.

        Environment variables:
            GROQ_API_KEY: Groq API key.
            DEPENDENCY_AGENT_USE_LLM: "true" to enable LLM.
            DEPENDENCY_AGENT_LLM_PROVIDER: Provider name.

        Returns:
            Configured DependencyAgentConfig.
        """
        use_llm = os.getenv(
            "DEPENDENCY_AGENT_USE_LLM", "false"
        ).lower() == "true"
        provider = os.getenv(
            "DEPENDENCY_AGENT_LLM_PROVIDER", "mock"
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
