"""
File: config.py
Purpose: Centralized configuration for the metrics agent pipeline.
Dependencies: pydantic >=2.0
Performance: Config validation <1ms

All thresholds, feature flags, and LLM settings in one place.
No secrets in code — API keys from environment variables only.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════
#  METRIC THRESHOLD CONFIGURATION
# ═══════════════════════════════════════════════════════════════


class ThresholdConfig(BaseModel):
    """Operational thresholds for metric breach detection.

    Example::

        ThresholdConfig(cpu_percent=80.0, memory_percent=90.0)
    """
    cpu_percent: float = Field(
        default=80.0, description="CPU usage threshold (%)"
    )
    memory_percent: float = Field(
        default=90.0, description="Memory usage threshold (%)"
    )
    disk_usage_percent: float = Field(
        default=85.0, description="Disk usage threshold (%)"
    )
    disk_io_read_mb_per_sec: float = Field(
        default=400.0, description="Disk read I/O threshold (MB/s)"
    )
    disk_io_write_mb_per_sec: float = Field(
        default=400.0, description="Disk write I/O threshold (MB/s)"
    )
    network_in_mbps: float = Field(
        default=500.0, description="Inbound network threshold (Mbps)"
    )
    network_out_mbps: float = Field(
        default=500.0, description="Outbound network threshold (Mbps)"
    )
    api_latency_p50_ms: float = Field(
        default=500.0, description="P50 latency threshold (ms)"
    )
    api_latency_p95_ms: float = Field(
        default=1000.0, description="P95 latency threshold (ms)"
    )
    api_latency_p99_ms: float = Field(
        default=1000.0, description="P99 latency threshold (ms)"
    )
    error_rate_percent: float = Field(
        default=5.0, description="Error rate threshold (%)"
    )
    request_rate_per_sec: float = Field(
        default=0.0,
        description="Min request rate threshold (0=disabled)",
    )

    def get_threshold(self, metric_name: str) -> Optional[float]:
        """Get the threshold for a metric name.

        Args:
            metric_name: Name of the metric (e.g. 'cpu_percent').

        Returns:
            Threshold value or None if no threshold configured.
        """
        return getattr(self, metric_name, None)


# ═══════════════════════════════════════════════════════════════
#  STATISTICAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════


class StatisticalConfig(BaseModel):
    """Statistical thresholds for anomaly detection.

    Example::

        StatisticalConfig(zscore_threshold=3.0, correlation_threshold=0.7)
    """
    zscore_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="Z-score threshold for anomaly detection (3-sigma rule)",
    )
    correlation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum |r| for significant Pearson correlation",
    )
    spike_threshold_percent: float = Field(
        default=50.0,
        description="Growth rate % above which trend is 'sudden_spike'",
    )
    increasing_threshold_percent: float = Field(
        default=20.0,
        description="Growth rate % above which trend is 'increasing'",
    )
    sustained_sigma: float = Field(
        default=2.0,
        description="Sigma threshold for sustained anomaly detection",
    )


# ═══════════════════════════════════════════════════════════════
#  FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════


class FeatureFlags(BaseModel):
    """Feature toggles for the metrics agent pipeline.

    Example::

        FeatureFlags(use_llm=False)  # deterministic-only mode
    """
    use_llm: bool = Field(
        default=False,
        description="Enable LLM classification (20% path)",
    )
    fallback_to_rules: bool = Field(
        default=True,
        description="Fallback to rule engine on LLM failure",
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching for LLM calls",
    )
    enable_validation: bool = Field(
        default=True,
        description="Run 23 validation checks on output",
    )
    enable_anomaly_detection: bool = Field(
        default=True,
        description="Run anomaly type classification",
    )
    enable_correlation: bool = Field(
        default=True,
        description="Run Pearson correlation detection",
    )


# ═══════════════════════════════════════════════════════════════
#  LLM CONFIGURATION
# ═══════════════════════════════════════════════════════════════


class LLMConfig(BaseModel):
    """LLM provider configuration.

    API key is NEVER stored in code — read from environment.

    Example::

        LLMConfig(provider='groq', model='llama-3.3-70b-versatile')
    """
    provider: str = Field(
        default="mock",
        description="LLM provider: 'groq' | 'mock'",
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model ID for the LLM provider",
    )
    api_key: str = Field(
        default="",
        description="API key (prefer GROQ_API_KEY env var)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic)",
    )
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Max output tokens",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Max wait time per LLM call (seconds)",
    )

    # Circuit breaker
    circuit_failure_threshold: int = Field(
        default=3,
        description="Consecutive failures to open circuit",
    )
    circuit_cooldown_seconds: float = Field(
        default=60.0,
        description="Seconds before half-open attempt",
    )
    circuit_success_threshold: int = Field(
        default=2,
        description="Consecutive successes to close circuit",
    )

    # Retry
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Max LLM call attempts",
    )
    retry_base_delay: float = Field(
        default=1.0,
        description="Base delay for exponential backoff (seconds)",
    )

    # Cache
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache TTL in seconds",
    )

    # Cost tracking
    cost_per_1k_input_tokens: float = Field(
        default=0.00059,
        description="Cost per 1K input tokens (USD)",
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.00079,
        description="Cost per 1K output tokens (USD)",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: str) -> str:
        """Read API key from environment if not provided."""
        if v:
            return v
        return os.environ.get("GROQ_API_KEY", "") or os.environ.get(
            "METRICS_AGENT_LLM_API_KEY", ""
        )


# ═══════════════════════════════════════════════════════════════
#  PERFORMANCE LIMITS
# ═══════════════════════════════════════════════════════════════


class PerformanceConfig(BaseModel):
    """Performance guard rails."""
    max_metrics_per_request: int = Field(
        default=1000,
        description="Max number of metrics per analysis request",
    )
    max_timeseries_length: int = Field(
        default=100,
        description="Max data points per timeseries",
    )


# ═══════════════════════════════════════════════════════════════
#  MASTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════


class MetricsAgentConfig(BaseModel):
    """Master configuration for the Metrics Agent.

    Groups all sub-configurations into a single entry point.

    Example::

        config = MetricsAgentConfig()
        config = MetricsAgentConfig(
            features=FeatureFlags(use_llm=True),
            llm=LLMConfig(provider='groq'),
        )
        config = MetricsAgentConfig.from_env()
    """
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @classmethod
    def from_env(cls) -> "MetricsAgentConfig":
        """Build configuration from environment variables.

        Reads:
            GROQ_API_KEY: Groq API key
            METRICS_AGENT_USE_LLM: '1' to enable LLM
            METRICS_AGENT_LLM_PROVIDER: 'groq' or 'mock'
            METRICS_AGENT_LLM_MODEL: model ID

        Returns:
            Configured MetricsAgentConfig instance.
        """
        use_llm = os.environ.get("METRICS_AGENT_USE_LLM", "0") == "1"
        provider = os.environ.get("METRICS_AGENT_LLM_PROVIDER", "mock")
        model = os.environ.get(
            "METRICS_AGENT_LLM_MODEL", "llama-3.3-70b-versatile"
        )

        return cls(
            features=FeatureFlags(use_llm=use_llm),
            llm=LLMConfig(provider=provider, model=model),
        )
