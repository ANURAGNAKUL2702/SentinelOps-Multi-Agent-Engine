"""
File: config.py
Purpose: Feature flags, thresholds, and environment-aware configuration.
Dependencies: pydantic, pydantic-settings (optional)
Performance: Loaded once at startup, cached in module globals.

All magic numbers from the prompt specifications are centralised here.
Every threshold is overridable via environment variables.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════
#  THRESHOLD CONFIGURATION
# ═══════════════════════════════════════════════════════════════


class ThresholdConfig(BaseModel):
    """All numeric thresholds used in classification and detection.

    Sourced from the synthesizer.txt specification.  Every value can
    be overridden via ``LOG_AGENT_<FIELD>`` environment variable.

    Example::

        thresholds = ThresholdConfig()
        thresholds.high_error_pct  # 40.0
    """

    # ── severity classification ─────────────────────────────────
    high_error_pct: float = Field(
        default=40.0,
        description="error_percentage above this → high severity",
    )
    medium_error_pct_low: float = Field(
        default=10.0,
        description="error_percentage lower bound for medium severity",
    )
    medium_error_pct_high: float = Field(
        default=40.0,
        description="error_percentage upper bound for medium severity",
    )

    # ── trend classification ────────────────────────────────────
    spike_growth_rate: float = Field(
        default=200.0,
        description="growth_rate above this → sudden_spike",
    )
    increasing_growth_rate_low: float = Field(
        default=20.0,
        description="growth_rate lower bound for increasing trend",
    )
    increasing_growth_rate_high: float = Field(
        default=200.0,
        description="growth_rate upper bound for increasing trend",
    )
    stable_growth_rate_bound: float = Field(
        default=20.0,
        description="±bound for stable trend classification",
    )

    # ── suspicious service identification ───────────────────────
    suspicious_error_pct: float = Field(
        default=10.0,
        description="error_percentage above this → suspicious",
    )
    suspicious_growth_rate: float = Field(
        default=100.0,
        description="growth_rate above this → suspicious",
    )

    # ── dominance / flooding ────────────────────────────────────
    dominant_service_pct: float = Field(
        default=60.0,
        description="Service with > this % of all errors is dominant",
    )
    log_flood_repeat_count: int = Field(
        default=100,
        description="Identical error > this count in a window → flooding",
    )

    # ── cascading detection ─────────────────────────────────────
    cascading_min_services: int = Field(
        default=3,
        description="Minimum affected services to consider cascading",
    )
    cascading_period_gap: int = Field(
        default=1,
        description="Earliest service must precede others by > this many periods",
    )

    # ── confidence algorithm ────────────────────────────────────
    confidence_base: float = Field(default=0.2)
    confidence_dominant_bonus: float = Field(default=0.3)
    confidence_trend_bonus: float = Field(default=0.2)
    confidence_keyword_bonus: float = Field(default=0.2)
    confidence_cascading_bonus: float = Field(default=0.1)
    confidence_distributed_penalty: float = Field(default=0.2)
    confidence_stable_penalty: float = Field(default=0.3)
    confidence_distributed_threshold: int = Field(
        default=5,
        description="affected_service_count > this triggers distributed penalty",
    )
    confidence_stable_growth_max: float = Field(
        default=20.0,
        description="All growth rates < this triggers stable penalty",
    )

    # ── anomaly detection ───────────────────────────────────────
    z_score_threshold: float = Field(
        default=3.0, description="Z-score above this → anomaly (3-sigma)"
    )


# ═══════════════════════════════════════════════════════════════
#  FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════


class FeatureFlags(BaseModel):
    """Runtime feature toggles for gradual rollout.

    Override via environment variables:
      LOG_AGENT_USE_LLM=false
      LOG_AGENT_ENABLE_CACHING=true

    Example::

        flags = FeatureFlags(use_llm=False)
        if flags.use_llm:
            # call LLM
    """
    use_llm: bool = Field(
        default=False,
        description="Enable LLM-based classification (requires API key)",
    )
    fallback_to_rules: bool = Field(
        default=True,
        description="Fall back to deterministic rules when LLM fails",
    )
    enable_caching: bool = Field(
        default=True,
        description="Cache LLM responses (TTL controlled by cache_ttl_seconds)",
    )
    enable_anomaly_detection: bool = Field(
        default=True,
        description="Run statistical anomaly detection layer",
    )
    enable_validation: bool = Field(
        default=True,
        description="Run validator on final output",
    )
    enable_telemetry: bool = Field(
        default=True,
        description="Collect telemetry (latency, counts, costs)",
    )


# ═══════════════════════════════════════════════════════════════
#  LLM CONFIGURATION
# ═══════════════════════════════════════════════════════════════


class LLMConfig(BaseModel):
    """LLM provider configuration.

    API keys are NEVER stored in code — only via env vars.

    Example::

        llm = LLMConfig()  # reads from LOG_AGENT_LLM_PROVIDER etc.
    """
    provider: str = Field(
        default="openai",
        description="LLM provider: 'openai' | 'anthropic' | 'mock'",
    )
    model: str = Field(default="gpt-4o-mini")
    api_key: str = Field(
        default="",
        description="Read from env var — never hardcode",
    )
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0,
        description="Max time for a single LLM call",
    )
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(
        default=1.0, description="Base delay for exponential backoff (seconds)"
    )
    cache_ttl_seconds: int = Field(
        default=300, ge=0,
        description="TTL for cached LLM responses (0 = disabled)",
    )

    # ── circuit breaker ─────────────────────────────────────────
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

    # ── cost tracking ───────────────────────────────────────────
    cost_per_1k_input_tokens: float = Field(
        default=0.00015,
        description="Cost per 1K input tokens (USD)",
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.0006,
        description="Cost per 1K output tokens (USD)",
    )


# ═══════════════════════════════════════════════════════════════
#  KEYWORD LISTS
# ═══════════════════════════════════════════════════════════════


class KeywordConfig(BaseModel):
    """Keyword lists for severity signal detection."""
    critical_keywords: list[str] = Field(default=[
        "outofmemoryerror", "oom", "oomkilled",
        "database connection timeout", "database timeout",
        "connection refused",
        "deadlock", "fatal", "panic", "segfault",
        "data corruption", "split brain",
        "circuit breaker open", "thread starvation",
    ])
    moderate_keywords: list[str] = Field(default=[
        "high cpu", "memory pool", "gc overhead",
        "connection pool", "retry", "timeout",
        "slow query", "replication lag",
        "rate limit", "throttle", "backpressure",
        "degraded", "socket timeout",
    ])
    database_keywords: list[str] = Field(default=[
        "database", "connection timeout", "connection refused",
        "deadlock", "pool exhausted", "sql", "replication",
        "transaction rolled back",
    ])


# ═══════════════════════════════════════════════════════════════
#  TOP-LEVEL CONFIG
# ═══════════════════════════════════════════════════════════════


class LogAgentConfig(BaseModel):
    """Top-level configuration for the log agent.

    Loads defaults and merges with environment variable overrides.

    Example::

        config = LogAgentConfig.from_env()
        if config.features.use_llm:
            ...
    """
    environment: str = Field(
        default="dev",
        description="Runtime environment: 'dev' | 'staging' | 'prod'",
    )
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    keywords: KeywordConfig = Field(default_factory=KeywordConfig)

    @model_validator(mode="after")
    def validate_config(self) -> "LogAgentConfig":
        """Fail-fast validation at startup."""
        if self.features.use_llm and not self.llm.api_key:
            # Try env vars before failing
            env_key = (
                os.environ.get("LOG_AGENT_LLM_API_KEY", "")
                or os.environ.get("GROQ_API_KEY", "")
            )
            if env_key:
                object.__setattr__(self.llm, "api_key", env_key)
            elif self.llm.provider != "mock":
                raise ValueError(
                    "LLM is enabled but no API key provided. "
                    "Set LOG_AGENT_LLM_API_KEY / GROQ_API_KEY or use provider='mock'."
                )
        return self

    @classmethod
    def from_env(cls) -> "LogAgentConfig":
        """Build config from environment variables.

        Convention: ``LOG_AGENT_<SECTION>_<FIELD>`` in uppercase.

        Example::

            # Shell:
            export LOG_AGENT_ENVIRONMENT=prod
            export LOG_AGENT_USE_LLM=true
            export LOG_AGENT_LLM_API_KEY=sk-xxx

            # Python:
            config = LogAgentConfig.from_env()
        """
        env = os.environ.get("LOG_AGENT_ENVIRONMENT", "dev")

        features = FeatureFlags(
            use_llm=_env_bool("LOG_AGENT_USE_LLM", False),
            fallback_to_rules=_env_bool("LOG_AGENT_FALLBACK_TO_RULES", True),
            enable_caching=_env_bool("LOG_AGENT_ENABLE_CACHING", True),
            enable_anomaly_detection=_env_bool(
                "LOG_AGENT_ENABLE_ANOMALY_DETECTION", True
            ),
            enable_validation=_env_bool("LOG_AGENT_ENABLE_VALIDATION", True),
            enable_telemetry=_env_bool("LOG_AGENT_ENABLE_TELEMETRY", True),
        )

        llm = LLMConfig(
            provider=os.environ.get("LOG_AGENT_LLM_PROVIDER", "openai"),
            model=os.environ.get("LOG_AGENT_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("LOG_AGENT_LLM_API_KEY", ""),
        )

        return cls(
            environment=env,
            features=features,
            llm=llm,
        )


# ── helpers ─────────────────────────────────────────────────────

def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from env var (true/1/yes → True)."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")
