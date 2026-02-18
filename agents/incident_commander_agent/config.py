"""
File: config.py
Purpose: Frozen dataclass configuration for the Incident Commander Agent.
Dependencies: Standard library only.
Performance: O(1) attribute access.

All configuration is immutable after construction.
Use from_env() for environment-based overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BlastRadiusConfig:
    """Blast radius calculation parameters.

    Attributes:
        avg_users_per_service: Default estimated users per service.
        revenue_per_minute_per_service: Default USD/min per service.
        customer_facing_services: Services that face customers.
    """
    avg_users_per_service: int = 5000
    revenue_per_minute_per_service: float = 150.0
    customer_facing_services: tuple = (
        "api-gateway", "merchant-portal", "checkout-service",
        "user-service", "notification-service",
    )


@dataclass(frozen=True)
class EscalationConfig:
    """Escalation decision thresholds.

    Attributes:
        low_confidence_threshold: Below this → escalate.
        low_accuracy_threshold: Below this → escalate.
        high_impact_user_threshold: Above this → escalate for P0.
        auto_resolve_min_confidence: Minimum confidence for auto-resolve.
        auto_resolve_min_accuracy: Minimum accuracy for auto-resolve.
        auto_resolve_max_severity: Max severity for auto-resolve.
    """
    low_confidence_threshold: float = 0.5
    low_accuracy_threshold: float = 0.6
    high_impact_user_threshold: int = 10000
    auto_resolve_min_confidence: float = 0.9
    auto_resolve_min_accuracy: float = 0.9
    auto_resolve_max_severity: str = "P2_MEDIUM"


@dataclass(frozen=True)
class RunbookConfig:
    """Runbook generation settings.

    Attributes:
        max_steps: Maximum steps in a runbook.
        default_step_minutes: Default time per step.
        require_validation_checks: Require validation in each step.
    """
    max_steps: int = 12
    default_step_minutes: float = 2.0
    require_validation_checks: bool = True


@dataclass(frozen=True)
class CommunicationConfig:
    """Communication builder settings.

    Attributes:
        default_update_frequency: Minutes between status updates.
        default_channels: Default notification channels.
    """
    default_update_frequency: int = 15
    default_channels: tuple = ("slack-incidents", "pagerduty")


@dataclass(frozen=True)
class FeatureFlags:
    """Feature toggle flags.

    Attributes:
        use_llm: Enable LLM-based remediation generation.
        enable_validation: Run output validation checks.
        fallback_to_deterministic: Fall back if LLM fails.
        enable_rollback_planning: Generate rollback plans.
        enable_communication: Generate communication plans.
        enable_prevention: Generate prevention recommendations.
    """
    use_llm: bool = False
    enable_validation: bool = True
    fallback_to_deterministic: bool = True
    enable_rollback_planning: bool = True
    enable_communication: bool = True
    enable_prevention: bool = True


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
        circuit_failure_threshold: Failures before OPEN.
        circuit_cooldown_seconds: Seconds before HALF_OPEN.
        circuit_success_threshold: Successes to CLOSE.
        cache_ttl_seconds: Cache entry TTL.
        cost_per_1k_input_tokens: USD per 1K input tokens.
        cost_per_1k_output_tokens: USD per 1K output tokens.
    """
    provider: str = "mock"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 2000
    api_key: Optional[str] = None
    timeout_seconds: float = 15.0
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 60.0
    circuit_success_threshold: int = 2
    cache_ttl_seconds: int = 300
    cost_per_1k_input_tokens: float = 0.00059
    cost_per_1k_output_tokens: float = 0.00079


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance budgets.

    Attributes:
        max_pipeline_ms: Max total pipeline latency.
        max_fallback_ms: Max fallback latency.
        max_cost_per_call: Max USD per LLM call.
    """
    max_pipeline_ms: float = 2000.0
    max_fallback_ms: float = 100.0
    max_cost_per_call: float = 0.001


@dataclass
class IncidentCommanderConfig:
    """Master configuration for the Incident Commander Agent.

    Composes all sub-configs with sensible defaults.
    """
    blast_radius: BlastRadiusConfig = None  # type: ignore[assignment]
    escalation: EscalationConfig = None  # type: ignore[assignment]
    runbook: RunbookConfig = None  # type: ignore[assignment]
    communication: CommunicationConfig = None  # type: ignore[assignment]
    features: FeatureFlags = None  # type: ignore[assignment]
    llm: LLMConfig = None  # type: ignore[assignment]
    performance: PerformanceConfig = None  # type: ignore[assignment]

    known_services: tuple = (
        "api-gateway", "auth-service", "payment-service",
        "fraud-service", "notification-service", "database",
        "cache-service", "merchant-portal", "user-service",
        "checkout-service",
    )

    all_failure_types: tuple = (
        "resource_exhaustion", "network_partition", "crash",
        "configuration_error", "deployment_failure",
        "security_breach", "database_failure", "memory_leak",
        "cascading_failure", "unknown",
    )

    def __post_init__(self) -> None:
        """Populate None fields with frozen defaults."""
        if self.blast_radius is None:
            object.__setattr__(self, "blast_radius", BlastRadiusConfig())
        if self.escalation is None:
            object.__setattr__(self, "escalation", EscalationConfig())
        if self.runbook is None:
            object.__setattr__(self, "runbook", RunbookConfig())
        if self.communication is None:
            object.__setattr__(
                self, "communication", CommunicationConfig()
            )
        if self.features is None:
            object.__setattr__(self, "features", FeatureFlags())
        if self.llm is None:
            object.__setattr__(self, "llm", LLMConfig())
        if self.performance is None:
            object.__setattr__(self, "performance", PerformanceConfig())

    @classmethod
    def from_env(cls) -> IncidentCommanderConfig:
        """Create config with environment variable overrides.

        Returns:
            IncidentCommanderConfig with env-based overrides.
        """
        llm_provider = os.environ.get(
            "INCIDENT_COMMANDER_LLM_PROVIDER", "mock"
        )
        api_key = os.environ.get("GROQ_API_KEY", "")
        use_llm = os.environ.get(
            "INCIDENT_COMMANDER_USE_LLM", "false"
        ).lower() == "true"

        return cls(
            llm=LLMConfig(provider=llm_provider, api_key=api_key or None),
            features=FeatureFlags(use_llm=use_llm),
        )
