"""Configuration management — load, validate, merge YAML + env vars.

Uses Pydantic v2 for schema validation and PyYAML for file parsing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Pydantic settings models ───────────────────────────────────────


class SystemSettings(BaseModel):
    """Top-level system settings."""

    model_config = ConfigDict(frozen=True)

    project_name: str = "Autonomous War-Room Simulator"
    version: str = "1.0.0"
    log_level: str = "INFO"
    correlation_id_header: str = "X-Correlation-ID"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got '{v}'")
        return upper


class SimulationSettings(BaseModel):
    """Simulation engine settings."""

    model_config = ConfigDict(frozen=True)

    default_scenario: str = "database_timeout"
    available_scenarios: List[str] = Field(
        default_factory=lambda: [
            "memory_leak",
            "cpu_spike",
            "database_timeout",
            "network_latency",
        ],
    )
    duration_minutes: int = 30
    metrics_interval_seconds: int = 60
    log_interval_seconds: int = 30
    random_seed: Optional[int] = None


class LogGenerationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    base_rate: int = 100
    error_spike_multiplier: int = 5


class MetricsCollectionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    interval_seconds: int = 10
    retention_points: int = 100


class DependencyGraphSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_depth: int = 5
    include_external_services: bool = True


class ObservabilitySettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    log_generation: LogGenerationSettings = Field(default_factory=LogGenerationSettings)
    metrics_collection: MetricsCollectionSettings = Field(
        default_factory=MetricsCollectionSettings,
    )
    dependency_graph: DependencyGraphSettings = Field(
        default_factory=DependencyGraphSettings,
    )


class AgentConfig(BaseModel):
    """Per-agent configuration block."""

    model_config = ConfigDict(frozen=True, extra="allow")

    timeout_seconds: float = 5.0
    enable_fallback: bool = True


class AgentSettings(BaseModel):
    """Settings for all 7 agents."""

    model_config = ConfigDict(frozen=True)

    log_agent: AgentConfig = Field(default_factory=AgentConfig)
    metrics_agent: AgentConfig = Field(default_factory=AgentConfig)
    dependency_agent: AgentConfig = Field(default_factory=AgentConfig)
    hypothesis_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(timeout_seconds=10.0),
    )
    root_cause_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(timeout_seconds=10.0),
    )
    validation_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(timeout_seconds=8.0),
    )
    incident_commander_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(timeout_seconds=8.0),
    )


class CircuitBreakerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    failure_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


class RetryPolicySettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 10.0


class OrchestratorSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    pipeline_timeout_seconds: float = 60.0
    enable_parallel_execution: bool = True
    fail_fast: bool = False
    max_retries: int = 2
    circuit_breaker: CircuitBreakerSettings = Field(
        default_factory=CircuitBreakerSettings,
    )
    retry_policy: RetryPolicySettings = Field(default_factory=RetryPolicySettings)


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    url: str = "sqlite:///incidents.db"
    enable: bool = True
    retention_days: int = 90


class ReportingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    default_formats: List[str] = Field(default_factory=lambda: ["html", "json"])
    output_directory: str = "reports"
    include_visualizations: bool = True
    include_cost_breakdown: bool = True
    enable_ai_insights: bool = False
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    templates_directory: str = "reporting/templates"
    chart_theme: str = "light"


class APISettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class GroqSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_key_env_var: str = "GROQ_API_KEY"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 2000


class CostTrackingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_cost_per_1k_tokens: float = 0.00005
    output_cost_per_1k_tokens: float = 0.00008


class LLMSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str = "groq"
    groq: GroqSettings = Field(default_factory=GroqSettings)
    cost_tracking: CostTrackingSettings = Field(default_factory=CostTrackingSettings)


# ── Top-level config ───────────────────────────────────────────────


class SystemConfig(BaseModel):
    """Complete system configuration — all phases."""

    model_config = ConfigDict(frozen=True)

    system: SystemSettings = Field(default_factory=SystemSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings,
    )
    agents: AgentSettings = Field(default_factory=AgentSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
    api: APISettings = Field(default_factory=APISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


# ── ConfigManager ──────────────────────────────────────────────────

# Environment variable → config path mapping.
_ENV_MAP: Dict[str, str] = {
    "WARROOM_LOG_LEVEL": "system.log_level",
    "WARROOM_DATABASE_URL": "reporting.database.url",
    "WARROOM_API_ENABLED": "api.enabled",
    "WARROOM_API_PORT": "api.port",
    "WARROOM_OUTPUT_DIR": "reporting.output_directory",
}


class ConfigManager:
    """Load, validate, and merge configuration from YAML + env vars."""

    @staticmethod
    def load(config_path: str = "config.yaml") -> SystemConfig:
        """Load config from *config_path*, validate, merge env vars.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Validated :class:`SystemConfig`.

        Raises:
            FileNotFoundError: If the config file does not exist and
                no defaults are acceptable.
            yaml.YAMLError: If the file contains invalid YAML.
            pydantic.ValidationError: If the data fails validation.
        """
        path = Path(config_path)
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        else:
            raw = {}

        config = SystemConfig.model_validate(raw)
        config = ConfigManager.merge_env_vars(config)
        return config

    @staticmethod
    def validate(config: SystemConfig) -> List[str]:
        """Return a list of human-readable validation issues.

        An empty list means the config is valid.
        """
        issues: List[str] = []

        if config.orchestrator.pipeline_timeout_seconds <= 0:
            issues.append("orchestrator.pipeline_timeout_seconds must be > 0")

        for sc in config.simulation.available_scenarios:
            # basic sanity — no whitespace-only names
            if not sc.strip():
                issues.append("simulation.available_scenarios contains an empty name")

        if config.reporting.database.enable and not config.reporting.database.url:
            issues.append("reporting.database.url is required when database is enabled")

        if config.api.enabled and config.api.port <= 0:
            issues.append("api.port must be a positive integer when API is enabled")

        return issues

    @staticmethod
    def merge_env_vars(config: SystemConfig) -> SystemConfig:
        """Override config values from environment variables.

        Returns a **new** frozen :class:`SystemConfig` with overrides
        applied.
        """
        overrides: Dict[str, Any] = {}

        for env_key, config_path in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value is None:
                continue

            parts = config_path.split(".")
            d = overrides
            for p in parts[:-1]:
                d = d.setdefault(p, {})

            # Coerce types
            if config_path.endswith(".port"):
                d[parts[-1]] = int(value)
            elif config_path.endswith(".enabled"):
                d[parts[-1]] = value.lower() in ("1", "true", "yes")
            else:
                d[parts[-1]] = value

        if not overrides:
            return config

        # Deep-merge overrides into the existing dump
        base = config.model_dump()
        _deep_merge(base, overrides)
        return SystemConfig.model_validate(base)

    @staticmethod
    def get_default_config() -> SystemConfig:
        """Return a :class:`SystemConfig` with all defaults."""
        return SystemConfig()

    @staticmethod
    def save(config: SystemConfig, path: str) -> None:
        """Dump *config* to a YAML file at *path*."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            yaml.dump(
                config.model_dump(),
                fh,
                default_flow_style=False,
                sort_keys=False,
            )


# ── helpers ────────────────────────────────────────────────────────


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* (mutating)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
