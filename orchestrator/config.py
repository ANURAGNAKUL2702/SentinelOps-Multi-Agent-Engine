"""Frozen-dataclass configuration for the Orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OrchestratorConfig:
    """Master configuration for the orchestrator pipeline.

    All values carry sensible defaults.  Override via constructor kwargs.
    """

    # ---- Per-agent timeouts (seconds) ------------------------------------
    log_agent_timeout: float = 5.0
    metrics_agent_timeout: float = 5.0
    dependency_agent_timeout: float = 5.0
    hypothesis_agent_timeout: float = 10.0
    root_cause_agent_timeout: float = 10.0
    validation_agent_timeout: float = 3.0
    incident_commander_timeout: float = 8.0

    # ---- Total pipeline timeout ------------------------------------------
    pipeline_timeout: float = 60.0

    # ---- Retry policy ----------------------------------------------------
    max_retries: int = 2
    retry_backoff_base: float = 1.0
    retry_backoff_multiplier: float = 2.0
    retry_jitter: float = 0.1  # ±10 %

    # ---- Circuit breaker -------------------------------------------------
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_timeout: float = 30.0
    circuit_breaker_half_open_max_calls: int = 1

    # ---- Parallelism -----------------------------------------------------
    max_parallel_agents: int = 3
    enable_parallel_execution: bool = True

    # ---- Error handling --------------------------------------------------
    fail_fast: bool = False
    allow_partial_results: bool = True

    # ---- Telemetry -------------------------------------------------------
    enable_prometheus_metrics: bool = True
    enable_detailed_tracing: bool = True

    # ---- Health checks ---------------------------------------------------
    health_check_interval: float = 5.0
    health_check_timeout: float = 2.0
    enable_health_checks: bool = True

    # ---- Agent timeout map (derived) -------------------------------------
    def agent_timeout(self, agent_name: str) -> float:
        """Return the configured timeout for *agent_name*."""
        _map = {
            "log_agent": self.log_agent_timeout,
            "metrics_agent": self.metrics_agent_timeout,
            "dependency_agent": self.dependency_agent_timeout,
            "hypothesis_agent": self.hypothesis_agent_timeout,
            "root_cause_agent": self.root_cause_agent_timeout,
            "validation_agent": self.validation_agent_timeout,
            "incident_commander_agent": self.incident_commander_timeout,
        }
        return _map.get(agent_name, 10.0)

    def __post_init__(self) -> None:
        """Validate invariants at construction time."""
        if self.pipeline_timeout <= 0:
            raise ValueError("pipeline_timeout must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_backoff_base <= 0:
            raise ValueError("retry_backoff_base must be > 0")
        if self.circuit_breaker_failure_threshold <= 0:
            raise ValueError("circuit_breaker_failure_threshold must be > 0")
        if self.circuit_breaker_recovery_timeout < 0:
            raise ValueError("circuit_breaker_recovery_timeout must be >= 0")

        _timeouts = [
            self.log_agent_timeout,
            self.metrics_agent_timeout,
            self.dependency_agent_timeout,
            self.hypothesis_agent_timeout,
            self.root_cause_agent_timeout,
            self.validation_agent_timeout,
            self.incident_commander_timeout,
        ]
        for t in _timeouts:
            if t <= 0:
                raise ValueError("All per-agent timeouts must be > 0")
