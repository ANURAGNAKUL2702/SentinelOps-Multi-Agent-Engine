"""Pydantic v2 schemas for the Orchestrator pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    """Overall pipeline execution status."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CircuitBreakerState(str, Enum):
    """Per-agent circuit-breaker state."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AgentStatus(str, Enum):
    """Per-agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    CIRCUIT_OPEN = "circuit_open"


class HealthStatusValue(str, Enum):
    """Agent health status value."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class StageResult(BaseModel):
    """Result of one execution stage."""
    model_config = ConfigDict(frozen=True)

    stage_name: str
    agents: List[str]
    duration: float = Field(ge=0.0, description="Stage duration in seconds")
    status: str  # SUCCESS / FAILED / PARTIAL
    start_time: datetime
    end_time: datetime


class AgentError(BaseModel):
    """Error captured from an agent execution."""
    model_config = ConfigDict(frozen=True)

    agent_name: str
    error_type: str  # TIMEOUT / VALIDATION_ERROR / LLM_ERROR / CIRCUIT_OPEN / UNKNOWN
    error_message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retries_attempted: int = 0


class PipelineTelemetry(BaseModel):
    """Aggregate telemetry for the pipeline run."""
    model_config = ConfigDict(frozen=True)

    total_llm_cost: float = Field(default=0.0, ge=0.0)
    total_llm_calls: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    agent_latencies: Dict[str, float] = Field(default_factory=dict)
    parallel_speedup: float = Field(default=1.0, ge=0.0)
    timeout_violations: int = Field(default=0, ge=0)
    circuit_breaker_trips: int = Field(default=0, ge=0)


class PipelineMetadata(BaseModel):
    """Metadata for a pipeline run."""
    model_config = ConfigDict(frozen=True)

    start_time: datetime
    end_time: datetime
    retries_by_agent: Dict[str, int] = Field(default_factory=dict)
    timeouts_by_agent: Dict[str, int] = Field(default_factory=dict)
    pipeline_version: str = "1.0.0"


class HealthStatus(BaseModel):
    """Agent health status."""
    model_config = ConfigDict(frozen=True)

    agent_name: str
    is_healthy: bool
    status: HealthStatusValue = HealthStatusValue.UNKNOWN
    last_checked: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent outputs container
# ---------------------------------------------------------------------------

class AgentOutputs(BaseModel):
    """Container for outputs from all 7 agents (None if agent was not executed
    or failed)."""
    model_config = ConfigDict(frozen=True)

    log_output: Optional[Any] = None
    metrics_output: Optional[Any] = None
    dependency_output: Optional[Any] = None
    hypothesis_output: Optional[Any] = None
    root_cause_output: Optional[Any] = None
    validation_output: Optional[Any] = None
    incident_response: Optional[Any] = None


# ---------------------------------------------------------------------------
# Top-level pipeline result
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """Complete result of a pipeline execution."""
    model_config = ConfigDict(frozen=True)

    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: PipelineStatus
    execution_time: float = Field(ge=0.0, description="Total runtime in seconds")
    stage_results: Dict[str, StageResult] = Field(default_factory=dict)
    agent_outputs: AgentOutputs = Field(default_factory=AgentOutputs)
    errors: List[AgentError] = Field(default_factory=list)
    telemetry: PipelineTelemetry = Field(default_factory=PipelineTelemetry)
    circuit_breaker_states: Dict[str, CircuitBreakerState] = Field(default_factory=dict)
    metadata: Optional[PipelineMetadata] = None


# ---------------------------------------------------------------------------
# DAG-related schemas
# ---------------------------------------------------------------------------

class AgentNode(BaseModel):
    """A node in the execution DAG."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    agent_instance: Any = None
    timeout: float = Field(default=10.0, gt=0.0)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open and rejects a call."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Circuit breaker OPEN for agent '{agent_name}' — call rejected")


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the execution DAG."""

    def __init__(self, cycle: List[str]) -> None:
        self.cycle = cycle
        super().__init__(f"Cycle detected in DAG: {' → '.join(cycle)}")


class StateMachineError(Exception):
    """Raised on an invalid state transition."""

    def __init__(self, from_state: str, to_state: str) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid transition: {from_state} → {to_state}")
