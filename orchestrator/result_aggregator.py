"""Collect and merge agent outputs into a single pipeline result."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    AgentError,
    AgentOutputs,
    PipelineTelemetry,
    StageResult,
)
from .telemetry import get_logger

_logger = get_logger(__name__)

# Map from agent name → AgentOutputs field name
_AGENT_FIELD_MAP: Dict[str, str] = {
    "log_agent": "log_output",
    "metrics_agent": "metrics_output",
    "dependency_agent": "dependency_output",
    "hypothesis_agent": "hypothesis_output",
    "root_cause_agent": "root_cause_output",
    "validation_agent": "validation_output",
    "incident_commander_agent": "incident_response",
}


class ResultAggregator:
    """Collect per-agent outputs, errors, and stage timings, then merge."""

    def __init__(self) -> None:
        self._outputs: Dict[str, Tuple[Any, float]] = {}  # name → (output, duration)
        self._errors: List[AgentError] = []
        self._stages: Dict[str, StageResult] = {}
        self._llm_costs: Dict[str, float] = {}
        self._llm_calls: Dict[str, int] = {}

    # ---- mutation ---------------------------------------------------------

    def add_agent_output(
        self,
        agent_name: str,
        output: Any,
        duration: float,
    ) -> None:
        """Store an agent's output (overwrites previous if retried).

        Args:
            agent_name: Agent identifier.
            output: Output object from the agent.
            duration: Agent execution time in seconds.
        """
        self._outputs[agent_name] = (output, duration)

    def add_agent_error(
        self,
        agent_name: str,
        error: Exception,
        error_type: str,
        retries: int = 0,
    ) -> None:
        """Record an agent error.

        Args:
            agent_name: Agent identifier.
            error: Exception that occurred.
            error_type: Category string (TIMEOUT, VALIDATION_ERROR, etc.).
            retries: How many retries were attempted.
        """
        self._errors.append(
            AgentError(
                agent_name=agent_name,
                error_type=error_type,
                error_message=str(error),
                timestamp=datetime.now(timezone.utc),
                retries_attempted=retries,
            )
        )

    def add_stage_result(self, stage: StageResult) -> None:
        """Store a completed stage result."""
        self._stages[stage.stage_name] = stage

    def record_llm_cost(self, agent_name: str, cost: float, calls: int = 1) -> None:
        """Accumulate LLM cost and call count for an agent."""
        self._llm_costs[agent_name] = self._llm_costs.get(agent_name, 0.0) + cost
        self._llm_calls[agent_name] = self._llm_calls.get(agent_name, 0) + calls

    # ---- queries ----------------------------------------------------------

    def aggregate(self) -> AgentOutputs:
        """Build an :class:`AgentOutputs` from collected outputs.

        Returns:
            Frozen :class:`AgentOutputs` with ``None`` for missing agents.
        """
        fields: Dict[str, Any] = {}
        for agent_name, field_name in _AGENT_FIELD_MAP.items():
            entry = self._outputs.get(agent_name)
            fields[field_name] = entry[0] if entry is not None else None
        return AgentOutputs(**fields)

    def get_stage_results(self) -> Dict[str, StageResult]:
        """Return all stage results."""
        return dict(self._stages)

    def get_errors(self) -> List[AgentError]:
        """Return all recorded errors."""
        return list(self._errors)

    def calculate_telemetry(self) -> PipelineTelemetry:
        """Compute aggregate telemetry.

        Returns:
            :class:`PipelineTelemetry` with merged metrics.
        """
        latencies: Dict[str, float] = {}
        for name, (_, dur) in self._outputs.items():
            latencies[name] = dur

        # Parallel speedup heuristic: compare stage-1 wallclock to sum of
        # individual times for agents in stage-1.
        stage_1 = self._stages.get("stage_1")
        if stage_1 is not None and len(stage_1.agents) > 1:
            sum_individual = sum(latencies.get(a, 0.0) for a in stage_1.agents)
            parallel_speedup = (sum_individual / stage_1.duration) if stage_1.duration > 0 else 1.0
        else:
            parallel_speedup = 1.0

        total_cost = sum(self._llm_costs.values())
        total_calls = sum(self._llm_calls.values())

        return PipelineTelemetry(
            total_llm_cost=total_cost,
            total_llm_calls=total_calls,
            total_tokens=0,
            agent_latencies=latencies,
            parallel_speedup=parallel_speedup,
            timeout_violations=sum(
                1 for e in self._errors if e.error_type == "TIMEOUT"
            ),
            circuit_breaker_trips=sum(
                1 for e in self._errors if e.error_type == "CIRCUIT_OPEN"
            ),
        )

    # ---- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Clear all collected results."""
        self._outputs.clear()
        self._errors.clear()
        self._stages.clear()
        self._llm_costs.clear()
        self._llm_calls.clear()
