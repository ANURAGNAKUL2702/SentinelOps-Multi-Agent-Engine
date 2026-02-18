"""Main Orchestrator — coordinates all 7 agents in a pipeline DAG."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .circuit_breaker import CircuitBreaker
from .config import OrchestratorConfig
from .correlation_tracker import CorrelationTracker
from .dag import ExecutionDAG
from .error_handler import ErrorHandler
from .execution_engine import ExecutionEngine
from .health_checker import HealthChecker
from .metrics_collector import MetricsCollector
from .result_aggregator import ResultAggregator
from .retry_policy import RetryPolicy
from .schema import (
    AgentError,
    AgentOutputs,
    CircuitBreakerState,
    PipelineMetadata,
    PipelineResult,
    PipelineStatus,
    PipelineTelemetry,
)
from .state_machine import StateMachine
from .timeout_manager import TimeoutManager
from .telemetry import get_logger

_logger = get_logger(__name__)

# Agent names used throughout the pipeline.
_AGENT_NAMES = [
    "log_agent",
    "metrics_agent",
    "dependency_agent",
    "hypothesis_agent",
    "root_cause_agent",
    "validation_agent",
    "incident_commander_agent",
]


class Orchestrator:
    """Coordinate all 7 agents in a dependency-ordered DAG.

    Responsibilities:
      * Build execution DAG with agent dependencies.
      * Execute agents in parallel where possible (Stage 1).
      * Enforce per-agent and pipeline-level timeouts.
      * Handle failures with circuit breakers and retries.
      * Propagate correlation IDs.
      * Aggregate results and telemetry.
      * Export Prometheus metrics.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        *,
        agents: Dict[str, Any] | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()

        # Sub-components
        self.timeout_manager = TimeoutManager()
        self.retry_policy = RetryPolicy(self.config)
        self.circuit_breakers = self._init_circuit_breakers()
        self.state_machine = StateMachine()
        self.correlation_tracker = CorrelationTracker()
        self.result_aggregator = ResultAggregator()
        self.error_handler = ErrorHandler(self.config)
        self.health_checker = HealthChecker(self.config)
        self.metrics_collector = MetricsCollector(self.config)

        # Agents can be injected (for testing) or left for later binding
        self._agents: Dict[str, Any] = agents or {}

        # Build DAG
        self.dag = self._build_dag()

        # Execution engine
        self.execution_engine = ExecutionEngine(
            config=self.config,
            timeout_manager=self.timeout_manager,
            retry_policy=self.retry_policy,
            circuit_breakers=self.circuit_breakers,
            error_handler=self.error_handler,
            result_aggregator=self.result_aggregator,
            state_machine=self.state_machine,
            metrics_collector=self.metrics_collector,
        )

        # Validate configuration
        self._validate_config()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_dag(self) -> ExecutionDAG:
        """Build the 5-stage execution DAG."""
        dag = ExecutionDAG()

        for name in _AGENT_NAMES:
            agent = self._agents.get(name)
            timeout = self.config.agent_timeout(name)
            dag.add_node(name, agent_instance=agent, timeout=timeout)

        # Stage 1 → Stage 2
        dag.add_edge("log_agent", "hypothesis_agent")
        dag.add_edge("metrics_agent", "hypothesis_agent")
        dag.add_edge("dependency_agent", "hypothesis_agent")
        # Stage 2 → Stage 3
        dag.add_edge("hypothesis_agent", "root_cause_agent")
        # Stage 3 → Stage 4
        dag.add_edge("root_cause_agent", "validation_agent")
        # Stage 4 → Stage 5
        dag.add_edge("validation_agent", "incident_commander_agent")

        dag.validate()
        return dag

    def _init_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        return {
            name: CircuitBreaker(
                agent_name=name,
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery_timeout,
                half_open_max_calls=self.config.circuit_breaker_half_open_max_calls,
            )
            for name in _AGENT_NAMES
        }

    def _validate_config(self) -> None:
        """Warn if pipeline timeout is tight."""
        critical_path = (
            max(
                self.config.log_agent_timeout,
                self.config.metrics_agent_timeout,
                self.config.dependency_agent_timeout,
            )
            + self.config.hypothesis_agent_timeout
            + self.config.root_cause_agent_timeout
            + self.config.validation_agent_timeout
            + self.config.incident_commander_timeout
        )
        if self.config.pipeline_timeout < critical_path:
            _logger.warning(
                f"Pipeline timeout ({self.config.pipeline_timeout}s) < "
                f"critical-path sum ({critical_path}s). Pipeline may timeout.",
            )

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        scenario: Any = None,
        observability_data: Any = None,
        correlation_id: str | None = None,
        *,
        ground_truth: Any = None,
        extra_inputs: Dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute the full 5-stage agent pipeline.

        Args:
            scenario: Simulation scenario dict.
            observability_data: Observability data dict.
            correlation_id: Optional; auto-generated if absent.
            ground_truth: Ground-truth data for the validation agent.
            extra_inputs: Additional data merged into pipeline inputs.

        Returns:
            :class:`PipelineResult` with all outputs, telemetry, and errors.
        """
        # Correlation ID
        if correlation_id is None:
            correlation_id = self.correlation_tracker.generate_correlation_id()
        self.correlation_tracker.set_correlation_id(correlation_id)

        start_time_dt = datetime.now(timezone.utc)
        t0 = time.perf_counter()

        _logger.info(
            "Starting pipeline execution",
            extra={"correlation_id": correlation_id},
        )

        # Reset per-run state
        self.result_aggregator.reset()
        self.error_handler.reset()
        self.state_machine.reset()
        self.metrics_collector.reset()
        self.timeout_manager.reset_stats()
        self.retry_policy.reset_stats()

        # Transition PENDING → RUNNING.  The state machine uses an
        # internal "RUNNING" label that is not a PipelineStatus member,
        # so we set it directly.
        self.state_machine._state = "RUNNING"

        # Build initial inputs
        initial_inputs: Dict[str, Any] = {
            "scenario": scenario,
            "observability_data": observability_data,
            "ground_truth": ground_truth,
        }
        if extra_inputs:
            initial_inputs.update(extra_inputs)

        try:
            run_result = await asyncio.wait_for(
                self.execution_engine.run_pipeline(
                    self.dag, initial_inputs, correlation_id,
                ),
                timeout=self.config.pipeline_timeout,
            )
            duration = time.perf_counter() - t0
            end_time_dt = datetime.now(timezone.utc)

            # Determine status
            errors = run_result.errors
            outputs = run_result.agent_outputs
            if not errors:
                final_status = PipelineStatus.SUCCESS
            elif outputs.incident_response is not None:
                final_status = PipelineStatus.PARTIAL_SUCCESS
            else:
                final_status = PipelineStatus.FAILED

            self.state_machine._state = final_status.value

            telemetry = self.result_aggregator.calculate_telemetry()
            meta = PipelineMetadata(
                start_time=start_time_dt,
                end_time=end_time_dt,
                retries_by_agent=self.retry_policy.get_retry_stats(),
                timeouts_by_agent=self.timeout_manager.get_timeout_stats(),
            )

            pipeline_result = PipelineResult(
                correlation_id=correlation_id,
                status=final_status,
                execution_time=duration,
                stage_results=run_result.stage_results,
                agent_outputs=outputs,
                errors=errors,
                telemetry=telemetry,
                circuit_breaker_states={
                    name: cb.get_state()
                    for name, cb in self.circuit_breakers.items()
                },
                metadata=meta,
            )

            self.metrics_collector.record_pipeline_result(final_status, duration)
            _logger.info(
                f"Pipeline completed: {final_status.value}",
                extra={
                    "correlation_id": correlation_id,
                    "execution_time": duration,
                    "errors": len(errors),
                },
            )
            return pipeline_result

        except asyncio.TimeoutError:
            duration = time.perf_counter() - t0
            end_time_dt = datetime.now(timezone.utc)
            _logger.error(
                f"Pipeline timeout exceeded ({self.config.pipeline_timeout}s)",
                extra={"correlation_id": correlation_id},
            )
            self.state_machine._state = PipelineStatus.TIMEOUT.value
            self.metrics_collector.record_pipeline_result(PipelineStatus.TIMEOUT, duration)
            return self._build_timeout_result(
                correlation_id, duration, start_time_dt, end_time_dt,
            )

        except Exception as exc:
            duration = time.perf_counter() - t0
            end_time_dt = datetime.now(timezone.utc)
            _logger.error(
                f"Pipeline failed: {exc}",
                extra={"correlation_id": correlation_id},
                exc_info=True,
            )
            self.state_machine._state = PipelineStatus.FAILED.value
            self.metrics_collector.record_pipeline_result(PipelineStatus.FAILED, duration)
            return self._build_error_result(
                correlation_id, exc, duration, start_time_dt, end_time_dt,
            )

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_timeout_result(
        self,
        cid: str,
        duration: float,
        start: datetime,
        end: datetime,
    ) -> PipelineResult:
        return PipelineResult(
            correlation_id=cid,
            status=PipelineStatus.TIMEOUT,
            execution_time=duration,
            stage_results=self.result_aggregator.get_stage_results(),
            agent_outputs=self.result_aggregator.aggregate(),
            errors=[
                AgentError(
                    agent_name="orchestrator",
                    error_type="TIMEOUT",
                    error_message=f"Pipeline timeout exceeded ({self.config.pipeline_timeout}s)",
                    timestamp=end,
                    retries_attempted=0,
                )
            ],
            telemetry=self.result_aggregator.calculate_telemetry(),
            circuit_breaker_states={
                n: cb.get_state() for n, cb in self.circuit_breakers.items()
            },
            metadata=PipelineMetadata(
                start_time=start,
                end_time=end,
                retries_by_agent=self.retry_policy.get_retry_stats(),
                timeouts_by_agent=self.timeout_manager.get_timeout_stats(),
            ),
        )

    def _build_error_result(
        self,
        cid: str,
        error: Exception,
        duration: float,
        start: datetime,
        end: datetime,
    ) -> PipelineResult:
        return PipelineResult(
            correlation_id=cid,
            status=PipelineStatus.FAILED,
            execution_time=duration,
            stage_results=self.result_aggregator.get_stage_results(),
            agent_outputs=self.result_aggregator.aggregate(),
            errors=[
                AgentError(
                    agent_name="orchestrator",
                    error_type="UNKNOWN",
                    error_message=str(error),
                    timestamp=end,
                    retries_attempted=0,
                )
            ],
            telemetry=self.result_aggregator.calculate_telemetry(),
            circuit_breaker_states={
                n: cb.get_state() for n, cb in self.circuit_breakers.items()
            },
            metadata=PipelineMetadata(
                start_time=start,
                end_time=end,
                retries_by_agent=self.retry_policy.get_retry_stats(),
                timeouts_by_agent=self.timeout_manager.get_timeout_stats(),
            ),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all registered agents.

        Returns:
            Mapping of agent_name → healthy (bool).
        """
        agents = {name: node.agent_instance for name, node in self.dag.nodes.items()}
        statuses = await self.health_checker.check_all_agents(agents)
        return {name: st.is_healthy for name, st in statuses.items()}

    def export_metrics(self) -> str:
        """Export Prometheus metrics in text exposition format.

        Returns:
            Multi-line Prometheus text format string.
        """
        return self.metrics_collector.export_metrics()
