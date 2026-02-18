"""Async DAG executor — parallel stage execution with timeouts, retries, and
circuit breakers."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .circuit_breaker import CircuitBreaker
from .config import OrchestratorConfig
from .dag import ExecutionDAG
from .error_handler import ErrorHandler
from .metrics_collector import MetricsCollector
from .result_aggregator import ResultAggregator
from .retry_policy import RetryPolicy
from .schema import (
    AgentNode,
    AgentOutputs,
    AgentStatus,
    CircuitBreakerOpenError,
    PipelineResult,
    PipelineStatus,
    StageResult,
)
from .state_machine import StateMachine
from .timeout_manager import TimeoutManager
from .telemetry import get_logger

_logger = get_logger(__name__)


# ── Inter-stage input adapters ─────────────────────────────────────
# These build typed Pydantic inputs for stages 2-5 from accumulated
# outputs of previous stages.  Called by _call_agent when the keyed
# input (e.g. "hypothesis_agent_input") is not already present.
# ───────────────────────────────────────────────────────────────────

def _build_hypothesis_input(inputs: Dict[str, Any], correlation_id: str) -> Any:
    """Build a :class:`HypothesisAgentInput` from stage-1 agent outputs."""
    from agents.hypothesis_agent.schema import (
        HypothesisAgentInput,
        LogFindings,
        MetricFindings,
        DependencyFindings,
    )

    log_out = inputs.get("log_agent")
    metrics_out = inputs.get("metrics_agent")
    dep_out = inputs.get("dependency_agent")

    # ── LogFindings from LogAgentOutput ──────────────────────────
    log_findings = LogFindings()
    if log_out is not None:
        suspicious = getattr(log_out, "suspicious_services", [])
        summary = getattr(log_out, "system_error_summary", None)
        log_findings = LogFindings(
            suspicious_services=[
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in suspicious
            ],
            total_error_logs=getattr(summary, "total_errors", 0)
            if summary
            else 0,
            dominant_service=getattr(summary, "dominant_service", None)
            if summary
            else None,
            system_wide_spike=getattr(summary, "system_wide_spike", False)
            if summary
            else False,
            potential_upstream_failure=getattr(
                summary, "potential_upstream_failure", False,
            )
            if summary
            else False,
            database_errors_detected=getattr(
                log_out, "database_related_errors_detected", False,
            ),
            confidence_score=getattr(log_out, "confidence_score", 0.0),
        )

    # ── MetricFindings from MetricsAgentOutput ──────────────────
    metric_findings = MetricFindings()
    if metrics_out is not None:
        anomalous = getattr(metrics_out, "anomalous_metrics", [])
        correlations = getattr(metrics_out, "correlations", [])
        sys_summary = getattr(metrics_out, "system_summary", None)
        metric_findings = MetricFindings(
            anomalous_metrics=[
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in anomalous
            ],
            correlations=[
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in correlations
            ],
            total_anomalies=getattr(sys_summary, "total_anomalies_detected", len(anomalous))
            if sys_summary
            else len(anomalous),
            critical_anomalies=getattr(sys_summary, "critical_anomalies", 0)
            if sys_summary
            else 0,
            resource_saturation=getattr(sys_summary, "resource_saturation", False)
            if sys_summary
            else False,
            cascading_degradation=getattr(sys_summary, "cascading_degradation", False)
            if sys_summary
            else False,
            confidence_score=getattr(metrics_out, "confidence_score", 0.0),
        )

    # ── DependencyFindings from DependencyAgentOutput ───────────
    dep_findings = DependencyFindings()
    if dep_out is not None:
        failed_info = getattr(dep_out, "failed_service", None)
        cascade_risk = getattr(dep_out, "cascading_failure_risk", None)
        dep_analysis = getattr(dep_out, "dependency_analysis", None)
        spofs = getattr(dep_out, "single_points_of_failure", [])
        bottlenecks = getattr(dep_out, "bottlenecks", [])
        dep_findings = DependencyFindings(
            failed_service=getattr(failed_info, "service_name", None)
            if failed_info
            else None,
            blast_radius_count=getattr(dep_analysis, "total_services", 0)
            if dep_analysis
            else 0,
            is_cascading=getattr(cascade_risk, "is_cascading", False)
            if cascade_risk
            else False,
            cascade_pattern=str(
                getattr(cascade_risk, "cascade_pattern", "isolated"),
            )
            if cascade_risk
            else "isolated",
            single_points_of_failure=[
                getattr(s, "service_name", str(s)) for s in spofs
            ],
            bottleneck_services=[
                getattr(b, "service_name", str(b)) for b in bottlenecks
            ],
            confidence_score=getattr(dep_out, "confidence_score", 0.0),
        )

    return HypothesisAgentInput(
        log_findings=log_findings,
        metric_findings=metric_findings,
        dependency_findings=dep_findings,
        correlation_id=correlation_id,
    )


def _build_root_cause_input(inputs: Dict[str, Any], correlation_id: str) -> Any:
    """Build a :class:`RootCauseAgentInput` from stage-1 + hypothesis outputs."""
    from agents.root_cause_agent.schema import (
        RootCauseAgentInput,
        LogAgentFindings,
        MetricsAgentFindings,
        DependencyAgentFindings,
        HypothesisFindings,
    )

    log_out = inputs.get("log_agent")
    metrics_out = inputs.get("metrics_agent")
    dep_out = inputs.get("dependency_agent")
    hyp_out = inputs.get("hypothesis_agent")

    # ── LogAgentFindings ─────────────────────────────────────────
    log_f = LogAgentFindings()
    if log_out is not None:
        suspicious = getattr(log_out, "suspicious_services", [])
        log_f = LogAgentFindings(
            suspicious_services=[
                getattr(s, "service", str(s)) for s in suspicious
            ],
            error_patterns=[
                kw
                for s in suspicious
                for kw in getattr(s, "error_keywords_detected", [])
            ],
            confidence=getattr(log_out, "confidence_score", 0.0),
        )

    # ── MetricsAgentFindings ─────────────────────────────────────
    metrics_f = MetricsAgentFindings()
    if metrics_out is not None:
        anomalous = getattr(metrics_out, "anomalous_metrics", [])
        correlations = getattr(metrics_out, "correlations", [])
        metrics_f = MetricsAgentFindings(
            anomalies=[
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in anomalous
            ],
            correlations=[
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in correlations
            ],
            confidence=getattr(metrics_out, "confidence_score", 0.0),
        )

    # ── DependencyAgentFindings ──────────────────────────────────
    dep_f = DependencyAgentFindings()
    if dep_out is not None:
        failed_info = getattr(dep_out, "failed_service", None)
        dep_analysis = getattr(dep_out, "dependency_analysis", None)
        cascade_risk = getattr(dep_out, "cascading_failure_risk", None)
        bottlenecks = getattr(dep_out, "bottlenecks", [])
        spofs = getattr(dep_out, "single_points_of_failure", [])
        dep_f = DependencyAgentFindings(
            impact_graph={},
            bottlenecks=[
                getattr(b, "service_name", str(b)) for b in bottlenecks
            ],
            blast_radius=getattr(dep_analysis, "total_services", 0)
            if dep_analysis
            else 0,
            affected_services=getattr(cascade_risk, "affected_services", [])
            if cascade_risk
            else [],
            confidence=getattr(dep_out, "confidence_score", 0.0),
        )

    # ── HypothesisFindings ───────────────────────────────────────
    hyp_f = HypothesisFindings()
    if hyp_out is not None:
        hypotheses = getattr(hyp_out, "hypotheses", [])
        hyp_f = HypothesisFindings(
            ranked_hypotheses=[
                h.model_dump() if hasattr(h, "model_dump") else h
                for h in hypotheses
            ],
            top_hypothesis=getattr(hyp_out, "recommended_hypothesis", ""),
            top_confidence=getattr(hyp_out, "confidence_score", 0.0),
            mttr_estimate=getattr(hyp_out, "estimated_mttr_minutes", 30.0),
            category=str(getattr(hyp_out, "category", "unknown")),
            confidence=getattr(hyp_out, "confidence_score", 0.0),
        )

    return RootCauseAgentInput(
        log_findings=log_f,
        metrics_findings=metrics_f,
        dependency_findings=dep_f,
        hypothesis_findings=hyp_f,
        correlation_id=correlation_id,
    )


def _build_commander_input(inputs: Dict[str, Any], correlation_id: str) -> Any:
    """Build an :class:`IncidentCommanderInput` from root cause + validation."""
    from agents.incident_commander_agent.schema import IncidentCommanderInput

    verdict = inputs.get("root_cause_agent")
    validation_report = inputs.get("validation_agent")

    return IncidentCommanderInput(
        verdict=verdict,
        validation_report=validation_report,
        correlation_id=correlation_id,
    )


class ExecutionEngine:
    """Execute an :class:`ExecutionDAG` stage-by-stage.

    Independent agents within a stage run concurrently via
    ``asyncio.gather(return_exceptions=True)``.  Each agent call is
    wrapped with timeout enforcement, exponential-backoff retries, and
    a per-agent circuit breaker.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        timeout_manager: TimeoutManager,
        retry_policy: RetryPolicy,
        circuit_breakers: Dict[str, CircuitBreaker],
        error_handler: ErrorHandler,
        result_aggregator: ResultAggregator,
        state_machine: StateMachine,
        metrics_collector: MetricsCollector,
    ) -> None:
        self.config = config
        self.timeout_manager = timeout_manager
        self.retry_policy = retry_policy
        self.circuit_breakers = circuit_breakers
        self.error_handler = error_handler
        self.result_aggregator = result_aggregator
        self.state_machine = state_machine
        self.metrics_collector = metrics_collector

    # ------------------------------------------------------------------
    # Single-agent execution
    # ------------------------------------------------------------------

    async def execute_agent(
        self,
        node: AgentNode,
        inputs: Dict[str, Any],
        correlation_id: str,
    ) -> Any:
        """Execute a single agent with timeout + retry + circuit-breaker.

        Args:
            node: The :class:`AgentNode` to execute.
            inputs: Data to pass to the agent.
            correlation_id: Pipeline correlation ID.

        Returns:
            The agent's output object, or raises on failure.
        """
        agent = node.agent_instance
        breaker = self.circuit_breakers.get(node.name)
        self.state_machine.set_agent_state(node.name, AgentStatus.RUNNING)

        async def _invoke() -> Any:
            coro = self._call_agent(agent, node.name, inputs, correlation_id)
            return await self.timeout_manager.execute_with_timeout(
                coro,
                timeout=node.timeout,
                agent_name=node.name,
                correlation_id=correlation_id,
            )

        start = time.perf_counter()
        try:
            if breaker is not None:
                result = await self.retry_policy.execute_with_retry(
                    _invoke,
                    agent_name=node.name,
                    circuit_breaker=breaker,
                )
            else:
                result = await self.retry_policy.execute_with_retry(
                    _invoke,
                    agent_name=node.name,
                )
            duration = time.perf_counter() - start

            if result is None:
                raise RuntimeError(f"Agent {node.name} returned None")

            self.state_machine.set_agent_state(node.name, AgentStatus.SUCCESS)
            self.result_aggregator.add_agent_output(node.name, result, duration)
            self.metrics_collector.record_agent_execution(node.name, duration, "success")
            return result

        except CircuitBreakerOpenError:
            duration = time.perf_counter() - start
            self.state_machine.set_agent_state(node.name, AgentStatus.CIRCUIT_OPEN)
            self.metrics_collector.record_agent_execution(node.name, duration, "circuit_open")
            raise

        except asyncio.TimeoutError:
            duration = time.perf_counter() - start
            self.state_machine.set_agent_state(node.name, AgentStatus.TIMEOUT)
            self.metrics_collector.record_agent_execution(node.name, duration, "timeout")
            self.metrics_collector.record_timeout(node.name)
            raise

        except Exception:
            duration = time.perf_counter() - start
            self.state_machine.set_agent_state(node.name, AgentStatus.FAILED)
            self.metrics_collector.record_agent_execution(node.name, duration, "failed")
            raise

    # ------------------------------------------------------------------
    # Stage execution (parallel within a stage)
    # ------------------------------------------------------------------

    async def execute_stage(
        self,
        stage_name: str,
        agent_names: List[str],
        dag: ExecutionDAG,
        inputs: Dict[str, Any],
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Execute all agents in a stage concurrently.

        Args:
            stage_name: Label for the stage (e.g. ``"stage_1"``).
            agent_names: List of agent names to execute in parallel.
            dag: The execution DAG (for node lookup).
            inputs: Current pipeline data.
            correlation_id: Correlation ID.

        Returns:
            Mapping of agent_name → output for agents that succeeded.
        """
        start_time = datetime.now(timezone.utc)
        t0 = time.perf_counter()

        tasks = []
        for name in agent_names:
            node = dag.nodes[name]
            tasks.append(self.execute_agent(node, inputs, correlation_id))

        if self.config.enable_parallel_execution and len(tasks) > 1:
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            raw_results = []
            for t in tasks:
                try:
                    raw_results.append(await t)
                except Exception as exc:
                    raw_results.append(exc)

        duration = time.perf_counter() - t0
        end_time = datetime.now(timezone.utc)

        outputs: Dict[str, Any] = {}
        stage_failed = False
        for name, result in zip(agent_names, raw_results):
            if isinstance(result, BaseException):
                error_type = self.error_handler.categorize_error(
                    result if isinstance(result, Exception) else RuntimeError(str(result))
                )
                self.error_handler.handle_agent_error(
                    name, result if isinstance(result, Exception) else RuntimeError(str(result)),
                    stage=stage_name,
                    retries=self.retry_policy.get_retry_stats().get(name, 0),
                )
                self.metrics_collector.record_agent_failure(name, error_type)
                if self.error_handler.should_abort(name, stage_name):
                    stage_failed = True
            else:
                outputs[name] = result

        status = "SUCCESS"
        if stage_failed:
            status = "FAILED"
        elif len(outputs) < len(agent_names):
            status = "PARTIAL"

        stage_result = StageResult(
            stage_name=stage_name,
            agents=agent_names,
            duration=duration,
            status=status,
            start_time=start_time,
            end_time=end_time,
        )
        self.result_aggregator.add_stage_result(stage_result)

        return outputs

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        dag: ExecutionDAG,
        initial_inputs: Dict[str, Any],
        correlation_id: str,
    ) -> _PipelineRunResult:
        """Execute the entire DAG end-to-end.

        Args:
            dag: Execution DAG (already validated).
            initial_inputs: Starting data (scenario, observability_data).
            correlation_id: Pipeline correlation ID.

        Returns:
            :class:`_PipelineRunResult` with all outputs, stage results, errors.
        """
        stages = dag.topological_sort()
        accumulated_inputs = dict(initial_inputs)
        abort = False

        for idx, agent_names in enumerate(stages):
            stage_name = f"stage_{idx + 1}"

            _logger.info(
                f"Executing {stage_name}: {agent_names}",
                extra={"correlation_id": correlation_id},
            )

            # Skip stage if upstream marked abort
            if abort:
                for name in agent_names:
                    self.state_machine.set_agent_state(name, AgentStatus.SKIPPED)
                continue

            outputs = await self.execute_stage(
                stage_name, agent_names, dag, accumulated_inputs, correlation_id,
            )

            # Merge outputs into accumulated inputs for next stages
            accumulated_inputs.update(outputs)

            # Check if stage failure should abort
            stage_res = self.result_aggregator.get_stage_results().get(stage_name)
            if stage_res and stage_res.status == "FAILED":
                abort = True

        return _PipelineRunResult(
            agent_outputs=self.result_aggregator.aggregate(),
            stage_results=self.result_aggregator.get_stage_results(),
            errors=self.error_handler.get_errors(),
        )

    # ------------------------------------------------------------------
    # Agent dispatch (adapts each agent's unique API)
    # ------------------------------------------------------------------

    async def _call_agent(
        self,
        agent: Any,
        agent_name: str,
        inputs: Dict[str, Any],
        correlation_id: str,
    ) -> Any:
        """Dispatch a call to the agent's main method.

        Each agent has a slightly different signature; this adapter
        normalises the calls.

        Args:
            agent: The agent instance (or async callable / mock).
            agent_name: Agent identifier.
            inputs: Accumulated pipeline data.
            correlation_id: Correlation ID.

        Returns:
            Agent output object.
        """
        # If the agent is just a callable (e.g. AsyncMock), call directly
        if callable(agent) and not hasattr(agent, "analyze") and not hasattr(agent, "validate") and not hasattr(agent, "command"):
            result = agent(inputs, correlation_id=correlation_id)
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Dispatch based on agent name
        if agent_name in ("log_agent", "metrics_agent", "dependency_agent"):
            input_data = inputs.get(f"{agent_name}_input") or inputs.get("observability_data")
            result = agent.analyze(input_data, correlation_id=correlation_id)
        elif agent_name == "hypothesis_agent":
            input_data = inputs.get("hypothesis_agent_input")
            if input_data is None:
                input_data = _build_hypothesis_input(inputs, correlation_id)
            result = agent.analyze(input_data, correlation_id=correlation_id)
        elif agent_name == "root_cause_agent":
            input_data = inputs.get("root_cause_agent_input")
            if input_data is None:
                input_data = _build_root_cause_input(inputs, correlation_id)
            result = agent.analyze(input_data)
        elif agent_name == "validation_agent":
            verdict = inputs.get("root_cause_agent")
            ground_truth = inputs.get("ground_truth")
            result = agent.validate(verdict, ground_truth, correlation_id=correlation_id)
        elif agent_name == "incident_commander_agent":
            input_data = inputs.get("incident_commander_agent_input")
            if input_data is None:
                input_data = _build_commander_input(inputs, correlation_id)
            result = agent.command(input_data)
        else:
            # Generic fallback — try .analyze() first, then call directly
            if hasattr(agent, "analyze"):
                result = agent.analyze(inputs, correlation_id=correlation_id)
            else:
                result = agent(inputs, correlation_id=correlation_id)

        if asyncio.iscoroutine(result):
            return await result
        return result


class _PipelineRunResult:
    """Internal container for pipeline run outputs (not exported)."""

    __slots__ = ("agent_outputs", "stage_results", "errors")

    def __init__(
        self,
        agent_outputs: AgentOutputs,
        stage_results: Dict[str, StageResult],
        errors: list,
    ) -> None:
        self.agent_outputs = agent_outputs
        self.stage_results = stage_results
        self.errors = errors
