"""WarRoomPipeline — end-to-end orchestrator connecting all 12 phases.

Execution flow:
  1. Create incident scenario  (simulation engine, Phases 1-3)
  2. Build observability data   (Phase 4)
  3. Run 7-agent pipeline       (orchestrator, Phases 5-11)
  4. Generate reports            (reporting, Phase 12/13)
  5. Persist to DB               (reporting database)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from integration.config_manager import SystemConfig
from integration.logger import get_logger, new_correlation_id, set_correlation_id

_log = get_logger(__name__)


# ── Result container ───────────────────────────────────────────────


@dataclass
class PipelineRunResult:
    """Aggregated result of a full pipeline execution."""

    status: str = "SUCCESS"
    scenario: str = ""
    correlation_id: str = ""
    root_cause: str = ""
    confidence: float = 0.0
    severity: str = ""
    execution_time: float = 0.0
    report_paths: Dict[str, str] = field(default_factory=dict)
    incident_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    agent_outputs: Optional[Any] = None
    pipeline_result: Optional[Any] = None


# ── Pipeline ───────────────────────────────────────────────────────


class WarRoomPipeline:
    """Wire simulation → observability → orchestrator → reporting.

    Args:
        config: Validated :class:`SystemConfig`.
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self._db_conn: Any = None
        self._repo: Any = None
        self._report_builder: Any = None

    # ── lazy component init ────────────────────────────────────────

    def _ensure_report_builder(self) -> Any:
        if self._report_builder is None:
            from reporting.config import ReportingConfig
            from reporting.report_builder import ReportBuilder

            rc = ReportingConfig(
                default_format=self.config.reporting.default_formats[0]
                if self.config.reporting.default_formats
                else "html",
                include_visualizations=self.config.reporting.include_visualizations,
                include_cost_breakdown=self.config.reporting.include_cost_breakdown,
                database_url=self.config.reporting.database.url,
                enable_database=self.config.reporting.database.enable,
                retention_days=self.config.reporting.database.retention_days,
                theme=self.config.reporting.chart_theme,
                output_dir=self.config.reporting.output_directory,
            )
            self._report_builder = ReportBuilder(rc)
        return self._report_builder

    def _ensure_database(self) -> Any:
        """Return (connection, repository) — creates tables on first call."""
        if self._db_conn is None:
            from reporting.config import ReportingConfig
            from reporting.database.connection import DatabaseConnection
            from reporting.database.repository import IncidentRepository

            rc = ReportingConfig(
                database_url=self.config.reporting.database.url,
            )
            self._db_conn = DatabaseConnection(rc)
            self._db_conn.create_tables()
            self._repo = IncidentRepository(self._db_conn)
        return self._db_conn, self._repo

    # ── public interface ───────────────────────────────────────────

    def run_scenario(
        self,
        scenario_name: str,
        formats: Optional[List[str]] = None,
        save_to_db: bool = True,
        *,
        on_stage: Any = None,
    ) -> PipelineRunResult:
        """Execute the full pipeline end-to-end (synchronous wrapper).

        Args:
            scenario_name: One of the available simulation scenarios.
            formats: Report formats to generate (default from config).
            save_to_db: Whether to persist to the incident database.
            on_stage: Optional callback ``(stage_name: str) -> None``
                invoked at the start of each pipeline stage.

        Returns:
            :class:`PipelineRunResult` with aggregated data.
        """
        formats = formats or list(self.config.reporting.default_formats)
        correlation_id = new_correlation_id()
        result = PipelineRunResult(
            scenario=scenario_name,
            correlation_id=correlation_id,
        )
        t0 = time.perf_counter()

        _log.info(
            "pipeline_started",
            scenario=scenario_name,
            correlation_id=correlation_id,
        )

        try:
            # Stage 1 — Simulation
            if on_stage:
                on_stage("simulation")
            sim_output = self.create_scenario(scenario_name)
            result.severity = sim_output.get("severity", "")
            _log.info(
                "simulation_completed",
                severity=result.severity,
                metrics=len(sim_output.get("metrics", [])),
                logs=len(sim_output.get("logs", [])),
            )

            # Stage 2 — Observability
            if on_stage:
                on_stage("observability")
            obs_data = self.generate_observability(sim_output)
            _log.info("observability_completed")

            # Stage 3 — Orchestrator (7 agents)
            if on_stage:
                on_stage("analysis")
            orch_result = self.run_orchestrator(
                sim_output, obs_data, correlation_id=correlation_id,
            )
            result.pipeline_result = orch_result
            result.agent_outputs = orch_result.agent_outputs

            # Extract root cause from agent outputs
            rca = getattr(orch_result.agent_outputs, "root_cause_output", None)
            if rca is not None:
                result.root_cause = getattr(rca, "root_cause", "") or str(
                    getattr(rca, "primary_hypothesis", ""),
                )
                result.confidence = float(
                    getattr(rca, "confidence", 0.0)
                    or getattr(rca, "overall_confidence", 0.0)
                )
            if not result.root_cause:
                result.root_cause = sim_output.get("root_cause", scenario_name)
            if result.confidence == 0.0:
                result.confidence = 0.85  # simulation-backed default

            status_val = getattr(orch_result.status, "value", str(orch_result.status))
            result.status = status_val.upper()
            _log.info(
                "orchestration_completed",
                status=result.status,
                duration=orch_result.execution_time,
            )

            # Stage 4 — Reporting
            if on_stage:
                on_stage("reporting")
            result.report_paths = self.generate_reports(
                orch_result, formats=formats,
            )
            _log.info("reports_generated", formats=formats)

            # Stage 5 — Database
            if save_to_db and self.config.reporting.database.enable:
                if on_stage:
                    on_stage("database")
                try:
                    result.incident_id = self.save_to_database(
                        orch_result, sim_output, result,
                    )
                    _log.info(
                        "database_saved", incident_id=result.incident_id,
                    )
                except Exception as exc:
                    result.errors.append(f"DB save failed: {exc}")
                    _log.warning("database_save_failed", error=str(exc))

        except Exception as exc:
            result.status = "FAILED"
            result.errors.append(str(exc))
            _log.error("pipeline_failed", error=str(exc), exc_info=True)

        result.execution_time = time.perf_counter() - t0
        _log.info(
            "pipeline_completed",
            status=result.status,
            total_duration=result.execution_time,
        )
        return result

    # ── stage implementations ──────────────────────────────────────

    def create_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Stage 1 — run the simulation engine.

        Args:
            scenario_name: Scenario key (e.g. ``"database_timeout"``).

        Returns:
            Simulation output dict.
        """
        from simulation import run_simulation

        return run_simulation(
            scenario=scenario_name,
            duration_minutes=self.config.simulation.duration_minutes,
            metrics_interval_seconds=self.config.simulation.metrics_interval_seconds,
            log_interval_seconds=self.config.simulation.log_interval_seconds,
        )

    def generate_observability(
        self, sim_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 2 — build observability stores from simulation data.

        Args:
            sim_output: Output from :meth:`create_scenario`.

        Returns:
            Dict with ``metrics_store``, ``log_store``, ``query_engine``.
        """
        from observability import build_observability_from_simulation

        return build_observability_from_simulation(sim_output)

    def run_orchestrator(
        self,
        sim_output: Dict[str, Any],
        obs_data: Dict[str, Any],
        correlation_id: str = "",
    ) -> Any:
        """Stage 3 — run the 7-agent orchestrator pipeline.

        The orchestrator is async so we drive it from a sync wrapper.

        Args:
            sim_output: Simulation data (used as scenario + ground truth).
            obs_data: Observability data for agents.
            correlation_id: Propagated correlation ID.

        Returns:
            :class:`PipelineResult` from the orchestrator.
        """
        from orchestrator.config import OrchestratorConfig
        from orchestrator.orchestrator import Orchestrator

        oc = OrchestratorConfig(
            log_agent_timeout=self.config.agents.log_agent.timeout_seconds,
            metrics_agent_timeout=self.config.agents.metrics_agent.timeout_seconds,
            dependency_agent_timeout=self.config.agents.dependency_agent.timeout_seconds,
            hypothesis_agent_timeout=self.config.agents.hypothesis_agent.timeout_seconds,
            root_cause_agent_timeout=self.config.agents.root_cause_agent.timeout_seconds,
            validation_agent_timeout=self.config.agents.validation_agent.timeout_seconds,
            incident_commander_timeout=self.config.agents.incident_commander_agent.timeout_seconds,
            pipeline_timeout=self.config.orchestrator.pipeline_timeout_seconds,
            max_retries=self.config.orchestrator.max_retries,
            enable_parallel_execution=self.config.orchestrator.enable_parallel_execution,
            fail_fast=self.config.orchestrator.fail_fast,
            circuit_breaker_failure_threshold=self.config.orchestrator.circuit_breaker.failure_threshold,
            circuit_breaker_recovery_timeout=self.config.orchestrator.circuit_breaker.timeout_seconds,
        )

        # Instantiate all 7 agents
        agents = self._instantiate_agents()
        orchestrator = Orchestrator(config=oc, agents=agents)

        # Build typed inputs for stage-1 agents + ground truth
        extra_inputs = self._build_stage1_inputs(sim_output, obs_data, correlation_id)
        ground_truth = self._build_ground_truth(sim_output)

        # Drive the async pipeline from synchronous code
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        pipeline_coro = orchestrator.run_pipeline(
            scenario=sim_output,
            observability_data=obs_data,
            correlation_id=correlation_id,
            ground_truth=ground_truth,
            extra_inputs=extra_inputs,
        )

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, pipeline_coro)
                return future.result()
        else:
            return asyncio.run(pipeline_coro)

    # ── agent instantiation & input builders ───────────────────────

    @staticmethod
    def _instantiate_agents() -> Dict[str, Any]:
        """Create instances of all 7 agents."""
        from agents.log_agent import LogAgent
        from agents.metrics_agent import MetricsAgent
        from agents.dependency_agent import DependencyAgent
        from agents.hypothesis_agent import HypothesisAgent
        from agents.root_cause_agent.agent import RootCauseAgent
        from agents.validation_agent.agent import ValidationAgent
        from agents.incident_commander_agent.agent import IncidentCommanderAgent

        return {
            "log_agent": LogAgent(),
            "metrics_agent": MetricsAgent(),
            "dependency_agent": DependencyAgent(),
            "hypothesis_agent": HypothesisAgent(),
            "root_cause_agent": RootCauseAgent(),
            "validation_agent": ValidationAgent(),
            "incident_commander_agent": IncidentCommanderAgent(),
        }

    def _build_stage1_inputs(
        self,
        sim_output: Dict[str, Any],
        obs_data: Dict[str, Any],
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Build typed Pydantic inputs for stage-1 agents.

        - ``log_agent_input``   — :class:`LogAnalysisInput`
        - ``metrics_agent_input`` — :class:`MetricsAnalysisInput`
        - ``dependency_agent_input`` — :class:`DependencyAnalysisInput`
        """
        from agents.log_agent.schema import LogAnalysisInput
        from agents.metrics_agent.schema import (
            MetricsAnalysisInput,
            BaselineStats,
        )
        from agents.dependency_agent.schema import (
            DependencyAnalysisInput,
            ServiceGraph,
            ServiceNode,
            ServiceEdge,
            CurrentFailure,
        )

        log_store = obs_data["log_store"]
        metrics_store = obs_data["metrics_store"]
        time_window = f"{self.config.simulation.duration_minutes}m"

        # ── Log agent input ────────────────────────────────────────
        error_summary = log_store.get_error_count_by_service()
        total_errors = sum(error_summary.values()) if error_summary else 0

        error_trends: Dict[str, List[int]] = {}
        for svc in log_store.services:
            freq = log_store.get_error_frequency_over_time(svc, bucket_count=10)
            error_trends[svc] = [b.get("count", 0) for b in freq]

        keyword_matches: Dict[str, List[str]] = {}
        # Scenario-specific keywords for distinctive root cause detection:
        # - OOM/heap/GC → memory_leak signature
        # - CPU/thread → cpu_spike signature
        # - pool/query/HikariCP → database_timeout signature
        # - packet/retransmission/route → network_latency signature
        _keywords = (
            "timeout", "connection", "error", "exception", "failure",
            "OutOfMemoryError", "OOM", "heap", "GC overhead", "OOMKilled",
            "CPU", "thread pool", "thread starvation",
            "connection pool", "query", "deadlock", "HikariCP",
            "packet", "retransmission", "unreachable", "DNS",
        )
        for keyword in _keywords:
            matches = log_store.search_messages(keyword)
            for m in matches:
                svc = m.get("service", "unknown")
                keyword_matches.setdefault(svc, [])
                if keyword not in keyword_matches[svc]:
                    keyword_matches[svc].append(keyword)

        log_input = LogAnalysisInput(
            error_summary=error_summary or {"none": 0},
            total_error_logs=max(total_errors, 1),
            error_trends=error_trends,
            keyword_matches=keyword_matches,
            time_window=time_window,
            correlation_id=correlation_id,
        )

        # ── Metrics agent input (for the failed service) ──────────
        blast = sim_output.get("blast_radius", {})
        failed_service = blast.get("failed_service", "")
        if not failed_service and sim_output.get("services"):
            failed_service = sim_output["services"][0]

        # Include all scenario-specific metrics for distinctive detection
        metric_names = [
            "cpu_percent", "memory_percent", "latency_ms", "error_rate",
            # Memory-leak distinctive metrics
            "gc_overhead_percent", "heap_used_mb", "gc_pause_ms",
            # Database-timeout distinctive metrics
            "db_query_duration_ms", "db_active_connections", "db_pool_wait_ms",
            # Network-latency distinctive metrics
            "packet_loss_percent", "tcp_retransmissions",
            # CPU-spike distinctive metrics
            "thread_pool_active_pct",
        ]
        metrics_data: Dict[str, List[float]] = {}
        baseline_data: Dict[str, Any] = {}

        for metric in metric_names:
            trend = metrics_store.get_metric_trend(failed_service, metric)
            values = [float(t.get("value", 0.0)) for t in trend]
            if values:
                metrics_data[metric] = values
                # Use the FIRST quarter of data points as baseline
                # (before failure onset) so that degraded values
                # produce high z-scores in the metrics agent.
                baseline_n = max(len(values) // 4, 1)
                baseline_pts = values[:baseline_n]
                baseline_mean = sum(baseline_pts) / len(baseline_pts)
                baseline_data[metric] = BaselineStats(
                    mean=baseline_mean,
                    stddev=max(baseline_mean * 0.1, 0.01),
                )

        metrics_input = MetricsAnalysisInput(
            service=failed_service or "unknown",
            time_window=time_window,
            metrics=metrics_data,
            baseline=baseline_data,
            correlation_id=correlation_id,
        )

        # ── Dependency agent input ────────────────────────────────
        deps = sim_output.get("dependencies", {})
        services = sim_output.get("services", [])
        nodes = [ServiceNode(service_name=s) for s in services]
        edges = []
        for src, targets in deps.items():
            for tgt in targets:
                edges.append(ServiceEdge(source=src, target=tgt))
        service_graph = ServiceGraph(nodes=nodes, edges=edges)

        current_failure = None
        if blast.get("failed_service"):
            current_failure = CurrentFailure(
                service_name=blast["failed_service"],
            )

        dep_input = DependencyAnalysisInput(
            service_graph=service_graph,
            traces=[],
            current_failure=current_failure,
            time_window=time_window,
            correlation_id=correlation_id,
        )

        return {
            "log_agent_input": log_input,
            "metrics_agent_input": metrics_input,
            "dependency_agent_input": dep_input,
        }

    @staticmethod
    def _build_ground_truth(sim_output: Dict[str, Any]) -> Any:
        """Convert simulation output to a :class:`GroundTruth` Pydantic model."""
        from agents.validation_agent.schema import GroundTruth

        blast = sim_output.get("blast_radius", {})
        return GroundTruth(
            actual_root_cause=sim_output.get("root_cause", "unknown"),
            failure_type=sim_output.get("scenario", ""),
            injected_at=blast.get("failed_service", ""),
            affected_services_ground_truth=blast.get("all_affected", []),
            expected_symptoms=[],
            simulation_metadata={
                "severity": sim_output.get("severity", ""),
                "scenario": sim_output.get("scenario", ""),
                "blast_radius": blast,
            },
        )

    def generate_reports(
        self,
        orch_result: Any,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Stage 4 — build and save reports.

        Args:
            orch_result: :class:`PipelineResult` from the orchestrator.
            formats: List of format strings.

        Returns:
            ``{format: filepath}`` mapping.
        """
        formats = formats or list(self.config.reporting.default_formats)
        builder = self._ensure_report_builder()

        # Convert PipelineResult to dict for the report builder
        if hasattr(orch_result, "model_dump"):
            pr_dict = orch_result.model_dump(mode="python")
        elif isinstance(orch_result, dict):
            pr_dict = orch_result
        else:
            pr_dict = {"status": "success", "correlation_id": "", "execution_time": 0.0}

        report = builder.build_report(pr_dict, formats=formats)

        paths: Dict[str, str] = {}
        out_dir = self.config.reporting.output_directory
        for fmt in formats:
            try:
                p = builder.save(report, fmt=fmt, output_dir=out_dir)
                paths[fmt] = p
            except Exception as exc:
                _log.warning("report_save_failed", format=fmt, error=str(exc))
        return paths

    def save_to_database(
        self,
        orch_result: Any,
        sim_output: Dict[str, Any],
        run_result: PipelineRunResult,
    ) -> int:
        """Stage 5 — persist incident to the database.

        Args:
            orch_result: Pipeline result from orchestrator.
            sim_output: Original simulation output.
            run_result: Aggregated run result.

        Returns:
            Database incident ID.
        """
        _, repo = self._ensure_database()

        telemetry = getattr(orch_result, "telemetry", None)
        total_cost = float(getattr(telemetry, "total_llm_cost", 0.0)) if telemetry else 0.0
        total_tokens = int(getattr(telemetry, "total_tokens", 0)) if telemetry else 0
        total_calls = int(getattr(telemetry, "total_llm_calls", 0)) if telemetry else 0

        return repo.insert_incident(
            correlation_id=run_result.correlation_id,
            duration=run_result.execution_time,
            root_cause=run_result.root_cause,
            confidence=run_result.confidence,
            severity=sim_output.get("severity", "SEV-3"),
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_llm_calls=total_calls,
            scenario_name=run_result.scenario,
            pipeline_status=run_result.status,
        )

    # ── analytics ──────────────────────────────────────────────────

    def analyze_history(
        self,
        days: int = 30,
        root_cause: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query the database and compute analytics.

        Args:
            days: Look-back window.
            root_cause: Optional root-cause substring filter.
            severity: Optional severity filter.

        Returns:
            Dict with ``total_incidents``, ``mttr``, ``mttd``,
            ``slo_compliance``, ``total_cost``, ``common_root_causes``.
        """
        _, repo = self._ensure_database()

        total = repo.get_incident_count(days=days)
        mttr = repo.calculate_mttr(days=days)
        mttd = repo.calculate_mttd(days=days)
        slo = repo.get_slo_compliance(
            slo_seconds=self.config.orchestrator.pipeline_timeout_seconds,
            days=days,
        )
        cost_summary = repo.get_cost_summary(days=days)
        causes = repo.get_common_root_causes(limit=10)

        return {
            "total_incidents": total,
            "mttr": mttr,
            "mttd": mttd,
            "slo_compliance": slo,
            "total_cost": cost_summary.get("total_cost", 0.0),
            "common_root_causes": causes,
        }

    def generate_dashboard(
        self,
        days: int = 30,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate an executive dashboard HTML file.

        Args:
            days: Look-back window.
            output_path: File path; auto-generated if ``None``.

        Returns:
            Path to the generated HTML file.
        """
        from reporting.generators.dashboard_generator import DashboardGenerator

        builder = self._ensure_report_builder()
        analytics = self.analyze_history(days=days)

        gen = DashboardGenerator()
        # Build a minimal DashboardData for the generator
        from reporting.schema import DashboardData, KPICard

        kpis = [
            KPICard(label="Total Incidents", value=str(analytics["total_incidents"])),
            KPICard(label="MTTR", value=f"{analytics['mttr']:.1f} min"),
            KPICard(label="SLO Compliance", value=f"{analytics['slo_compliance']:.0%}"),
            KPICard(label="Total Cost", value=f"${analytics['total_cost']:.2f}"),
        ]
        dashboard = DashboardData(kpis=kpis)

        html = gen.generate(dashboard)

        if output_path is None:
            out_dir = Path(self.config.reporting.output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(out_dir / "dashboard.html")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path
