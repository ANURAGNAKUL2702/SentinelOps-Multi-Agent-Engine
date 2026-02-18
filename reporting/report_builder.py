"""Report builder — main orchestrator that assembles incident reports.

Transforms ``PipelineResult`` from the orchestration layer into
fully-rendered incident reports in HTML, Markdown, JSON, and PDF
formats, optionally persisting results to the incident database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ReportingConfig
from .schema import (
    ActionItem,
    CostReport,
    DashboardData,
    ExecutiveSummary,
    HistoricalAnalytics,
    IncidentDetails,
    IncidentReport,
    IncidentStatus,
    IncidentTimeline,
    KPICard,
    PerformanceMetrics,
    Recommendation,
    RemediationPlan,
    ReportFormat,
    ReportMetadata,
    RootCauseAnalysis,
    SeverityLevel,
    TimelineEvent,
    TrendDirection,
)
from .telemetry import get_logger

_logger = get_logger(__name__)


class ReportBuilder:
    """Assemble a complete :class:`IncidentReport` from a pipeline result.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self._cfg = config or ReportingConfig()
        self._generated_reports: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_report(
        self,
        pipeline_result: Dict[str, Any],
        formats: Optional[List[str]] = None,
    ) -> IncidentReport:
        """Build an :class:`IncidentReport` from a pipeline result dict.

        Args:
            pipeline_result: Dictionary representing a ``PipelineResult``
                (or its ``.model_dump()`` output).
            formats: Optional list of format strings to generate.

        Returns:
            A fully-populated :class:`IncidentReport`.
        """
        report_id = str(uuid.uuid4())
        correlation_id = str(pipeline_result.get("correlation_id", ""))

        metadata = ReportMetadata(
            report_id=report_id,
            correlation_id=correlation_id,
            generated_at=datetime.now(timezone.utc),
            report_version="1.0.0",
        )

        executive_summary = self._build_executive_summary(pipeline_result)
        incident_details = self._build_incident_details(pipeline_result)
        root_cause = self._build_root_cause(pipeline_result)
        remediation = self._build_remediation(pipeline_result)
        timeline = self._build_timeline(pipeline_result)
        cost = self._build_cost_report(pipeline_result)
        performance = self._build_performance(pipeline_result)
        recommendations = self._build_recommendations(pipeline_result)
        visualizations = self._build_visualizations(pipeline_result)

        report = IncidentReport(
            metadata=metadata,
            executive_summary=executive_summary,
            incident_details=incident_details,
            root_cause_analysis=root_cause,
            remediation_plan=remediation,
            timeline=timeline,
            cost_report=cost,
            performance_metrics=performance,
            recommendations=recommendations,
            visualizations=visualizations,
        )

        # Track generated report for later retrieval
        self._generated_reports[report_id] = {
            "report_id": report_id,
            "correlation_id": correlation_id,
            "generated_at": metadata.generated_at.isoformat(),
            "formats": formats or [self._cfg.default_format],
            "report": report,
        }

        _logger.info(
            "Built report %s for correlation_id=%s",
            report_id,
            correlation_id,
        )
        return report

    def render(
        self,
        report: IncidentReport,
        fmt: str = "html",
    ) -> str:
        """Render a report to a string in the given format.

        Args:
            report: The incident report to render.
            fmt: Output format (``html``, ``markdown``, ``json``).

        Returns:
            Rendered string content.
        """
        fmt_lower = fmt.lower()
        if fmt_lower == "json":
            from .generators.json_generator import JSONGenerator
            return JSONGenerator().generate(report)
        elif fmt_lower == "markdown":
            from .generators.markdown_generator import MarkdownGenerator
            return MarkdownGenerator(config=self._cfg).generate(report)
        elif fmt_lower == "html":
            from .generators.html_generator import HTMLGenerator
            return HTMLGenerator(config=self._cfg).generate(report)
        else:
            from .generators.json_generator import JSONGenerator
            return JSONGenerator().generate(report)

    def save(
        self,
        report: IncidentReport,
        fmt: str = "html",
        output_dir: Optional[str] = None,
    ) -> str:
        """Render and save a report to disk.

        Args:
            report: The incident report.
            fmt: Output format.
            output_dir: Directory to write files.  Uses config default
                if omitted.

        Returns:
            Absolute path of the written file.
        """
        out_dir = Path(output_dir or self._cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        content = self.render(report, fmt)
        ext = {"html": ".html", "markdown": ".md", "json": ".json", "pdf": ".pdf"}.get(
            fmt.lower(), ".txt"
        )
        filename = f"{report.metadata.correlation_id or report.metadata.report_id}{ext}"
        path = out_dir / filename
        path.write_text(content, encoding="utf-8")
        _logger.info("Saved report to %s", path)

        # Track file path
        rid = report.metadata.report_id
        if rid in self._generated_reports:
            files = self._generated_reports[rid].setdefault("files", {})
            files[fmt.lower()] = str(path)

        return str(path)

    def get_report_metadata(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a previously generated report.

        Args:
            report_id: UUID of the report.

        Returns:
            Metadata dict or ``None``.
        """
        entry = self._generated_reports.get(report_id)
        if entry is None:
            return None
        return {
            "report_id": entry["report_id"],
            "correlation_id": entry["correlation_id"],
            "generated_at": entry["generated_at"],
            "formats": entry["formats"],
            "files": entry.get("files", {}),
        }

    def get_report_file(self, report_id: str, fmt: str) -> Optional[str]:
        """Return the file path for a generated report.

        Args:
            report_id: UUID of the report.
            fmt: Requested format.

        Returns:
            File path string or ``None``.
        """
        entry = self._generated_reports.get(report_id)
        if entry is None:
            return None
        return entry.get("files", {}).get(fmt.lower())

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_executive_summary(
        self, pr: Dict[str, Any],
    ) -> ExecutiveSummary:
        agent_outputs = pr.get("agent_outputs", {})
        rc_output = agent_outputs.get("root_cause_output") or {}
        val_output = agent_outputs.get("validation_output") or {}
        ic_output = agent_outputs.get("incident_response") or {}

        root_cause = rc_output.get("root_cause", "Unknown")
        confidence = float(rc_output.get("confidence", 0.0))
        severity_str = rc_output.get("severity", "medium")
        sev_level = self._map_severity(severity_str)

        # Build summary text 
        status_str = str(pr.get("status", "success"))
        exec_time = float(pr.get("execution_time", 0.0))
        summary = (
            f"Incident resolved in {exec_time:.1f}s. "
            f"Root cause: {root_cause} "
            f"(confidence: {confidence:.0%}). "
            f"Pipeline status: {status_str}."
        )

        dep_output = agent_outputs.get("dependency_output") or {}
        affected = len(dep_output.get("affected_services", []))

        return ExecutiveSummary(
            summary=summary,
            severity=sev_level,
            status=IncidentStatus.RESOLVED,
            affected_services=affected,
            confidence=confidence,
        )

    def _build_incident_details(
        self, pr: Dict[str, Any],
    ) -> IncidentDetails:
        agent_outputs = pr.get("agent_outputs", {})
        dep_output = agent_outputs.get("dependency_output") or {}
        log_output = agent_outputs.get("log_output") or {}

        meta = pr.get("metadata") or {}
        start = meta.get("start_time")
        end = meta.get("end_time")

        return IncidentDetails(
            incident_id=pr.get("correlation_id", ""),
            started_at=start,
            detected_at=start,
            resolved_at=end,
            duration_seconds=float(pr.get("execution_time", 0.0)),
            affected_services=dep_output.get("affected_services", []),
        )

    def _build_root_cause(
        self, pr: Dict[str, Any],
    ) -> RootCauseAnalysis:
        agent_outputs = pr.get("agent_outputs", {})
        rc = agent_outputs.get("root_cause_output") or {}
        val = agent_outputs.get("validation_output") or {}

        evidence = rc.get("evidence", [])
        if isinstance(evidence, list):
            evidence_trail = [
                e if isinstance(e, dict) else {"description": str(e)}
                for e in evidence
            ]
        else:
            evidence_trail = []

        causal_chain = rc.get("causal_chain", [])
        if isinstance(causal_chain, list):
            causal_chain = [
                c.get("description", str(c)) if isinstance(c, dict) else str(c)
                for c in causal_chain
            ]

        hallucinations = val.get("hallucinations", [])
        n_hall = len(hallucinations) if isinstance(hallucinations, list) else 0

        return RootCauseAnalysis(
            root_cause=rc.get("root_cause", ""),
            confidence=float(rc.get("confidence", 0.0)),
            evidence_trail=evidence_trail,
            causal_chain=causal_chain,
            validation_accuracy=float(val.get("accuracy", 0.0)),
            hallucinations_detected=n_hall,
        )

    def _build_remediation(
        self, pr: Dict[str, Any],
    ) -> RemediationPlan:
        agent_outputs = pr.get("agent_outputs", {})
        ic = agent_outputs.get("incident_response") or {}
        runbook = ic.get("runbook") or {}

        steps = runbook.get("steps", [])
        if isinstance(steps, list):
            steps = [
                s.get("description", str(s)) if isinstance(s, dict) else str(s)
                for s in steps
            ]

        action_items_raw = ic.get("action_items", [])
        action_items = []
        for ai in (action_items_raw if isinstance(action_items_raw, list) else []):
            if isinstance(ai, dict):
                action_items.append(ActionItem(
                    priority=ai.get("priority", "P2"),
                    description=ai.get("description", ""),
                    owner=ai.get("owner", "SRE Team"),
                    estimated_minutes=float(ai.get("estimated_minutes", 15)),
                ))

        return RemediationPlan(
            runbook_title=runbook.get("title", ""),
            runbook_steps=steps,
            action_items=action_items,
            rollback_plan=self._rollback_to_str(ic.get("rollback_plan", "")),
        )

    @staticmethod
    def _rollback_to_str(value: Any) -> str:
        """Coerce a rollback_plan value (dict or str) to a readable string."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            strategy = value.get("strategy", "")
            steps = value.get("steps", [])
            parts = [str(strategy)] if strategy else []
            for s in (steps if isinstance(steps, list) else []):
                desc = s.get("description", str(s)) if isinstance(s, dict) else str(s)
                parts.append(f"- {desc}")
            return "\n".join(parts) if parts else str(value)
        if hasattr(value, "model_dump"):
            return str(value.model_dump())
        return str(value)

    def _build_timeline(
        self, pr: Dict[str, Any],
    ) -> IncidentTimeline:
        agent_outputs = pr.get("agent_outputs", {})
        log_output = agent_outputs.get("log_output") or {}

        events_raw = log_output.get("timeline", [])
        events = []
        for e in (events_raw if isinstance(events_raw, list) else []):
            if isinstance(e, dict):
                events.append(TimelineEvent(
                    source=e.get("source", ""),
                    event=e.get("event", e.get("message", "")),
                    service=e.get("service", ""),
                    severity=e.get("severity", "info"),
                ))

        return IncidentTimeline(events=events)

    def _build_cost_report(
        self, pr: Dict[str, Any],
    ) -> CostReport:
        telemetry = pr.get("telemetry") or {}
        return CostReport(
            total_cost=float(telemetry.get("total_llm_cost", 0.0)),
            total_tokens=int(telemetry.get("total_tokens", 0)),
            total_llm_calls=int(telemetry.get("total_llm_calls", 0)),
            cost_by_agent=telemetry.get("agent_costs", {}),
            tokens_by_agent=telemetry.get("agent_tokens", {}),
        )

    def _build_performance(
        self, pr: Dict[str, Any],
    ) -> PerformanceMetrics:
        telemetry = pr.get("telemetry") or {}
        return PerformanceMetrics(
            total_pipeline_time=float(pr.get("execution_time", 0.0)),
            agent_latencies=telemetry.get("agent_latencies", {}),
            parallel_speedup=float(telemetry.get("parallel_speedup", 1.0)),
            timeout_violations=int(telemetry.get("timeout_violations", 0)),
            circuit_breaker_trips=int(telemetry.get("circuit_breaker_trips", 0)),
        )

    def _build_recommendations(
        self, pr: Dict[str, Any],
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []
        telemetry = pr.get("telemetry") or {}

        if float(pr.get("execution_time", 0.0)) > self._cfg.target_pipeline_time:
            recs.append(Recommendation(
                category="performance",
                description="Pipeline exceeded target time — investigate slow agents.",
                priority="P1",
            ))

        if int(telemetry.get("timeout_violations", 0)) > 0:
            recs.append(Recommendation(
                category="reliability",
                description="Timeout violations detected — tune agent timeouts.",
                priority="P1",
            ))

        if int(telemetry.get("circuit_breaker_trips", 0)) > 0:
            recs.append(Recommendation(
                category="reliability",
                description="Circuit breaker tripped — review error rates.",
                priority="P2",
            ))

        return recs

    def _build_visualizations(
        self, pr: Dict[str, Any],
    ) -> Dict[str, str]:
        if not self._cfg.include_visualizations:
            return {}

        viz: Dict[str, str] = {}

        try:
            from .visualizations.timeline_chart import TimelineChart

            agent_outputs = pr.get("agent_outputs", {})
            log_output = agent_outputs.get("log_output") or {}
            events = log_output.get("timeline", [])
            chart = TimelineChart()
            viz["timeline"] = chart.generate(events)
        except Exception:
            _logger.debug("Timeline chart generation skipped", exc_info=True)

        try:
            from .visualizations.cost_breakdown_chart import CostBreakdownChart

            telemetry = pr.get("telemetry") or {}
            cost_by_agent = telemetry.get("agent_costs", {})
            if cost_by_agent:
                chart = CostBreakdownChart()
                viz["cost_pie"] = chart.generate_pie_chart(cost_by_agent)
        except Exception:
            _logger.debug("Cost chart generation skipped", exc_info=True)

        try:
            from .visualizations.confidence_gauge import ConfidenceGauge

            rc = (pr.get("agent_outputs") or {}).get("root_cause_output") or {}
            confidence = float(rc.get("confidence", 0.0))
            gauge = ConfidenceGauge()
            viz["confidence"] = gauge.generate(confidence)
        except Exception:
            _logger.debug("Confidence gauge generation skipped", exc_info=True)

        return viz

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_severity(raw: str) -> SeverityLevel:
        mapping = {
            "critical": SeverityLevel.P0_CRITICAL,
            "high": SeverityLevel.P1_HIGH,
            "medium": SeverityLevel.P2_MEDIUM,
            "low": SeverityLevel.P3_LOW,
            "P0_CRITICAL": SeverityLevel.P0_CRITICAL,
            "P1_HIGH": SeverityLevel.P1_HIGH,
            "P2_MEDIUM": SeverityLevel.P2_MEDIUM,
            "P3_LOW": SeverityLevel.P3_LOW,
        }
        return mapping.get(raw, SeverityLevel.P2_MEDIUM)
