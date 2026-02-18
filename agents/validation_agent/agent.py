"""
File: agent.py
Purpose: Validation Agent — the orchestration layer.
Dependencies: All submodules.
Performance: <50ms fallback, <1s with LLM.

Pipeline:
  1. Run deterministic validation (all 7 core algorithms)
  2. If verdict is incorrect and LLM enabled, analyze discrepancies
  3. Merge LLM recommendations into report
  4. Run output validation checks
  5. Return ValidationReport

Entry point::

    agent = ValidationAgent()
    report = agent.validate(verdict, ground_truth)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.validation_agent.config import ValidationAgentConfig
from agents.validation_agent.fallback import DeterministicFallback
from agents.validation_agent.llm.discrepancy_analyzer import (
    DiscrepancyAnalyzer,
)
from agents.validation_agent.schema import (
    GroundTruth,
    RootCauseVerdict,
    ValidationAgentInput,
    ValidationReport,
)
from agents.validation_agent.telemetry import TelemetryCollector, get_logger
from agents.validation_agent.validator import ReportValidator

logger = get_logger("validation_agent.agent")


class ValidationAgent:
    """Validation Agent — quality assurance layer.

    Cross-checks the Root Cause Agent's verdict against ground truth
    simulation data, detects hallucinations, and produces
    accuracy/precision/recall metrics.

    Pipeline::

        Phase 1: Deterministic validation (7 core algorithms)
        Phase 2: LLM discrepancy analysis (if enabled + incorrect)
        Phase 3: Output validation checks

    Args:
        config: Agent configuration.

    Example::

        agent = ValidationAgent()
        report = agent.validate(verdict, ground_truth)
        print(report.verdict_correct, report.accuracy_score)
    """

    def __init__(
        self, config: Optional[ValidationAgentConfig] = None
    ) -> None:
        self._config = config or ValidationAgentConfig()
        self._telemetry = TelemetryCollector()
        self._fallback = DeterministicFallback(
            self._config, self._telemetry
        )
        self._validator = ReportValidator(self._config)

        # LLM discrepancy analyzer (only created if enabled)
        self._analyzer: Optional[DiscrepancyAnalyzer] = None
        if self._config.features.use_llm:
            self._analyzer = DiscrepancyAnalyzer(
                self._config, self._telemetry
            )

    def validate(
        self,
        verdict: RootCauseVerdict,
        ground_truth: GroundTruth,
        correlation_id: str = "",
        history: List[Dict[str, Any]] | None = None,
    ) -> ValidationReport:
        """Run the full validation pipeline.

        Args:
            verdict: Root cause verdict to validate.
            ground_truth: Simulation ground truth.
            correlation_id: Request correlation ID.
            history: Historical prediction data for calibration.

        Returns:
            Complete ValidationReport.
        """
        pipeline_start = time.perf_counter()
        self._telemetry.validations_total.inc()

        logger.info(
            f"Validation started — verdict='{verdict.root_cause[:50]}'",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
            },
        )

        try:
            report = self._execute_pipeline(
                verdict, ground_truth, correlation_id,
                history, pipeline_start,
            )
            self._telemetry.validations_succeeded.inc()
            return report

        except Exception as exc:
            self._telemetry.validations_failed.inc()
            logger.error(
                f"Validation pipeline failed: {exc}",
                extra={
                    "correlation_id": correlation_id,
                    "layer": "pipeline",
                },
            )
            return self._error_report(str(exc), pipeline_start)

    def validate_input(
        self, input_data: ValidationAgentInput
    ) -> ValidationReport:
        """Validate using a ValidationAgentInput object.

        Args:
            input_data: Structured input with verdict + ground truth.

        Returns:
            Complete ValidationReport.
        """
        return self.validate(
            verdict=input_data.verdict,
            ground_truth=input_data.ground_truth,
            correlation_id=input_data.correlation_id,
            history=input_data.history,
        )

    def _execute_pipeline(
        self,
        verdict: RootCauseVerdict,
        ground_truth: GroundTruth,
        correlation_id: str,
        history: List[Dict[str, Any]] | None,
        pipeline_start: float,
    ) -> ValidationReport:
        """Execute the validation pipeline.

        Args:
            verdict: Root cause verdict.
            ground_truth: Simulation ground truth.
            correlation_id: Correlation ID.
            history: Historical prediction data.
            pipeline_start: Pipeline start timestamp.

        Returns:
            ValidationReport.
        """
        # Phase 1: Deterministic validation
        report = self._fallback.validate(
            verdict, ground_truth, correlation_id, history
        )

        # Phase 2: LLM discrepancy analysis (if enabled + incorrect)
        use_llm = (
            self._config.features.use_llm
            and self._analyzer is not None
            and not report.verdict_correct
        )

        if use_llm:
            report = self._llm_enhance(
                report, verdict, ground_truth, correlation_id
            )

        # Phase 3: Output validation
        if self._config.features.enable_validation:
            t0 = time.perf_counter()
            validation = self._validator.validate(report)
            val_ms = (time.perf_counter() - t0) * 1000
            self._telemetry.measure_value("output_validation", val_ms)

            if not validation.validation_passed:
                self._telemetry.output_validation_failures.inc()
                logger.warning(
                    f"Report validation failed: "
                    f"{len(validation.errors)} errors",
                    extra={"correlation_id": correlation_id},
                )

            report = report.model_copy(
                update={"output_validation": validation}
            )

        # Update pipeline latency
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        report = report.model_copy(
            update={"pipeline_latency_ms": round(pipeline_ms, 3)}
        )
        self._telemetry.measure_value("pipeline_total", pipeline_ms)

        logger.info(
            f"Validation complete: "
            f"correct={report.verdict_correct} "
            f"accuracy={report.accuracy_score:.2f} "
            f"in {pipeline_ms:.1f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
            },
        )

        return report

    def _llm_enhance(
        self,
        report: ValidationReport,
        verdict: RootCauseVerdict,
        ground_truth: GroundTruth,
        correlation_id: str,
    ) -> ValidationReport:
        """Enhance report with LLM discrepancy analysis.

        Args:
            report: Base deterministic report.
            verdict: Root cause verdict.
            ground_truth: Ground truth.
            correlation_id: Correlation ID.

        Returns:
            Updated ValidationReport with LLM recommendations.
        """
        assert self._analyzer is not None

        disc_summary = "; ".join(
            d.description[:100] for d in report.discrepancies[:5]
        )

        recommendations, cache_hit, success = self._analyzer.analyze(
            verdict_root_cause=verdict.root_cause,
            actual_root_cause=ground_truth.actual_root_cause,
            accuracy_score=report.accuracy_score,
            discrepancy_summary=disc_summary,
            correlation_id=correlation_id,
        )

        if not success:
            self._telemetry.fallback_triggers.inc()
            logger.info(
                "LLM analysis failed, keeping deterministic recommendations",
                extra={"correlation_id": correlation_id},
            )
            return report

        # Merge LLM recommendations with deterministic ones
        merged_recs = list(report.recommendations)
        for rec in recommendations:
            if rec and rec not in merged_recs:
                merged_recs.append(rec)

        # Update metadata
        metadata_update: Dict[str, Any] = {}
        if report.metadata:
            metadata_update = {
                "metadata": report.metadata.model_copy(update={
                    "used_llm": True,
                    "used_fallback": False,
                    "cache_hit": cache_hit,
                })
            }

        return report.model_copy(update={
            "recommendations": merged_recs,
            "classification_source": "llm" if not cache_hit else "cached",
            **metadata_update,
        })

    def _error_report(
        self, error_msg: str, pipeline_start: float
    ) -> ValidationReport:
        """Build a minimal error report.

        Args:
            error_msg: Error message.
            pipeline_start: Pipeline start timestamp.

        Returns:
            Minimal ValidationReport.
        """
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        return ValidationReport(
            verdict_correct=False,
            accuracy_score=0.0,
            recommendations=[f"Validation failed: {error_msg}"],
            classification_source="error",
            pipeline_latency_ms=round(pipeline_ms, 3),
        )

    @property
    def telemetry(self) -> TelemetryCollector:
        """Get the telemetry collector."""
        return self._telemetry

    @property
    def config(self) -> ValidationAgentConfig:
        """Get the agent configuration."""
        return self._config
