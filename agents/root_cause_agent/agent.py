"""
File: agent.py
Purpose: Root Cause Agent — the orchestration layer.
Dependencies: All submodules.
Performance: <100ms fallback, <2s LLM.

3-phase pipeline:
  Phase 1: Evidence synthesis + contradiction resolution.
  Phase 2: LLM verdict (or fallback).
  Phase 3: Validation + assembly.

Entry point::

    agent = RootCauseAgent()
    verdict = agent.analyze(input_data)
"""

from __future__ import annotations

import time
from typing import Optional

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.fallback import DeterministicFallback
from agents.root_cause_agent.llm.verdict_generator import VerdictGenerator
from agents.root_cause_agent.schema import (
    IncidentCategory,
    RootCauseAgentInput,
    RootCauseVerdict,
    Severity,
    ValidationResult,
    VerdictMetadata,
)
from agents.root_cause_agent.telemetry import TelemetryCollector, get_logger
from agents.root_cause_agent.validator import VerdictValidator

logger = get_logger("root_cause_agent.agent")


class RootCauseAgent:
    """Root Cause Agent — synthesis layer.

    Combines outputs from 4 upstream agents (log, metrics,
    dependency, hypothesis) into a final RootCauseVerdict
    with confidence scoring, evidence trail, causal chain,
    timeline, impact assessment, and contradiction resolution.

    Pipeline::

        Phase 1: Evidence synthesis + scoring + contradictions
        Phase 2: LLM verdict (or deterministic fallback)
        Phase 3: Validation + assembly

    Args:
        config: Agent configuration.

    Example::

        agent = RootCauseAgent()
        verdict = agent.analyze(input_data)
        print(verdict.root_cause, verdict.confidence)
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()
        self._telemetry = TelemetryCollector()
        self._fallback = DeterministicFallback(
            self._config, self._telemetry
        )
        self._validator = VerdictValidator(self._config)

        # LLM verdict generator (only created if enabled)
        self._verdict_gen: Optional[VerdictGenerator] = None
        if self._config.features.use_llm:
            self._verdict_gen = VerdictGenerator(
                self._config, self._telemetry
            )

    def analyze(
        self,
        input_data: RootCauseAgentInput,
    ) -> RootCauseVerdict:
        """Run the full root cause analysis pipeline.

        Args:
            input_data: Findings from all 4 upstream agents.

        Returns:
            Complete RootCauseVerdict.
        """
        pipeline_start = time.perf_counter()
        correlation_id = input_data.correlation_id or input_data.incident_id

        self._telemetry.analyses_total.inc()

        logger.info(
            f"Root cause analysis started — incident={input_data.incident_id}",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
            },
        )

        try:
            verdict = self._execute_pipeline(
                input_data, correlation_id, pipeline_start
            )
            self._telemetry.analyses_succeeded.inc()
            return verdict

        except Exception as exc:
            self._telemetry.analyses_failed.inc()
            logger.error(
                f"Pipeline failed: {exc}",
                extra={
                    "correlation_id": correlation_id,
                    "layer": "pipeline",
                },
            )
            # Return a minimal error verdict
            return self._error_verdict(str(exc), pipeline_start)

    def _execute_pipeline(
        self,
        input_data: RootCauseAgentInput,
        correlation_id: str,
        pipeline_start: float,
    ) -> RootCauseVerdict:
        """Execute the 3-phase pipeline.

        Args:
            input_data: Agent input.
            correlation_id: Correlation ID.
            pipeline_start: Pipeline start timestamp.

        Returns:
            Root cause verdict.
        """
        # ── Decide: LLM or fallback? ───────────────────────────
        use_llm = (
            self._config.features.use_llm
            and self._verdict_gen is not None
        )

        if use_llm:
            verdict = self._llm_pipeline(
                input_data, correlation_id, pipeline_start
            )
        else:
            verdict = self._fallback.analyze(
                input_data, correlation_id
            )

        # ── Phase 3: Validation ─────────────────────────────────
        if self._config.features.enable_validation:
            t0 = time.perf_counter()
            validation = self._validator.validate(verdict)
            validation_ms = (time.perf_counter() - t0) * 1000

            if self._telemetry:
                self._telemetry.measure_value("validation", validation_ms)

            if not validation.validation_passed:
                self._telemetry.validation_failures.inc()
                logger.warning(
                    f"Verdict validation failed: "
                    f"{len(validation.errors)} errors",
                    extra={"correlation_id": correlation_id},
                )

            verdict = verdict.model_copy(
                update={"validation": validation}
            )

        # ── Update pipeline latency ─────────────────────────────
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        verdict = verdict.model_copy(
            update={"pipeline_latency_ms": round(pipeline_ms, 2)}
        )

        if self._telemetry:
            self._telemetry.measure_value("pipeline_total", pipeline_ms)

        logger.info(
            f"Root cause analysis complete: "
            f"'{verdict.root_cause[:50]}' "
            f"conf={verdict.confidence:.2f} "
            f"in {pipeline_ms:.1f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
            },
        )

        return verdict

    def _llm_pipeline(
        self,
        input_data: RootCauseAgentInput,
        correlation_id: str,
        pipeline_start: float,
    ) -> RootCauseVerdict:
        """Run the LLM-augmented pipeline.

        Falls back to deterministic if LLM fails.

        Args:
            input_data: Agent input.
            correlation_id: Correlation ID.
            pipeline_start: Pipeline start timestamp.

        Returns:
            Root cause verdict.
        """
        # Run fallback first to get all deterministic components
        det_verdict = self._fallback.analyze(
            input_data, correlation_id
        )

        # Try LLM verdict generation
        assert self._verdict_gen is not None
        from agents.root_cause_agent.core.evidence_synthesizer import EvidenceSynthesizer

        synthesizer = EvidenceSynthesizer(self._config)
        synthesis = synthesizer.synthesize(input_data, correlation_id)

        llm_result, cache_hit, success = self._verdict_gen.generate(
            synthesis, correlation_id
        )

        if not success:
            if self._config.features.fallback_to_deterministic:
                logger.info(
                    "LLM failed, using deterministic fallback",
                    extra={"correlation_id": correlation_id},
                )
                return det_verdict
            return det_verdict

        # Merge LLM result into deterministic verdict
        root_cause = llm_result.get("root_cause", det_verdict.root_cause)
        reasoning = llm_result.get("reasoning", det_verdict.reasoning)
        category = llm_result.get("category", det_verdict.category)
        severity = llm_result.get("severity", det_verdict.severity)
        llm_confidence = llm_result.get("confidence", det_verdict.confidence)
        mttr = llm_result.get(
            "estimated_mttr_minutes", det_verdict.estimated_mttr_minutes
        )

        # Average LLM and deterministic confidence
        merged_confidence = (llm_confidence + det_verdict.confidence) / 2.0

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

        metadata = det_verdict.metadata.model_copy(update={
            "used_llm": True,
            "used_fallback": False,
            "cache_hit": cache_hit,
        }) if det_verdict.metadata else VerdictMetadata(
            used_llm=True, used_fallback=False, cache_hit=cache_hit,
        )

        return det_verdict.model_copy(update={
            "root_cause": root_cause,
            "confidence": round(merged_confidence, 4),
            "reasoning": reasoning,
            "category": category,
            "severity": severity,
            "estimated_mttr_minutes": mttr,
            "classification_source": "llm",
            "pipeline_latency_ms": round(pipeline_ms, 2),
            "metadata": metadata,
        })

    def _error_verdict(
        self, error_msg: str, pipeline_start: float
    ) -> RootCauseVerdict:
        """Build a minimal error verdict.

        Args:
            error_msg: Error message.
            pipeline_start: Pipeline start timestamp.

        Returns:
            Minimal RootCauseVerdict with error info.
        """
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
        return RootCauseVerdict(
            root_cause=f"Analysis failed: {error_msg}",
            confidence=0.0,
            reasoning=f"Pipeline encountered an error: {error_msg}",
            category=IncidentCategory.UNKNOWN.value,
            severity=Severity.LOW.value,
            classification_source="error",
            pipeline_latency_ms=round(pipeline_ms, 2),
            metadata=VerdictMetadata(
                used_llm=False,
                used_fallback=False,
                cache_hit=False,
            ),
        )

    @property
    def telemetry(self) -> TelemetryCollector:
        """Get the telemetry collector."""
        return self._telemetry

    @property
    def config(self) -> RootCauseAgentConfig:
        """Get the agent configuration."""
        return self._config
