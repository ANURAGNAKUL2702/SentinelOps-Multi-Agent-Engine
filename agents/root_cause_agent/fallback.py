"""
File: fallback.py
Purpose: Deterministic fallback analysis when LLM is unavailable.
Dependencies: Core algorithms + schema.
Performance: <100ms total, zero network I/O.

Pure-algorithmic verdict using the 8 core algorithms.
Called when use_llm=False, circuit breaker is OPEN,
or LLM call fails and fallback_to_deterministic=True.
"""

from __future__ import annotations

import time
from typing import List, Optional

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.core.causal_chain_builder import CausalChainBuilder
from agents.root_cause_agent.core.confidence_calculator import ConfidenceCalculator
from agents.root_cause_agent.core.contradiction_resolver import ContradictionResolver
from agents.root_cause_agent.core.evidence_scorer import EvidenceScorer
from agents.root_cause_agent.core.evidence_synthesizer import EvidenceSynthesizer
from agents.root_cause_agent.core.impact_assessor import ImpactAssessor
from agents.root_cause_agent.core.timeline_reconstructor import TimelineReconstructor
from agents.root_cause_agent.core.verdict_ranker import VerdictRanker
from agents.root_cause_agent.llm.explanation_builder import ExplanationBuilder
from agents.root_cause_agent.schema import (
    CausalLink,
    Contradiction,
    Evidence,
    ImpactAssessment,
    IncidentCategory,
    RootCauseAgentInput,
    RootCauseVerdict,
    Severity,
    SynthesisResult,
    TimelineEvent,
    VerdictMetadata,
)
from agents.root_cause_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("root_cause_agent.fallback")


class DeterministicFallback:
    """Pure-algorithmic root cause verdict (no LLM).

    Pipeline::

        input → synthesize → score → confidence → chain
        → rank → timeline → impact → contradictions
        → explain → assemble verdict → validate

    Performance budget: <100ms total.

    Args:
        config: Agent configuration.
        telemetry: Optional telemetry collector.
    """

    def __init__(
        self,
        config: Optional[RootCauseAgentConfig] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._config = config or RootCauseAgentConfig()
        self._telemetry = telemetry

        # Initialise core algorithms
        self._synthesizer = EvidenceSynthesizer(self._config)
        self._scorer = EvidenceScorer(self._config)
        self._confidence_calc = ConfidenceCalculator(self._config)
        self._chain_builder = CausalChainBuilder(self._config)
        self._ranker = VerdictRanker(self._config)
        self._timeline = TimelineReconstructor(self._config)
        self._impact = ImpactAssessor(self._config)
        self._contradictions = ContradictionResolver(self._config)
        self._explainer = ExplanationBuilder(self._config)

    def analyze(
        self,
        input_data: RootCauseAgentInput,
        correlation_id: str = "",
    ) -> RootCauseVerdict:
        """Run deterministic analysis pipeline.

        Args:
            input_data: Root cause agent input.
            correlation_id: Request correlation ID.

        Returns:
            Fully populated RootCauseVerdict.
        """
        pipeline_start = time.perf_counter()

        # ── Phase 1: Evidence synthesis ─────────────────────────
        t0 = time.perf_counter()
        synthesis = self._synthesizer.synthesize(
            input_data, correlation_id
        )
        synthesis_ms = (time.perf_counter() - t0) * 1000

        if self._telemetry:
            self._telemetry.measure_value("evidence_synthesis", synthesis_ms)

        # ── Phase 2: Score evidence ─────────────────────────────
        t0 = time.perf_counter()
        scored_evidence = self._scorer.score_all(
            synthesis.evidence_trail,
            correlation_id=correlation_id,
        )
        scoring_ms = (time.perf_counter() - t0) * 1000

        # ── Phase 3: Confidence calculation ─────────────────────
        t0 = time.perf_counter()
        agent_confidences = [
            input_data.log_findings.confidence,
            input_data.metrics_findings.confidence,
            input_data.dependency_findings.confidence,
            input_data.hypothesis_findings.confidence,
        ]
        confidence = self._confidence_calc.calculate(
            synthesis, agent_confidences, correlation_id
        )
        confidence_ms = (time.perf_counter() - t0) * 1000

        if self._telemetry:
            self._telemetry.measure_value("confidence_calculation", confidence_ms)

        # ── Phase 4: Causal chain ───────────────────────────────
        t0 = time.perf_counter()
        causal_chain = self._chain_builder.build(
            input_data,
            primary_service=synthesis.primary_service,
            correlation_id=correlation_id,
        )
        chain_ms = (time.perf_counter() - t0) * 1000

        if self._telemetry:
            self._telemetry.measure_value("causal_chain", chain_ms)

        # ── Phase 5: Verdict ranking ────────────────────────────
        t0 = time.perf_counter()
        hypotheses = input_data.hypothesis_findings.ranked_hypotheses
        top_verdict, top_conf, alternatives = self._ranker.rank(
            hypotheses, synthesis, correlation_id
        )
        ranking_ms = (time.perf_counter() - t0) * 1000

        if self._telemetry:
            self._telemetry.measure_value("verdict_ranking", ranking_ms)

        # ── Phase 6: Timeline reconstruction ────────────────────
        t0 = time.perf_counter()
        timeline: List[TimelineEvent] = []
        if self._config.features.enable_timeline_reconstruction:
            timeline = self._timeline.reconstruct(
                input_data, correlation_id
            )
        timeline_ms = (time.perf_counter() - t0) * 1000

        if self._telemetry:
            self._telemetry.measure_value("timeline_reconstruction", timeline_ms)

        # ── Phase 7: Impact assessment ──────────────────────────
        t0 = time.perf_counter()
        impact: Optional[ImpactAssessment] = None
        if self._config.features.enable_impact_assessment:
            impact = self._impact.assess(
                input_data,
                root_cause_service=synthesis.primary_service,
                correlation_id=correlation_id,
            )
        impact_ms = (time.perf_counter() - t0) * 1000

        if self._telemetry:
            self._telemetry.measure_value("impact_assessment", impact_ms)

        # ── Phase 8: Contradiction resolution ───────────────────
        t0 = time.perf_counter()
        contradictions: List[Contradiction] = []
        if self._config.features.enable_contradiction_resolution:
            contradictions = self._contradictions.resolve(
                input_data, scored_evidence, correlation_id
            )
        contradiction_ms = (time.perf_counter() - t0) * 1000

        if self._telemetry:
            self._telemetry.measure_value("contradiction_resolution", contradiction_ms)

        # ── Phase 9: Build explanation ──────────────────────────
        reasoning = self._explainer.build(
            root_cause=top_verdict,
            confidence=confidence,
            evidence_trail=scored_evidence,
            causal_chain=causal_chain,
            impact=impact,
            contradictions=contradictions,
            timeline=timeline,
            correlation_id=correlation_id,
        )

        # ── Phase 10: Determine category and severity ───────────
        category = self._determine_category(input_data)
        severity = self._determine_severity(confidence, impact)
        mttr = self._estimate_mttr(input_data, severity)

        # ── Assemble verdict ────────────────────────────────────
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

        if self._telemetry:
            self._telemetry.measure_value("pipeline_total", pipeline_ms)
            self._telemetry.fallback_triggers.inc()

        affected_services = []
        if impact:
            affected_services = impact.affected_services

        metadata = VerdictMetadata(
            evidence_synthesis_ms=round(synthesis_ms, 2),
            confidence_calculation_ms=round(confidence_ms, 2),
            causal_chain_ms=round(chain_ms, 2),
            verdict_ranking_ms=round(ranking_ms, 2),
            timeline_reconstruction_ms=round(timeline_ms, 2),
            impact_assessment_ms=round(impact_ms, 2),
            contradiction_resolution_ms=round(contradiction_ms, 2),
            total_pipeline_ms=round(pipeline_ms, 2),
            used_llm=False,
            used_fallback=True,
            cache_hit=False,
        )

        verdict = RootCauseVerdict(
            root_cause=top_verdict,
            confidence=confidence,
            evidence_trail=scored_evidence,
            causal_chain=causal_chain,
            affected_services=affected_services,
            timeline=timeline,
            impact=impact,
            contradictions=contradictions,
            alternative_causes=alternatives,
            reasoning=reasoning,
            category=category,
            severity=severity,
            estimated_mttr_minutes=mttr,
            classification_source="deterministic",
            pipeline_latency_ms=round(pipeline_ms, 2),
            metadata=metadata,
        )

        logger.info(
            f"Fallback analysis complete: '{top_verdict[:50]}' "
            f"conf={confidence:.2f} in {pipeline_ms:.1f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "fallback",
            },
        )

        return verdict

    def _determine_category(
        self, input_data: RootCauseAgentInput
    ) -> str:
        """Determine incident category from hypothesis agent."""
        cat = input_data.hypothesis_findings.category
        try:
            return IncidentCategory(cat).value
        except (ValueError, KeyError):
            return IncidentCategory.UNKNOWN.value

    def _determine_severity(
        self,
        confidence: float,
        impact: Optional[ImpactAssessment],
    ) -> str:
        """Determine severity from confidence and impact."""
        if impact and impact.severity_score >= 0.9:
            return Severity.CRITICAL.value
        if impact and impact.severity_score >= 0.7:
            return Severity.HIGH.value

        if confidence >= 0.8:
            return Severity.HIGH.value
        if confidence >= 0.5:
            return Severity.MEDIUM.value
        return Severity.LOW.value

    def _estimate_mttr(
        self,
        input_data: RootCauseAgentInput,
        severity: str,
    ) -> float:
        """Estimate MTTR from hypothesis agent or severity."""
        hyp_mttr = input_data.hypothesis_findings.mttr_estimate
        if hyp_mttr > 0:
            return hyp_mttr

        mttr_map = {
            "critical": 60.0,
            "high": 30.0,
            "medium": 15.0,
            "low": 5.0,
        }
        return mttr_map.get(severity, 30.0)
