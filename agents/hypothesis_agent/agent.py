"""
File: agent.py
Purpose: Hybrid orchestrator — 3-phase pipeline (evidence+pattern → hypothesize → validate).
Dependencies: All hypothesis_agent sub-modules.
Performance: <2s end-to-end, <100ms deterministic fallback.

Coordinates the full hypothesis generation pipeline:
  Phase 1: Evidence aggregation + pattern matching (deterministic, <35ms)
  Phase 2: Hypothesis generation + causal reasoning (LLM or fallback, ~1900ms/<10ms)
  Phase 3: Ranking + validation + output assembly (deterministic, <30ms)

Graceful degradation: if LLM fails, falls back to deterministic rules.
Every call is traced via correlation ID.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.core.evidence_aggregator import (
    EvidenceAggregator,
)
from agents.hypothesis_agent.core.hypothesis_ranker import (
    HypothesisRanker,
)
from agents.hypothesis_agent.core.pattern_matcher import (
    PatternMatcher,
)
from agents.hypothesis_agent.core.validation_suggester import (
    ValidationSuggester,
)
from agents.hypothesis_agent.fallback import FallbackGenerator
from agents.hypothesis_agent.llm.causal_reasoner import (
    CausalReasoner,
)
from agents.hypothesis_agent.llm.hypothesis_refiner import (
    HypothesisRefiner,
)
from agents.hypothesis_agent.llm.theory_generator import (
    CircuitState,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
    TheoryGenerator,
)
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    Hypothesis,
    HypothesisAgentInput,
    HypothesisAgentOutput,
    PatternMatch,
    PipelineMetadata,
    ValidationResult,
)
from agents.hypothesis_agent.telemetry import (
    TelemetryCollector,
    get_logger,
)
from agents.hypothesis_agent.validator import OutputValidator

logger = get_logger("hypothesis_agent.agent")


class HypothesisAgent:
    """Hybrid hypothesis analysis agent — 60% deterministic, 40% LLM.

    Pipeline::

        Input → [Phase 1: Analyze] → [Phase 2: Hypothesize] → [Phase 3: Validate] → Output

    Phase 1 (deterministic, <35ms):
        - Evidence aggregation from all agent findings
        - Pattern matching against known failure library

    Phase 2 (hybrid):
        - LLM hypothesis generation (if enabled + circuit closed)
        - LLM causal chain construction
        - Fallback to deterministic rules on failure

    Phase 3 (deterministic, <30ms):
        - Hypothesis scoring, pruning, ranking
        - Validation test suggestion
        - 27-check validation
        - Output assembly

    Args:
        config: Agent configuration.
        llm_provider: LLM provider adapter (defaults to MockLLMProvider).

    Example::

        agent = HypothesisAgent(HypothesisAgentConfig())
        input_data = HypothesisAgentInput(
            log_findings=LogFindings(...),
            metric_findings=MetricFindings(...),
            dependency_findings=DependencyFindings(...),
        )
        output = agent.analyze(input_data)
    """

    def __init__(
        self,
        config: Optional[HypothesisAgentConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self._config = config or HypothesisAgentConfig()
        self._telemetry = TelemetryCollector()

        # ── layer instances ─────────────────────────────────────
        self._evidence_aggregator = EvidenceAggregator(self._config)
        self._pattern_matcher = PatternMatcher(self._config)
        self._ranker = HypothesisRanker(self._config)
        self._validation_suggester = ValidationSuggester(self._config)
        self._fallback = FallbackGenerator(self._config)
        self._refiner = HypothesisRefiner(self._config)
        self._validator = OutputValidator(self._config)

        # ── LLM modules (optional) ──────────────────────────────
        self._theory_generator: Optional[TheoryGenerator] = None
        self._causal_reasoner: Optional[CausalReasoner] = None

        provider = llm_provider or MockLLMProvider()

        if self._config.features.use_llm:
            if llm_provider is None:
                provider = self._resolve_provider()

            self._theory_generator = TheoryGenerator(
                config=self._config,
                provider=provider,
                telemetry=self._telemetry,
            )
            self._causal_reasoner = CausalReasoner(
                config=self._config,
                provider=provider,
                telemetry=self._telemetry,
            )
            self._refiner = HypothesisRefiner(
                config=self._config,
                provider=provider,
                telemetry=self._telemetry,
            )

        logger.info(
            f"HypothesisAgent initialized — "
            f"llm={'enabled' if self._config.features.use_llm else 'disabled'}, "
            f"fallback={'enabled' if self._config.features.fallback_to_rules else 'disabled'}, "
            f"pattern_matching={'enabled' if self._config.features.enable_pattern_matching else 'disabled'}, "
            f"validation={'enabled' if self._config.features.enable_validation else 'disabled'}",
        )

    # ─── provider resolution ────────────────────────────────────

    def _resolve_provider(self) -> LLMProvider:
        """Auto-detect LLM provider from config."""
        provider_name = self._config.llm.provider.lower()

        if provider_name == "groq":
            try:
                from agents.hypothesis_agent.llm.groq_provider import (
                    GroqProvider,
                )
                return GroqProvider(
                    api_key=self._config.llm.api_key or None,
                    model=self._config.llm.model,
                )
            except ImportError:
                logger.warning(
                    "GroqProvider not available, using MockLLMProvider"
                )

        return MockLLMProvider()

    # ─── public API ─────────────────────────────────────────────

    def analyze(
        self,
        input_data: HypothesisAgentInput,
        correlation_id: str = "",
    ) -> HypothesisAgentOutput:
        """Run the full hybrid analysis pipeline.

        Args:
            input_data: Validated hypothesis agent input.
            correlation_id: Optional correlation ID.

        Returns:
            HypothesisAgentOutput with all fields populated.
        """
        cid = correlation_id or input_data.correlation_id
        pipeline_start = time.perf_counter()
        self._telemetry.analyses_total.inc()

        logger.info(
            "Pipeline started",
            extra={
                "correlation_id": cid,
                "layer": "pipeline",
                "context": {
                    "incident_id": input_data.incident_id,
                    "time_window": input_data.time_window,
                },
            },
        )

        try:
            # ── Phase 1: Deterministic analysis ─────────────────
            evidence, pattern_matches, phase1_times = (
                self._phase1_analyze(input_data, cid)
            )

            # ── Phase 2: Hypothesis generation ──────────────────
            hypotheses, used_llm, used_fallback, phase2_times = (
                self._phase2_hypothesize(
                    evidence, pattern_matches, cid
                )
            )

            # ── Phase 3: Rank + validate + assemble ─────────────
            pipeline_elapsed = (
                (time.perf_counter() - pipeline_start) * 1000
            )
            output = self._phase3_validate_and_assemble(
                hypotheses=hypotheses,
                evidence=evidence,
                pattern_matches=pattern_matches,
                input_data=input_data,
                used_llm=used_llm,
                used_fallback=used_fallback,
                pipeline_latency_ms=pipeline_elapsed,
                phase1_times=phase1_times,
                phase2_times=phase2_times,
                correlation_id=cid,
            )

            self._telemetry.analyses_succeeded.inc()
            self._telemetry.measure_value(
                "pipeline_total", pipeline_elapsed
            )

            logger.info(
                f"Pipeline completed — "
                f"confidence={output.confidence_score}, "
                f"source={output.classification_source}, "
                f"hypotheses={len(output.hypotheses)}, "
                f"{pipeline_elapsed:.2f}ms",
                extra={
                    "correlation_id": cid,
                    "layer": "pipeline",
                    "context": {
                        "confidence": output.confidence_score,
                        "source": output.classification_source,
                        "hypotheses": len(output.hypotheses),
                        "latency_ms": round(pipeline_elapsed, 2),
                    },
                },
            )

            return output

        except Exception as e:
            self._telemetry.analyses_failed.inc()
            logger.error(
                f"Pipeline failed: {e}",
                extra={
                    "correlation_id": cid,
                    "layer": "pipeline",
                    "context": {"error": str(e)},
                },
                exc_info=True,
            )
            raise

    def health_check(self) -> Dict[str, Any]:
        """Return agent health status."""
        circuit_state = "n/a"
        if self._theory_generator:
            circuit_state = (
                self._theory_generator.circuit_state.value
            )

        components = {
            "evidence_aggregator": "healthy",
            "pattern_matcher": "healthy",
            "hypothesis_ranker": "healthy",
            "validation_suggester": "healthy",
            "fallback_generator": "healthy",
            "validator": "healthy",
            "theory_generator": (
                "healthy"
                if circuit_state in ("closed", "half_open", "n/a")
                else "degraded"
            ),
        }

        overall = "healthy"
        if circuit_state == "open":
            overall = "degraded"

        snap = self._telemetry.snapshot()

        return {
            "agent": "hypothesis_agent",
            "status": overall,
            "circuit_breaker": circuit_state,
            "components": components,
            "metrics": snap["counters"],
        }

    @property
    def telemetry(self) -> TelemetryCollector:
        """Access the telemetry collector."""
        return self._telemetry

    # ─── Phase 1: Deterministic Analysis ────────────────────────

    def _phase1_analyze(
        self,
        input_data: HypothesisAgentInput,
        correlation_id: str,
    ) -> tuple:
        """Phase 1: Evidence aggregation + pattern matching.

        Returns:
            (evidence, pattern_matches, phase1_times)
        """
        times: Dict[str, float] = {}

        # Evidence aggregation
        start = time.perf_counter()
        with self._telemetry.measure("evidence_aggregation"):
            evidence = self._evidence_aggregator.aggregate(
                input_data, correlation_id
            )
        times["evidence_aggregation"] = (
            (time.perf_counter() - start) * 1000
        )

        # Pattern matching
        pattern_matches: List[PatternMatch] = []
        start = time.perf_counter()
        if self._config.features.enable_pattern_matching:
            with self._telemetry.measure("pattern_matching"):
                pattern_matches = self._pattern_matcher.match(
                    evidence, correlation_id
                )
        times["pattern_matching"] = (
            (time.perf_counter() - start) * 1000
        )

        logger.info(
            f"Phase 1 complete — "
            f"{evidence.total_evidence_count} evidence, "
            f"{len(pattern_matches)} patterns, "
            f"{sum(times.values()):.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
            },
        )

        return evidence, pattern_matches, times

    # ─── Phase 2: Hypothesis Generation ─────────────────────────

    def _phase2_hypothesize(
        self,
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
        correlation_id: str,
    ) -> tuple:
        """Phase 2: Generate hypotheses via LLM or fallback.

        Returns:
            (hypotheses, used_llm, used_fallback, phase2_times)
        """
        times: Dict[str, float] = {}
        used_llm = False
        used_fallback = False
        hypotheses: List[Hypothesis] = []

        # Try LLM generation
        if (
            self._config.features.use_llm
            and self._theory_generator
        ):
            start = time.perf_counter()
            try:
                with self._telemetry.measure(
                    "hypothesis_generation"
                ):
                    hypotheses = self._theory_generator.generate(
                        evidence=evidence,
                        pattern_matches=pattern_matches,
                        correlation_id=correlation_id,
                    )
                used_llm = True
                times["hypothesis_generation"] = (
                    (time.perf_counter() - start) * 1000
                )

                # Causal reasoning enhancement
                if (
                    self._causal_reasoner
                    and self._config.features.enable_causal_reasoning
                ):
                    cr_start = time.perf_counter()
                    with self._telemetry.measure(
                        "causal_reasoning"
                    ):
                        hypotheses = (
                            self._causal_reasoner.enhance_chains(
                                hypotheses=hypotheses,
                                evidence=evidence,
                                correlation_id=correlation_id,
                            )
                        )
                    times["causal_reasoning"] = (
                        (time.perf_counter() - cr_start) * 1000
                    )

            except LLMProviderError as e:
                logger.warning(
                    f"LLM failed, falling back: {e}",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "pipeline",
                    },
                )
                times["hypothesis_generation"] = (
                    (time.perf_counter() - start) * 1000
                )
                # Fall through to fallback

        # Fallback if needed
        if not hypotheses:
            start = time.perf_counter()
            self._telemetry.fallback_triggers.inc()
            hypotheses = self._fallback.generate(
                evidence=evidence,
                pattern_matches=pattern_matches,
                correlation_id=correlation_id,
            )
            used_fallback = True
            times["fallback_generation"] = (
                (time.perf_counter() - start) * 1000
            )

        logger.info(
            f"Phase 2 complete — "
            f"{len(hypotheses)} hypotheses, "
            f"llm={used_llm}, fallback={used_fallback}, "
            f"{sum(times.values()):.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
            },
        )

        return hypotheses, used_llm, used_fallback, times

    # ─── Phase 3: Validate + Assemble ──────────────────────────

    def _phase3_validate_and_assemble(
        self,
        hypotheses: List[Hypothesis],
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
        input_data: HypothesisAgentInput,
        used_llm: bool,
        used_fallback: bool,
        pipeline_latency_ms: float,
        phase1_times: Dict[str, float],
        phase2_times: Dict[str, float],
        correlation_id: str,
    ) -> HypothesisAgentOutput:
        """Phase 3: Rank, validate, assemble output."""

        # ── Ranking ─────────────────────────────────────────────
        ranking_start = time.perf_counter()
        with self._telemetry.measure("ranking"):
            ranked = self._ranker.rank(
                hypotheses=hypotheses,
                evidence=evidence,
                pattern_matches=pattern_matches,
                historical=input_data.historical_context,
                correlation_id=correlation_id,
            )
        ranking_ms = (time.perf_counter() - ranking_start) * 1000

        # ── Validation suggestion ───────────────────────────────
        ranked = self._validation_suggester.suggest(
            hypotheses=ranked,
            pattern_matches=pattern_matches,
            correlation_id=correlation_id,
        )

        # ── Compute confidence ──────────────────────────────────
        confidence = self._ranker.compute_overall_confidence(
            ranked, evidence, pattern_matches
        )

        # ── Summary generation ──────────────────────────────────
        summary = self._refiner.generate_summary(ranked)
        category = self._refiner.determine_category(ranked)
        severity = self._refiner.determine_severity(ranked)

        # ── Determine classification source ─────────────────────
        if used_llm:
            source = "llm"
        elif used_fallback:
            source = "fallback"
        else:
            source = "deterministic"

        # ── MTTR from top hypothesis ────────────────────────────
        mttr = 30.0
        if ranked:
            mttr = ranked[0].estimated_mttr_minutes

        # ── Recommended hypothesis ──────────────────────────────
        recommended_id = ""
        if ranked:
            recommended_id = ranked[0].hypothesis_id

        # ── Assemble output ─────────────────────────────────────
        output = HypothesisAgentOutput(
            time_window=input_data.time_window,
            incident_id=input_data.incident_id,
            hypotheses=ranked,
            confidence_score=confidence,
            pattern_matches=pattern_matches,
            recommended_hypothesis=recommended_id,
            hypothesis_summary=summary,
            estimated_mttr_minutes=mttr,
            category=category,
            severity=severity,
            correlation_id=correlation_id,
            classification_source=source,
            pipeline_latency_ms=round(pipeline_latency_ms, 2),
            metadata=PipelineMetadata(
                evidence_aggregation_time_ms=round(
                    phase1_times.get("evidence_aggregation", 0.0), 2
                ),
                pattern_matching_time_ms=round(
                    phase1_times.get("pattern_matching", 0.0), 2
                ),
                hypothesis_generation_time_ms=round(
                    phase2_times.get(
                        "hypothesis_generation",
                        phase2_times.get("fallback_generation", 0.0),
                    ),
                    2,
                ),
                causal_reasoning_time_ms=round(
                    phase2_times.get("causal_reasoning", 0.0), 2
                ),
                ranking_time_ms=round(ranking_ms, 2),
                validation_time_ms=0.0,  # filled below
                total_time_ms=round(pipeline_latency_ms, 2),
                used_llm=used_llm,
                used_fallback=used_fallback,
                cache_hit=False,
                correlation_id=correlation_id,
            ),
        )

        # ── Validation ──────────────────────────────────────────
        if self._config.features.enable_validation:
            val_start = time.perf_counter()
            with self._telemetry.measure("validation"):
                validation = self._validator.validate(
                    output, correlation_id
                )
            val_ms = (time.perf_counter() - val_start) * 1000

            if not validation.validation_passed:
                self._telemetry.validation_failures.inc()

            # Update output with validation
            output = output.model_copy(update={
                "validation": validation,
            })
            if output.metadata:
                output = output.model_copy(update={
                    "metadata": output.metadata.model_copy(update={
                        "validation_time_ms": round(val_ms, 2),
                    }),
                })

        return output
