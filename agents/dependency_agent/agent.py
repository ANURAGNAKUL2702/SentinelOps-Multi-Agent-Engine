"""
File: agent.py
Purpose: Hybrid orchestrator — 3-phase pipeline (graph+trace+impact → classify → validate).
Dependencies: All dependency_agent sub-modules.
Performance: <2s end-to-end, <200ms deterministic path.

Coordinates the full dependency analysis pipeline:
  Phase 1: Deterministic graph build + trace analysis + impact + bottleneck
  Phase 2: Classification (LLM with fallback to rule engine)
  Phase 3: Validation + output assembly

Graceful degradation: if LLM fails, falls back to deterministic rules.
Every call is traced via correlation ID.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.core.bottleneck_detector import (
    BottleneckDetector,
)
from agents.dependency_agent.core.graph_builder import GraphBuilder
from agents.dependency_agent.core.impact_calculator import (
    ImpactCalculator,
)
from agents.dependency_agent.core.trace_analyzer import TraceAnalyzer
from agents.dependency_agent.fallback import FallbackClassifier
from agents.dependency_agent.llm.classifier import (
    CircuitState,
    LLMClassifier,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
)
from agents.dependency_agent.llm.synthesizer import Synthesizer
from agents.dependency_agent.schema import (
    BottleneckDetectionResult,
    ClassificationResult,
    DependencyAgentOutput,
    DependencyAnalysisInput,
    GraphBuildResult,
    ImpactAnalysisResult,
    TraceAnalysisResult,
    ValidationResult,
)
from agents.dependency_agent.telemetry import (
    TelemetryCollector,
    get_logger,
)
from agents.dependency_agent.validator import OutputValidator

logger = get_logger("dependency_agent.agent")


class DependencyAgent:
    """Hybrid dependency analysis agent — 80% deterministic, 20% LLM.

    Pipeline::

        Input → [Phase 1: Analyze] → [Phase 2: Classify] → [Phase 3: Validate] → Output

    Phase 1 (deterministic, <200ms):
        - Graph build (adjacency, cycles, depth)
        - Trace analysis (critical path, slow spans)
        - Impact calculation (blast radius, criticality, SPOF)
        - Bottleneck detection (fan-in, fan-out, sequential)

    Phase 2 (hybrid):
        - LLM classification (if enabled + circuit closed)
        - Fallback to deterministic rule engine on failure

    Phase 3 (deterministic, <5ms):
        - 25-check validation
        - Output assembly per DependencyAgentOutput schema

    Args:
        config: Agent configuration.
        llm_provider: LLM provider adapter (defaults to MockLLMProvider).

    Example::

        agent = DependencyAgent(DependencyAgentConfig())
        input_data = DependencyAnalysisInput(
            service_graph=ServiceGraph(nodes=[...], edges=[...]),
        )
        output = agent.analyze(input_data)
    """

    def __init__(
        self,
        config: Optional[DependencyAgentConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self._config = config or DependencyAgentConfig()
        self._telemetry = TelemetryCollector()

        # ── layer instances ─────────────────────────────────────
        self._graph_builder = GraphBuilder(self._config)
        self._trace_analyzer = TraceAnalyzer(self._config)
        self._impact_calculator = ImpactCalculator(self._config)
        self._bottleneck_detector = BottleneckDetector(self._config)
        self._fallback = FallbackClassifier(self._config)
        self._synthesizer = Synthesizer(self._config)
        self._validator = OutputValidator(self._config)

        # ── LLM classifier (optional) ──────────────────────────
        self._llm_classifier: Optional[LLMClassifier] = None
        if self._config.features.use_llm:
            provider = llm_provider or self._resolve_provider()
            self._llm_classifier = LLMClassifier(
                config=self._config,
                provider=provider,
                telemetry=self._telemetry,
            )

        logger.info(
            f"DependencyAgent initialized — "
            f"llm={'enabled' if self._config.features.use_llm else 'disabled'}, "
            f"fallback={'enabled' if self._config.features.fallback_to_rules else 'disabled'}, "
            f"trace={'enabled' if self._config.features.enable_trace_analysis else 'disabled'}, "
            f"bottleneck={'enabled' if self._config.features.enable_bottleneck_detection else 'disabled'}, "
            f"validation={'enabled' if self._config.features.enable_validation else 'disabled'}",
        )

    # ─── provider resolution ────────────────────────────────────

    def _resolve_provider(self) -> LLMProvider:
        """Auto-detect LLM provider from config.

        Returns:
            Configured LLMProvider instance.
        """
        provider_name = self._config.llm.provider.lower()

        if provider_name == "groq":
            try:
                from agents.dependency_agent.llm.groq_provider import (
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

        logger.info(
            f"Using MockLLMProvider (provider='{provider_name}')"
        )
        return MockLLMProvider()

    # ─── public API ─────────────────────────────────────────────

    def analyze(
        self,
        input_data: DependencyAnalysisInput,
        correlation_id: str = "",
    ) -> DependencyAgentOutput:
        """Run the full hybrid analysis pipeline.

        Args:
            input_data: Validated dependency analysis input.
            correlation_id: Optional correlation ID.

        Returns:
            DependencyAgentOutput with all fields populated.
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
                    "node_count": len(
                        input_data.service_graph.nodes
                    ),
                    "edge_count": len(
                        input_data.service_graph.edges
                    ),
                },
            },
        )

        try:
            # ── Phase 1: Deterministic analysis ─────────────────
            (
                graph_result,
                trace_result,
                impact,
                bottleneck_result,
                phase1_times,
            ) = self._phase1_analyze(input_data, cid)

            # ── Phase 2: Classification ─────────────────────────
            classification, classification_ms = (
                self._phase2_classify(
                    input_data,
                    graph_result,
                    impact,
                    trace_result,
                    bottleneck_result,
                    cid,
                )
            )

            # ── Phase 3: Validate + assemble ────────────────────
            pipeline_elapsed = (
                (time.perf_counter() - pipeline_start) * 1000
            )
            output = self._phase3_validate_and_assemble(
                classification=classification,
                input_data=input_data,
                pipeline_latency_ms=pipeline_elapsed,
                correlation_id=cid,
                graph_build_time_ms=phase1_times.get(
                    "graph_build", 0.0
                ),
                trace_analysis_time_ms=phase1_times.get(
                    "trace_analysis", 0.0
                ),
                impact_calculation_time_ms=phase1_times.get(
                    "impact_calculation", 0.0
                ),
                classification_time_ms=classification_ms,
            )

            self._telemetry.analyses_succeeded.inc()
            self._telemetry.measure_value(
                "pipeline_total", pipeline_elapsed
            )

            logger.info(
                f"Pipeline completed — "
                f"confidence={output.confidence_score}, "
                f"source={output.classification_source}, "
                f"{pipeline_elapsed:.2f}ms",
                extra={
                    "correlation_id": cid,
                    "layer": "pipeline",
                    "context": {
                        "confidence": output.confidence_score,
                        "source": output.classification_source,
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
        """Return agent health status.

        Returns:
            Dict with component health, circuit state, metrics.
        """
        circuit_state = "n/a"
        if self._llm_classifier:
            circuit_state = self._llm_classifier.circuit_state.value

        components = {
            "graph_builder": "healthy",
            "trace_analyzer": "healthy",
            "impact_calculator": "healthy",
            "bottleneck_detector": "healthy",
            "fallback_classifier": "healthy",
            "validator": "healthy",
            "llm_classifier": (
                "healthy"
                if circuit_state
                in ("closed", "half_open", "n/a")
                else "degraded"
            ),
        }

        overall = "healthy"
        if circuit_state == "open":
            overall = "degraded"

        return {
            "status": overall,
            "agent": "dependency_agent",
            "config": {
                "llm_enabled": self._config.features.use_llm,
                "fallback_enabled": (
                    self._config.features.fallback_to_rules
                ),
                "circuit_state": circuit_state,
            },
            "components": components,
            "metrics": self._telemetry.snapshot(),
        }

    @property
    def telemetry(self) -> TelemetryCollector:
        """Access the telemetry collector."""
        return self._telemetry

    # ─── Phase 1: Deterministic Analysis ────────────────────────

    def _phase1_analyze(
        self,
        input_data: DependencyAnalysisInput,
        correlation_id: str,
    ) -> tuple[
        GraphBuildResult,
        Optional[TraceAnalysisResult],
        Dict[str, ImpactAnalysisResult],
        Optional[BottleneckDetectionResult],
        Dict[str, float],
    ]:
        """Phase 1: Graph build, trace analysis, impact, bottleneck.

        Fully deterministic. Same input → same output.
        Target: <200ms for 1000 services.

        Args:
            input_data: Validated input.
            correlation_id: Request correlation ID.

        Returns:
            Tuple of (graph_result, trace_result, impact, bottleneck, timings).
        """
        timings: Dict[str, float] = {}

        # ── graph build ─────────────────────────────────────────
        gb_start = time.perf_counter()
        with self._telemetry.measure(
            "graph_build", correlation_id
        ):
            graph_result = self._graph_builder.build(
                input_data, correlation_id
            )
        timings["graph_build"] = (
            (time.perf_counter() - gb_start) * 1000
        )

        # ── trace analysis ──────────────────────────────────────
        trace_result: Optional[TraceAnalysisResult] = None
        if self._config.features.enable_trace_analysis:
            ta_start = time.perf_counter()
            with self._telemetry.measure(
                "trace_analysis", correlation_id
            ):
                trace_result = self._trace_analyzer.analyze(
                    input_data, correlation_id
                )
            timings["trace_analysis"] = (
                (time.perf_counter() - ta_start) * 1000
            )

        # ── impact calculation ──────────────────────────────────
        ic_start = time.perf_counter()
        critical_path = (
            trace_result.critical_path
            if trace_result
            else None
        )
        with self._telemetry.measure(
            "impact_calculation", correlation_id
        ):
            impact = self._impact_calculator.calculate(
                input_data=input_data,
                graph=graph_result.graph,
                critical_path=critical_path,
                correlation_id=correlation_id,
            )
        timings["impact_calculation"] = (
            (time.perf_counter() - ic_start) * 1000
        )

        # ── bottleneck detection ────────────────────────────────
        bottleneck_result: Optional[BottleneckDetectionResult] = None
        if self._config.features.enable_bottleneck_detection:
            bd_start = time.perf_counter()
            spans = None
            if input_data.traces:
                spans = input_data.traces[0].spans

            with self._telemetry.measure(
                "bottleneck_detection", correlation_id
            ):
                bottleneck_result = self._bottleneck_detector.detect(
                    graph=graph_result.graph,
                    critical_path=critical_path,
                    spans=spans,
                    correlation_id=correlation_id,
                )
            timings["bottleneck_detection"] = (
                (time.perf_counter() - bd_start) * 1000
            )

        total_ms = sum(timings.values())
        logger.debug(
            f"Phase 1 complete — "
            f"{len(input_data.service_graph.nodes)} services, "
            f"{total_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "pipeline",
                "context": {
                    "timings": {
                        k: round(v, 2) for k, v in timings.items()
                    },
                },
            },
        )

        return (
            graph_result,
            trace_result,
            impact,
            bottleneck_result,
            timings,
        )

    # ─── Phase 2: Classification ────────────────────────────────

    def _phase2_classify(
        self,
        input_data: DependencyAnalysisInput,
        graph_result: GraphBuildResult,
        impact: Dict[str, ImpactAnalysisResult],
        trace_result: Optional[TraceAnalysisResult],
        bottleneck_result: Optional[BottleneckDetectionResult],
        correlation_id: str,
    ) -> tuple[ClassificationResult, float]:
        """Phase 2: Classify using LLM or fallback.

        Args:
            input_data: Original input.
            graph_result: Graph build result.
            impact: Per-service impact.
            trace_result: Trace analysis result.
            bottleneck_result: Bottleneck detection result.
            correlation_id: Request correlation ID.

        Returns:
            Tuple of (ClassificationResult, classification_ms).
        """
        classify_start = time.perf_counter()

        with self._telemetry.measure(
            "classification", correlation_id
        ):
            # ── try LLM path ────────────────────────────────────
            if (
                self._config.features.use_llm
                and self._llm_classifier is not None
            ):
                try:
                    signals = self._build_llm_signals(
                        graph_result, impact,
                        trace_result, bottleneck_result,
                    )
                    result = self._llm_classifier.classify(
                        signals=signals,
                        input_data=input_data,
                        correlation_id=correlation_id,
                    )
                    elapsed = (
                        (time.perf_counter() - classify_start)
                        * 1000
                    )
                    return result, elapsed

                except LLMProviderError as e:
                    logger.warning(
                        f"LLM classification failed: {e}",
                        extra={
                            "correlation_id": correlation_id,
                            "layer": "classification",
                        },
                    )
                    if not self._config.features.fallback_to_rules:
                        raise
                    self._telemetry.record_fallback(correlation_id)

            # ── fallback / default: rule engine ─────────────────
            result = self._fallback.classify(
                input_data=input_data,
                graph_result=graph_result,
                impact=impact,
                trace_result=trace_result,
                bottleneck_result=bottleneck_result,
                correlation_id=correlation_id,
            )

        elapsed = (
            (time.perf_counter() - classify_start) * 1000
        )
        return result, elapsed

    def _build_llm_signals(
        self,
        graph_result: GraphBuildResult,
        impact: Dict[str, ImpactAnalysisResult],
        trace_result: Optional[TraceAnalysisResult],
        bottleneck_result: Optional[BottleneckDetectionResult],
    ) -> Dict[str, Any]:
        """Build signals dict for LLM classification.

        Args:
            graph_result: Graph build result.
            impact: Per-service impact.
            trace_result: Trace analysis.
            bottleneck_result: Bottleneck detection.

        Returns:
            Dict of pre-calculated signals.
        """
        signals: Dict[str, Any] = {
            "total_services": graph_result.total_services,
            "total_dependencies": graph_result.total_dependencies,
            "has_cycles": graph_result.has_cycles,
            "cycle_paths": graph_result.cycle_paths,
            "max_depth": graph_result.max_depth,
        }

        if impact:
            signals["impact_summary"] = {
                svc: {
                    "criticality_score": imp.criticality_score,
                    "is_spof": imp.is_single_point_of_failure,
                    "blast_radius": (
                        imp.blast_radius.total_affected_count
                    ),
                }
                for svc, imp in impact.items()
            }

        if trace_result and trace_result.critical_path:
            cp = trace_result.critical_path
            signals["critical_path"] = {
                "path": cp.path,
                "total_duration_ms": cp.total_duration_ms,
                "bottleneck_service": cp.bottleneck_service,
                "bottleneck_percentage": cp.bottleneck_percentage,
            }
            signals["slow_span_count"] = len(
                trace_result.slow_spans
            )

        if bottleneck_result:
            signals["bottlenecks"] = [
                {
                    "service": bn.service_name,
                    "type": bn.bottleneck_type.value
                    if hasattr(bn.bottleneck_type, "value")
                    else str(bn.bottleneck_type),
                    "severity": bn.severity.value
                    if hasattr(bn.severity, "value")
                    else str(bn.severity),
                }
                for bn in bottleneck_result.bottlenecks
            ]

        return signals

    # ─── Phase 3: Validate + Assemble ───────────────────────────

    def _phase3_validate_and_assemble(
        self,
        classification: ClassificationResult,
        input_data: DependencyAnalysisInput,
        pipeline_latency_ms: float,
        correlation_id: str,
        graph_build_time_ms: float = 0.0,
        trace_analysis_time_ms: float = 0.0,
        impact_calculation_time_ms: float = 0.0,
        classification_time_ms: float = 0.0,
    ) -> DependencyAgentOutput:
        """Phase 3: Validate output and assemble final result.

        Args:
            classification: Result from Phase 2.
            input_data: Original input.
            pipeline_latency_ms: Total pipeline time.
            correlation_id: Request correlation ID.
            graph_build_time_ms: Graph build time.
            trace_analysis_time_ms: Trace analysis time.
            impact_calculation_time_ms: Impact calculation time.
            classification_time_ms: Classification time.

        Returns:
            DependencyAgentOutput with validation attached.
        """
        # ── assemble output ─────────────────────────────────────
        output = self._synthesizer.assemble_output(
            classification=classification,
            input_data=input_data,
            pipeline_latency_ms=pipeline_latency_ms,
            correlation_id=correlation_id,
            graph_build_time_ms=graph_build_time_ms,
            trace_analysis_time_ms=trace_analysis_time_ms,
            impact_calculation_time_ms=impact_calculation_time_ms,
            classification_time_ms=classification_time_ms,
        )

        # ── validate ────────────────────────────────────────────
        validation: Optional[ValidationResult] = None
        validation_ms = 0.0
        if self._config.features.enable_validation:
            v_start = time.perf_counter()
            with self._telemetry.measure(
                "validation", correlation_id
            ):
                validation = self._validator.validate(
                    output=output,
                    input_data=input_data,
                    correlation_id=correlation_id,
                )
            validation_ms = (
                (time.perf_counter() - v_start) * 1000
            )

            if not validation.validation_passed:
                self._telemetry.record_validation_failure(
                    correlation_id
                )
                logger.warning(
                    f"Validation failed — "
                    f"{len(validation.errors)} errors, "
                    f"{len(validation.warnings)} warnings",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "validation",
                    },
                )

            # Rebuild output with validation + final latency
            final_latency = pipeline_latency_ms + validation_ms
            output = self._synthesizer.assemble_output(
                classification=classification,
                input_data=input_data,
                pipeline_latency_ms=final_latency,
                correlation_id=correlation_id,
                graph_build_time_ms=graph_build_time_ms,
                trace_analysis_time_ms=trace_analysis_time_ms,
                impact_calculation_time_ms=impact_calculation_time_ms,
                classification_time_ms=classification_time_ms,
                validation_time_ms=validation_ms,
            )

            # Attach validation
            output = DependencyAgentOutput(
                **{
                    **output.model_dump(),
                    "validation": validation,
                }
            )

        return output
