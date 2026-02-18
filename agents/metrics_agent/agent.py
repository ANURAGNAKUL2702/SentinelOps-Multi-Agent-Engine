"""
File: agent.py
Purpose: Hybrid orchestrator — 3-phase pipeline (extract → classify → validate).
Dependencies: All metrics_agent sub-modules.
Performance: <2s end-to-end, <100ms deterministic path.

Coordinates the full metrics analysis pipeline:
  Phase 1: Deterministic metric aggregation + anomaly detection + correlation
  Phase 2: Classification (LLM with fallback to rule engine)
  Phase 3: Validation + output assembly

Graceful degradation: if LLM fails, falls back to deterministic rules.
Every call is traced via correlation ID.  All telemetry is collected.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.core.anomaly_detector import AnomalyDetector
from agents.metrics_agent.core.correlation_detector import CorrelationDetector
from agents.metrics_agent.core.metric_aggregator import MetricAggregator
from agents.metrics_agent.fallback import RuleEngine
from agents.metrics_agent.llm.classifier import (
    CircuitState,
    LLMClassifier,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
)
from agents.metrics_agent.llm.synthesizer import Synthesizer
from agents.metrics_agent.schema import (
    AggregationResult,
    AnomalyDetectionResult,
    ClassificationResult,
    CorrelationDetectionResult,
    MetricsAgentOutput,
    MetricsAnalysisInput,
    ValidationResult,
)
from agents.metrics_agent.telemetry import TelemetryCollector, get_logger
from agents.metrics_agent.validator import OutputValidator

logger = get_logger("metrics_agent.agent")


class MetricsAgent:
    """Hybrid metrics analysis agent — 80% deterministic, 20% LLM.

    Pipeline::

        Input → [Phase 1: Extract] → [Phase 2: Classify] → [Phase 3: Validate] → Output

    Phase 1 (deterministic, <100ms):
        - Metric aggregation (z-score, growth rate, trend, threshold)
        - Anomaly detection (type classification, severity)
        - Correlation detection (Pearson r, relationship)

    Phase 2 (hybrid):
        - LLM classification (if enabled + circuit closed)
        - Fallback to deterministic rule engine on failure

    Phase 3 (deterministic, <5ms):
        - 23-check validation
        - Output assembly per MetricsAgentOutput schema

    Args:
        config: Agent configuration (thresholds, feature flags, LLM config).
        llm_provider: LLM provider adapter (defaults to MockLLMProvider).

    Example::

        agent = MetricsAgent(MetricsAgentConfig())
        input_data = MetricsAnalysisInput(
            service="payment-api",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={"cpu_percent": [45.2, 48.1, 52.3, 87.6, 95.2]},
            baseline={"cpu_percent": BaselineStats(mean=50.0, stddev=10.0)},
        )
        output = agent.analyze(input_data)
        print(output.model_dump_json(indent=2))
    """

    def __init__(
        self,
        config: Optional[MetricsAgentConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self._config = config or MetricsAgentConfig()
        self._telemetry = TelemetryCollector()

        # ── layer instances ─────────────────────────────────────
        self._aggregator = MetricAggregator(self._config)
        self._anomaly_detector = AnomalyDetector(self._config)
        self._correlation_detector = CorrelationDetector(self._config)
        self._rule_engine = RuleEngine(self._config)
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
            f"MetricsAgent initialized — "
            f"llm={'enabled' if self._config.features.use_llm else 'disabled'}, "
            f"fallback={'enabled' if self._config.features.fallback_to_rules else 'disabled'}, "
            f"anomaly={'enabled' if self._config.features.enable_anomaly_detection else 'disabled'}, "
            f"correlation={'enabled' if self._config.features.enable_correlation else 'disabled'}, "
            f"validation={'enabled' if self._config.features.enable_validation else 'disabled'}",
        )

    # ─── provider resolution ────────────────────────────────────

    def _resolve_provider(self) -> LLMProvider:
        """Auto-detect LLM provider from config.

        Provider selection:
            'groq'  → GroqProvider (real API calls)
            'mock'  → MockLLMProvider (testing)
            other   → MockLLMProvider (safe default)

        Returns:
            Configured LLMProvider instance.
        """
        provider_name = self._config.llm.provider.lower()

        if provider_name == "groq":
            from agents.metrics_agent.llm.groq_provider import GroqProvider
            return GroqProvider(
                api_key=self._config.llm.api_key or None,
                model=self._config.llm.model,
            )

        logger.info(f"Using MockLLMProvider (provider='{provider_name}')")
        return MockLLMProvider()

    # ─── public API ─────────────────────────────────────────────

    def analyze(
        self,
        input_data: MetricsAnalysisInput,
        correlation_id: str = "",
    ) -> MetricsAgentOutput:
        """Run the full hybrid analysis pipeline.

        This is the single entry point for the metrics agent.
        Idempotent: same input → same output (when LLM is disabled
        or cache is warm).

        Args:
            input_data: Validated metrics analysis input.
            correlation_id: Optional correlation ID (auto-generated if empty).

        Returns:
            MetricsAgentOutput with all fields populated.

        Raises:
            ValueError: If input_data fails validation.
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
                    "service": input_data.service,
                    "metric_count": len(input_data.metrics),
                },
            },
        )

        try:
            # ── Phase 1: Deterministic extraction ───────────────
            aggregation, anomalies, correlations, extraction_ms = (
                self._phase1_extract(input_data, cid)
            )

            # ── Phase 2: Classification ─────────────────────────
            classification = self._phase2_classify(
                aggregation, anomalies, correlations, input_data, cid
            )

            # ── Phase 3: Validate + assemble ────────────────────
            pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
            output = self._phase3_validate_and_assemble(
                classification=classification,
                aggregation=aggregation,
                input_data=input_data,
                pipeline_latency_ms=pipeline_elapsed,
                extraction_time_ms=extraction_ms,
                correlation_id=cid,
            )

            self._telemetry.analyses_succeeded.inc()
            self._telemetry.measure_value("pipeline_total", pipeline_elapsed)

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
                        "classification_source": output.classification_source,
                        "anomaly_count": len(output.anomalous_metrics),
                        "correlation_count": len(output.correlations),
                        "pipeline_latency_ms": round(pipeline_elapsed, 2),
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
        """Return agent health status for monitoring endpoints.

        Returns:
            Dict with component health, circuit state, and metrics.

        Example::

            health = agent.health_check()
            # {
            #   "status": "healthy",
            #   "components": {...},
            #   "metrics": {...},
            # }
        """
        circuit_state = "n/a"
        if self._llm_classifier:
            circuit_state = self._llm_classifier.circuit_state.value

        components = {
            "metric_aggregator": "healthy",
            "anomaly_detector": "healthy",
            "correlation_detector": "healthy",
            "rule_engine": "healthy",
            "validator": "healthy",
            "llm_classifier": (
                "healthy" if circuit_state in ("closed", "half_open", "n/a")
                else "degraded"
            ),
        }

        overall = "healthy"
        if circuit_state == "open":
            overall = "degraded"

        return {
            "status": overall,
            "agent": "metrics_agent",
            "config": {
                "llm_enabled": self._config.features.use_llm,
                "fallback_enabled": self._config.features.fallback_to_rules,
                "circuit_state": circuit_state,
            },
            "components": components,
            "metrics": self._telemetry.snapshot(),
        }

    @property
    def telemetry(self) -> TelemetryCollector:
        """Access the telemetry collector for metrics export."""
        return self._telemetry

    # ─── Phase 1: Deterministic Extraction ──────────────────────

    def _phase1_extract(
        self,
        input_data: MetricsAnalysisInput,
        correlation_id: str,
    ) -> tuple[
        AggregationResult,
        Optional[AnomalyDetectionResult],
        Optional[CorrelationDetectionResult],
        float,
    ]:
        """Phase 1: Aggregate metrics, detect anomalies, find correlations.

        Fully deterministic.  Same input → same output.
        Target: <100ms for 1000 metrics.

        Args:
            input_data: Validated input.
            correlation_id: Request correlation ID.

        Returns:
            Tuple of (aggregation, anomalies, correlations, extraction_ms).
        """
        phase_start = time.perf_counter()

        # ── metric aggregation ──────────────────────────────────
        with self._telemetry.measure("extraction", correlation_id):
            aggregation = self._aggregator.aggregate(input_data)

        # ── anomaly detection ───────────────────────────────────
        anomalies: Optional[AnomalyDetectionResult] = None
        if self._config.features.enable_anomaly_detection:
            with self._telemetry.measure("anomaly_detection", correlation_id):
                anomalies = self._anomaly_detector.detect(
                    aggregation, input_data
                )

        # ── correlation detection ───────────────────────────────
        correlations: Optional[CorrelationDetectionResult] = None
        if self._config.features.enable_correlation:
            with self._telemetry.measure("correlation", correlation_id):
                correlations = self._correlation_detector.detect(
                    aggregation, input_data
                )

        extraction_ms = (time.perf_counter() - phase_start) * 1000

        logger.debug(
            f"Phase 1 complete — {aggregation.total_metrics_analyzed} metrics, "
            f"{extraction_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "extraction",
                "context": {
                    "metrics_analyzed": aggregation.total_metrics_analyzed,
                    "extraction_ms": round(extraction_ms, 2),
                    "anomalies_found": (
                        len(anomalies.anomalies) if anomalies else 0
                    ),
                    "correlations_found": (
                        len(correlations.correlations)
                        if correlations else 0
                    ),
                },
            },
        )

        return aggregation, anomalies, correlations, extraction_ms

    # ─── Phase 2: Classification ────────────────────────────────

    def _phase2_classify(
        self,
        aggregation: AggregationResult,
        anomalies: Optional[AnomalyDetectionResult],
        correlations: Optional[CorrelationDetectionResult],
        input_data: MetricsAnalysisInput,
        correlation_id: str,
    ) -> ClassificationResult:
        """Phase 2: Classify using LLM or fallback rule engine.

        Decision tree::

            use_llm=True AND circuit!=OPEN
              → try LLM
                → success → return LLM result
                → failure AND fallback=True → return fallback result
                → failure AND fallback=False → raise
            use_llm=False OR circuit=OPEN
              → return fallback result

        Args:
            aggregation: Pre-calculated metric signals.
            anomalies: Anomaly detection results.
            correlations: Correlation detection results.
            input_data: Original input for cross-reference.
            correlation_id: Request correlation ID.

        Returns:
            ClassificationResult from LLM or fallback.
        """
        with self._telemetry.measure("classification", correlation_id):
            # ── try LLM path ────────────────────────────────────
            if (
                self._config.features.use_llm
                and self._llm_classifier is not None
            ):
                try:
                    result = self._llm_classifier.classify(
                        aggregation=aggregation,
                        input_data=input_data,
                        correlation_id=correlation_id,
                    )
                    logger.info(
                        f"LLM classification succeeded — "
                        f"source={result.classification_source}",
                        extra={
                            "correlation_id": correlation_id,
                            "layer": "classification",
                        },
                    )
                    return result

                except LLMProviderError as e:
                    logger.warning(
                        f"LLM classification failed: {e}",
                        extra={
                            "correlation_id": correlation_id,
                            "layer": "classification",
                            "context": {"error": str(e)},
                        },
                    )
                    if not self._config.features.fallback_to_rules:
                        raise

                    # Fall through to rule engine
                    self._telemetry.record_fallback(correlation_id)

            # ── fallback / default: rule engine ─────────────────
            result = self._rule_engine.classify(
                aggregation=aggregation,
                anomaly_result=anomalies,
                correlation_result=correlations,
                input_data=input_data,
                correlation_id=correlation_id,
            )

            if not self._config.features.use_llm:
                logger.debug(
                    "Rule engine classification (LLM disabled)",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "classification",
                    },
                )
            else:
                logger.info(
                    "Fallback rule engine classification",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "classification",
                    },
                )

            return result

    # ─── Phase 3: Validate + Assemble ───────────────────────────

    def _phase3_validate_and_assemble(
        self,
        classification: ClassificationResult,
        aggregation: AggregationResult,
        input_data: MetricsAnalysisInput,
        pipeline_latency_ms: float,
        extraction_time_ms: float,
        correlation_id: str,
    ) -> MetricsAgentOutput:
        """Phase 3: Validate output and assemble final result.

        Runs all 23 validation checks.  Validation failures are logged
        but do NOT block output generation (warnings are informational).

        Args:
            classification: Result from Phase 2.
            aggregation: Aggregation from Phase 1.
            input_data: Original input.
            pipeline_latency_ms: Total pipeline time so far.
            extraction_time_ms: Extraction phase time.
            correlation_id: Request correlation ID.

        Returns:
            MetricsAgentOutput with validation results attached.
        """
        # ── assemble output ─────────────────────────────────────
        output = self._synthesizer.assemble_output(
            classification=classification,
            aggregation=aggregation,
            input_data=input_data,
            pipeline_latency_ms=pipeline_latency_ms,
            extraction_time_ms=extraction_time_ms,
            correlation_id=correlation_id,
        )

        # ── validate ────────────────────────────────────────────
        validation: Optional[ValidationResult] = None
        if self._config.features.enable_validation:
            with self._telemetry.measure("validation", correlation_id):
                validation = self._validator.validate(
                    output=output,
                    input_data=input_data,
                    correlation_id=correlation_id,
                )

            if not validation.validation_passed:
                self._telemetry.record_validation_failure(correlation_id)
                logger.warning(
                    f"Validation failed — {len(validation.errors)} errors, "
                    f"{len(validation.warnings)} warnings",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "validation",
                        "context": {
                            "errors": [
                                e.model_dump() for e in validation.errors
                            ],
                            "warnings": [
                                w.model_dump() for w in validation.warnings
                            ],
                        },
                    },
                )

            # Attach validation result to output
            output = MetricsAgentOutput(
                agent=output.agent,
                analysis_timestamp=output.analysis_timestamp,
                time_window=output.time_window,
                service=output.service,
                anomalous_metrics=output.anomalous_metrics,
                correlations=output.correlations,
                system_summary=output.system_summary,
                confidence_score=output.confidence_score,
                confidence_reasoning=output.confidence_reasoning,
                correlation_id=output.correlation_id,
                classification_source=output.classification_source,
                pipeline_latency_ms=output.pipeline_latency_ms,
                metadata=output.metadata,
                validation=validation,
            )

        return output
