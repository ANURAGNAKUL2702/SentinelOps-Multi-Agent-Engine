"""
File: agent.py
Purpose: Hybrid orchestrator — 3-phase pipeline (extract → classify → validate).
Dependencies: All log_agent sub-modules.
Performance: <2s end-to-end, <100ms deterministic path.

Coordinates the full log analysis pipeline:
  Phase 1: Deterministic signal extraction + anomaly detection
  Phase 2: Classification (LLM with fallback to rule engine)
  Phase 3: Validation + output assembly

Graceful degradation: if LLM fails, falls back to deterministic rules.
Every call is traced via correlation ID.  All telemetry is collected.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from agents.log_agent.config import LogAgentConfig
from agents.log_agent.core.anomaly_detector import AnomalyDetector
from agents.log_agent.core.signal_extractor import SignalExtractor
from agents.log_agent.fallback import RuleEngine
from agents.log_agent.llm.classifier import (
    CircuitState,
    LLMClassifier,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
)
from agents.log_agent.llm.synthesizer import Synthesizer
from agents.log_agent.schema import (
    AnomalyDetectionResult,
    ClassificationResult,
    LogAgentOutput,
    LogAnalysisInput,
    SignalExtractionResult,
    ValidationResult,
)
from agents.log_agent.telemetry import TelemetryCollector, get_logger
from agents.log_agent.validator import OutputValidator

logger = get_logger("log_agent.agent")


class LogAgent:
    """Hybrid log analysis agent — 80% deterministic, 20% LLM.

    Pipeline::

        Input → [Phase 1: Extract] → [Phase 2: Classify] → [Phase 3: Validate] → Output

    Phase 1 (deterministic, <100ms):
        - Signal extraction (error_pct, growth_rate, trends, keywords)
        - Anomaly detection (z-score 3-sigma)

    Phase 2 (hybrid):
        - LLM classification (if enabled + circuit closed)
        - Fallback to deterministic rule engine on failure

    Phase 3 (deterministic, <5ms):
        - 23-check validation
        - Output assembly per synthesizer.txt schema

    Args:
        config: Agent configuration (thresholds, feature flags, LLM config).
        llm_provider: LLM provider adapter (defaults to MockLLMProvider).

    Example::

        agent = LogAgent(LogAgentConfig())
        input_data = LogAnalysisInput(
            error_summary={"payment-service": 340},
            total_error_logs=340,
            error_trends={"payment-service": [0, 0, 5, 30, 340]},
            keyword_matches={"payment-service": ["database timeout"]},
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
        )
        output = agent.analyze(input_data)
        print(output.model_dump_json(indent=2))
    """

    def __init__(
        self,
        config: Optional[LogAgentConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self._config = config or LogAgentConfig()
        self._telemetry = TelemetryCollector()

        # ── layer instances ─────────────────────────────────────
        self._extractor = SignalExtractor(self._config)
        self._anomaly_detector = AnomalyDetector(self._config)
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
            f"LogAgent initialized — "
            f"llm={'enabled' if self._config.features.use_llm else 'disabled'}, "
            f"fallback={'enabled' if self._config.features.fallback_to_rules else 'disabled'}, "
            f"anomaly={'enabled' if self._config.features.enable_anomaly_detection else 'disabled'}, "
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
            from agents.log_agent.llm.groq_provider import GroqProvider
            return GroqProvider(
                api_key=self._config.llm.api_key or None,
                model=self._config.llm.model,
            )

        logger.info(f"Using MockLLMProvider (provider='{provider_name}')")
        return MockLLMProvider()

    # ─── public API ─────────────────────────────────────────────

    def analyze(
        self,
        input_data: LogAnalysisInput,
        correlation_id: str = "",
    ) -> LogAgentOutput:
        """Run the full hybrid analysis pipeline.

        This is the single entry point for the log agent.
        Idempotent: same input → same output (when LLM is disabled
        or cache is warm).

        Args:
            input_data: Validated log analysis input.
            correlation_id: Optional correlation ID (auto-generated if empty).

        Returns:
            LogAgentOutput with all fields populated.

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
                    "total_errors": input_data.total_error_logs,
                    "services": list(input_data.error_summary.keys()),
                },
            },
        )

        try:
            # ── Phase 1: Deterministic extraction ───────────────
            signals, anomalies = self._phase1_extract(input_data, cid)

            # ── Phase 2: Classification ─────────────────────────
            classification = self._phase2_classify(
                signals, input_data, cid
            )

            # ── Phase 3: Validate + assemble ────────────────────
            pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
            output = self._phase3_validate_and_assemble(
                classification=classification,
                signals=signals,
                input_data=input_data,
                pipeline_latency_ms=pipeline_elapsed,
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
                        "suspicious_count": len(output.suspicious_services),
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
            "signal_extractor": "healthy",
            "anomaly_detector": "healthy",
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
            "agent": "log_agent",
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
        input_data: LogAnalysisInput,
        correlation_id: str,
    ) -> tuple[SignalExtractionResult, Optional[AnomalyDetectionResult]]:
        """Phase 1: Extract signals and detect anomalies.

        Fully deterministic.  Same input → same output.
        Target: <100ms for 1000 services.

        Args:
            input_data: Validated input.
            correlation_id: Request correlation ID.

        Returns:
            Tuple of (signals, anomalies).
        """
        # ── signal extraction ───────────────────────────────────
        with self._telemetry.measure("extraction", correlation_id):
            signals = self._extractor.extract(input_data)

        # ── anomaly detection ───────────────────────────────────
        anomalies: Optional[AnomalyDetectionResult] = None
        if self._config.features.enable_anomaly_detection:
            with self._telemetry.measure("anomaly_detection", correlation_id):
                anomalies = self._anomaly_detector.detect(signals)

        logger.debug(
            f"Phase 1 complete — {len(signals.service_signals)} services, "
            f"{signals.extraction_latency_ms:.2f}ms extraction",
            extra={
                "correlation_id": correlation_id,
                "layer": "extraction",
                "context": {
                    "service_count": len(signals.service_signals),
                    "extraction_ms": signals.extraction_latency_ms,
                    "anomalies_found": (
                        len(anomalies.anomalies) if anomalies else 0
                    ),
                },
            },
        )

        return signals, anomalies

    # ─── Phase 2: Classification ────────────────────────────────

    def _phase2_classify(
        self,
        signals: SignalExtractionResult,
        input_data: LogAnalysisInput,
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
            signals: Pre-calculated signals from Phase 1.
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
                        signals=signals,
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
                signals=signals,
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
        signals: SignalExtractionResult,
        input_data: LogAnalysisInput,
        pipeline_latency_ms: float,
        correlation_id: str,
    ) -> LogAgentOutput:
        """Phase 3: Validate output and assemble final result.

        Runs all 23 validation checks.  Validation failures are logged
        but do NOT block output generation (warnings are informational).

        Args:
            classification: Result from Phase 2.
            signals: Signals from Phase 1.
            input_data: Original input.
            pipeline_latency_ms: Total pipeline time so far.
            correlation_id: Request correlation ID.

        Returns:
            LogAgentOutput with validation results attached.
        """
        # ── assemble output ─────────────────────────────────────
        output = self._synthesizer.assemble_output(
            classification=classification,
            signals=signals,
            input_data=input_data,
            pipeline_latency_ms=pipeline_latency_ms,
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
            output = LogAgentOutput(
                agent=output.agent,
                analysis_timestamp=output.analysis_timestamp,
                time_window=output.time_window,
                suspicious_services=output.suspicious_services,
                system_error_summary=output.system_error_summary,
                database_related_errors_detected=output.database_related_errors_detected,
                confidence_score=output.confidence_score,
                correlation_id=output.correlation_id,
                classification_source=output.classification_source,
                pipeline_latency_ms=output.pipeline_latency_ms,
                validation=validation,
            )

        return output
