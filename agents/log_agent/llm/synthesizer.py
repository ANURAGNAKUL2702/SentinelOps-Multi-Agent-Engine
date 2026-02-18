"""
File: llm/synthesizer.py
Purpose: Root cause synthesis combining log/metric/dependency analysis.
Dependencies: Standard library only.
Performance: <50ms deterministic, <10s with LLM call.

Generates causal chains, remediation suggestions, and combined
confidence scores from multi-agent analysis results.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.log_agent.config import LogAgentConfig
from agents.log_agent.schema import (
    ClassificationResult,
    LogAgentOutput,
    LogAnalysisInput,
    SignalExtractionResult,
    SuspiciousService,
    ValidationResult,
)
from agents.log_agent.telemetry import get_logger

logger = get_logger("log_agent.llm.synthesizer")


# ═══════════════════════════════════════════════════════════════
#  CAUSAL CHAIN
# ═══════════════════════════════════════════════════════════════


class CausalStep:
    """A single step in the root cause causal chain.

    Example::

        step = CausalStep(
            order=1,
            service="database",
            event="Connection pool exhausted",
            evidence="error_percentage=96.6%, keyword='database timeout'",
        )
    """

    __slots__ = ("order", "service", "event", "evidence")

    def __init__(
        self,
        order: int,
        service: str,
        event: str,
        evidence: str,
    ) -> None:
        self.order = order
        self.service = service
        self.event = event
        self.evidence = evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order": self.order,
            "service": self.service,
            "event": self.event,
            "evidence": self.evidence,
        }


class Remediation:
    """A suggested remediation action.

    Example::

        fix = Remediation(
            priority=1,
            action="Restart database connection pool",
            service="payment-service",
            category="immediate",
        )
    """

    __slots__ = ("priority", "action", "service", "category")

    def __init__(
        self,
        priority: int,
        action: str,
        service: str,
        category: str = "immediate",
    ) -> None:
        self.priority = priority
        self.action = action
        self.service = service
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "action": self.action,
            "service": self.service,
            "category": self.category,
        }


class SynthesisResult:
    """Complete synthesis output with causal chain and remediations.

    Example::

        synthesis = SynthesisResult(
            root_cause_service="database",
            root_cause_summary="Database timeout cascade",
            causal_chain=[CausalStep(...)],
            remediations=[Remediation(...)],
            combined_confidence=0.9,
        )
    """

    def __init__(
        self,
        root_cause_service: Optional[str],
        root_cause_summary: str,
        causal_chain: List[CausalStep],
        remediations: List[Remediation],
        combined_confidence: float,
        synthesis_latency_ms: float = 0.0,
    ) -> None:
        self.root_cause_service = root_cause_service
        self.root_cause_summary = root_cause_summary
        self.causal_chain = causal_chain
        self.remediations = remediations
        self.combined_confidence = max(0.0, min(1.0, combined_confidence))
        self.synthesis_latency_ms = synthesis_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause_service": self.root_cause_service,
            "root_cause_summary": self.root_cause_summary,
            "causal_chain": [s.to_dict() for s in self.causal_chain],
            "remediations": [r.to_dict() for r in self.remediations],
            "combined_confidence": round(self.combined_confidence, 2),
            "synthesis_latency_ms": round(self.synthesis_latency_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════
#  SYNTHESIZER
# ═══════════════════════════════════════════════════════════════


class Synthesizer:
    """Root cause synthesizer — combines signals into actionable output.

    Generates causal chains, remediation suggestions, and assembles
    the final LogAgentOutput from classification results.

    Args:
        config: Agent configuration.

    Example::

        synth = Synthesizer(LogAgentConfig())
        output = synth.assemble_output(
            classification=classification_result,
            signals=extraction_result,
            input_data=input_data,
        )
        print(output.model_dump_json(indent=2))
    """

    def __init__(self, config: Optional[LogAgentConfig] = None) -> None:
        self._config = config or LogAgentConfig()

    def synthesize(
        self,
        signals: SignalExtractionResult,
        classification: ClassificationResult,
        input_data: LogAnalysisInput,
        correlation_id: str = "",
    ) -> SynthesisResult:
        """Generate root cause synthesis from signals and classification.

        Builds a causal chain by analysing temporal ordering of errors,
        identifies the root cause service, and generates remediation
        suggestions.

        Args:
            signals: Pre-calculated signals from SignalExtractor.
            classification: Classification result (LLM or fallback).
            input_data: Original analysis input.
            correlation_id: Request correlation ID.

        Returns:
            SynthesisResult with causal chain and remediations.
        """
        start = time.perf_counter()

        # ── identify root cause service ─────────────────────────
        root_service = self._identify_root_cause(
            signals, classification
        )

        # ── build causal chain ──────────────────────────────────
        causal_chain = self._build_causal_chain(
            signals, classification, root_service
        )

        # ── generate remediations ───────────────────────────────
        remediations = self._generate_remediations(
            classification, input_data, root_service
        )

        # ── build summary ───────────────────────────────────────
        summary = self._build_summary(
            root_service, classification, input_data
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Synthesis completed — root_cause={root_service}, "
            f"{len(causal_chain)} causal steps, "
            f"{len(remediations)} remediations, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "synthesis",
                "context": {
                    "root_cause": root_service,
                    "causal_steps": len(causal_chain),
                    "remediations": len(remediations),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return SynthesisResult(
            root_cause_service=root_service,
            root_cause_summary=summary,
            causal_chain=causal_chain,
            remediations=remediations,
            combined_confidence=classification.confidence_score,
            synthesis_latency_ms=round(elapsed_ms, 2),
        )

    def assemble_output(
        self,
        classification: ClassificationResult,
        signals: SignalExtractionResult,
        input_data: LogAnalysisInput,
        validation: Optional[ValidationResult] = None,
        pipeline_latency_ms: float = 0.0,
        correlation_id: str = "",
    ) -> LogAgentOutput:
        """Assemble the final LogAgentOutput per synthesizer.txt schema.

        This is the last step — takes the classification result and
        wraps it into the output schema that downstream consumers expect.

        Args:
            classification: Classification result (LLM or fallback).
            signals: Pre-calculated signals.
            input_data: Original input.
            validation: Optional validation result.
            pipeline_latency_ms: Total pipeline latency.
            correlation_id: Request correlation ID.

        Returns:
            LogAgentOutput ready for downstream consumers.
        """
        return LogAgentOutput(
            agent="log_agent",
            analysis_timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            time_window=input_data.time_window,
            suspicious_services=classification.suspicious_services,
            system_error_summary=classification.system_error_summary,
            database_related_errors_detected=classification.database_related_errors_detected,
            confidence_score=classification.confidence_score,
            correlation_id=correlation_id or input_data.correlation_id,
            classification_source=classification.classification_source,
            pipeline_latency_ms=round(pipeline_latency_ms, 2),
            validation=validation,
        )

    # ── private helpers ─────────────────────────────────────────

    def _identify_root_cause(
        self,
        signals: SignalExtractionResult,
        classification: ClassificationResult,
    ) -> Optional[str]:
        """Identify the most likely root cause service.

        Priority:
            1. Dominant service (>60% of errors)
            2. Earliest error service (temporal analysis)
            3. Highest severity service from classification
        """
        # 1. dominant service
        dominant = classification.system_error_summary.dominant_service
        if dominant:
            return dominant

        # 2. earliest error service (cascading origin)
        if signals.system_signals.cascading_candidate:
            earliest = signals.system_signals.earliest_error_service
            if earliest:
                return earliest

        # 3. highest severity suspicious service
        high_severity = [
            s for s in classification.suspicious_services
            if s.severity_hint.value == "high"
        ]
        if high_severity:
            # Pick the one with highest error count
            return max(high_severity, key=lambda s: s.error_count).service

        # 4. any suspicious service
        if classification.suspicious_services:
            return max(
                classification.suspicious_services,
                key=lambda s: s.error_count,
            ).service

        return None

    def _build_causal_chain(
        self,
        signals: SignalExtractionResult,
        classification: ClassificationResult,
        root_service: Optional[str],
    ) -> List[CausalStep]:
        """Build a temporal causal chain from signals.

        Orders services by their first error occurrence and creates
        a step-by-step chain showing how the incident propagated.

        Args:
            signals: Extraction signals with temporal data.
            classification: For severity context.
            root_service: Identified root cause.

        Returns:
            Ordered list of CausalStep.
        """
        if not classification.suspicious_services:
            return []

        # Sort services by first_non_zero_trend_index
        service_order: List[tuple] = []
        for sig in signals.service_signals:
            if sig.first_non_zero_trend_index >= 0:
                service_order.append(
                    (sig.service, sig.first_non_zero_trend_index, sig)
                )

        service_order.sort(key=lambda x: x[1])

        # Build chain
        chain: List[CausalStep] = []
        suspicious_set = {
            s.service for s in classification.suspicious_services
        }

        for i, (svc, idx, sig) in enumerate(service_order):
            if svc not in suspicious_set:
                continue

            if svc == root_service:
                event = (
                    f"Initial errors detected — "
                    f"error_percentage={sig.error_percentage}%, "
                    f"trend={sig.trend_type.value}"
                )
            else:
                event = (
                    f"Cascading impact — "
                    f"errors began at period {idx}, "
                    f"growth_rate={sig.growth_rate_last_period}%"
                )

            evidence_parts = []
            if sig.critical_keyword:
                evidence_parts.append("critical_keyword=true")
            if sig.dominant_service_signal:
                evidence_parts.append("dominant_service=true")
            if sig.log_flooding_signal:
                evidence_parts.append("log_flooding=true")
            evidence_parts.append(
                f"error_count={sig.error_count}"
            )

            chain.append(CausalStep(
                order=len(chain) + 1,
                service=svc,
                event=event,
                evidence=", ".join(evidence_parts),
            ))

        return chain

    def _generate_remediations(
        self,
        classification: ClassificationResult,
        input_data: LogAnalysisInput,
        root_service: Optional[str],
    ) -> List[Remediation]:
        """Generate remediation suggestions based on classification.

        Suggestions are based on detected keywords, severity, and
        error patterns.  Priority 1 = most urgent.

        Args:
            classification: Classification result.
            input_data: Original input for keyword context.
            root_service: Identified root cause service.

        Returns:
            Prioritized list of Remediation objects.
        """
        remediations: List[Remediation] = []
        priority = 1

        for svc in classification.suspicious_services:
            target = svc.service
            keywords = input_data.keyword_matches.get(target, [])
            kw_lower = " ".join(k.lower() for k in keywords)

            # Database-related remediation
            if "database" in kw_lower or "connection" in kw_lower:
                remediations.append(Remediation(
                    priority=priority,
                    action=(
                        "Check database connection pool. "
                        "Consider increasing pool size or restarting connections."
                    ),
                    service=target,
                    category="immediate",
                ))
                priority += 1

            # OOM-related remediation
            if "oom" in kw_lower or "outofmemory" in kw_lower or "memory" in kw_lower:
                remediations.append(Remediation(
                    priority=priority,
                    action=(
                        "Investigate memory leak. "
                        "Check heap dumps and consider increasing memory limits."
                    ),
                    service=target,
                    category="immediate",
                ))
                priority += 1

            # High severity generic remediation
            if svc.severity_hint.value == "high" and not remediations:
                remediations.append(Remediation(
                    priority=priority,
                    action=(
                        f"Investigate {target} — high severity errors detected. "
                        f"Check service logs and recent deployments."
                    ),
                    service=target,
                    category="immediate",
                ))
                priority += 1

            # Log flooding remediation
            if svc.log_flooding:
                remediations.append(Remediation(
                    priority=priority,
                    action=(
                        f"Enable log rate limiting for {target} — "
                        f"log flooding detected."
                    ),
                    service=target,
                    category="preventive",
                ))
                priority += 1

        # Cascading failure remediation
        if classification.system_error_summary.potential_upstream_failure:
            remediations.append(Remediation(
                priority=priority,
                action=(
                    "Cascading failure detected. "
                    "Implement circuit breakers between dependent services."
                ),
                service=root_service or "system",
                category="preventive",
            ))

        return remediations

    @staticmethod
    def _build_summary(
        root_service: Optional[str],
        classification: ClassificationResult,
        input_data: LogAnalysisInput,
    ) -> str:
        """Build a human-readable root cause summary.

        Args:
            root_service: Identified root cause service.
            classification: Classification result.
            input_data: Original input.

        Returns:
            One-sentence summary string.
        """
        if not root_service:
            return "No clear root cause identified — insufficient signal confidence."

        svc_count = len(classification.suspicious_services)
        total_errors = classification.system_error_summary.total_error_logs
        keywords = input_data.keyword_matches.get(root_service, [])

        parts = [
            f"Root cause: {root_service} "
            f"({total_errors} total errors across {svc_count} services)"
        ]

        if keywords:
            parts.append(f"Keywords: {', '.join(keywords[:3])}")

        if classification.system_error_summary.potential_upstream_failure:
            parts.append("Pattern: cascading failure")
        elif classification.system_error_summary.system_wide_spike:
            parts.append("Pattern: system-wide spike")

        return ". ".join(parts) + "."
