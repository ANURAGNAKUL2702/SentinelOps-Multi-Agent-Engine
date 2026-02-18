"""
File: fallback.py
Purpose: Rule-based classification engine — deterministic fallback when LLM is unavailable.
Dependencies: Standard library only
Performance: <5ms per classification

Implements the exact same classification rules from synthesizer.txt as
pure Python.  Produces the same output schema as the LLM classifier.
Every decision is explainable via the `reasoning` trace.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from agents.log_agent.config import LogAgentConfig, ThresholdConfig
from agents.log_agent.schema import (
    ClassificationResult,
    LogAnalysisInput,
    ServiceSignal,
    SeverityHint,
    SignalExtractionResult,
    SuspiciousService,
    SystemErrorSummary,
    TrendType,
)
from agents.log_agent.telemetry import get_logger

logger = get_logger("log_agent.fallback")


class RuleEngine:
    """Pure-deterministic rule-based classification engine.

    Implements the synthesizer.txt specification as Python code.
    Produces ``ClassificationResult`` with the same schema as the
    LLM classifier.  Every decision includes reasoning for debugging.

    Args:
        config: Agent configuration.

    Example::

        engine = RuleEngine(LogAgentConfig())
        result = engine.classify(signals, input_data)
        print(result.confidence_score)       # 0.9
        print(result.classification_source)  # "deterministic"
    """

    def __init__(self, config: Optional[LogAgentConfig] = None) -> None:
        self._config = config or LogAgentConfig()
        self._t = self._config.thresholds
        self._kw = self._config.keywords

    def classify(
        self,
        signals: SignalExtractionResult,
        input_data: LogAnalysisInput,
        correlation_id: str = "",
    ) -> ClassificationResult:
        """Classify signals using deterministic rules.

        Args:
            signals: Output of SignalExtractor.extract().
            input_data: Original input (for keyword cross-reference).
            correlation_id: Request correlation ID.

        Returns:
            ClassificationResult matching the LLM classifier's schema.
        """
        start = time.perf_counter()
        reasoning: List[str] = []

        # ── 1. identify suspicious services ─────────────────────
        suspicious: List[SuspiciousService] = []
        for sig in signals.service_signals:
            is_suspicious, reasons = self._is_suspicious(sig)
            if is_suspicious:
                severity = self._classify_severity(sig)
                keywords = input_data.keyword_matches.get(sig.service, [])

                suspicious.append(SuspiciousService(
                    service=sig.service,
                    error_count=sig.error_count,
                    error_percentage=sig.error_percentage,
                    error_keywords_detected=keywords,
                    error_trend=sig.trend_type,
                    severity_hint=severity,
                    log_flooding=sig.log_flooding_signal,
                ))
                reasoning.extend(
                    f"  [{sig.service}] {r}" for r in reasons
                )

        # ── 2. build system error summary ───────────────────────
        dominant_svc = self._find_dominant_service(signals.service_signals)
        system_wide_spike = self._detect_system_wide_spike(
            signals.service_signals, signals.system_signals.affected_service_count
        )

        summary = SystemErrorSummary(
            total_error_logs=signals.system_signals.total_error_logs,
            dominant_service=dominant_svc,
            system_wide_spike=system_wide_spike,
            potential_upstream_failure=signals.system_signals.cascading_candidate,
        )

        # ── 3. database-related detection ───────────────────────
        db_related = self._detect_database_errors(input_data.keyword_matches)

        # ── 4. confidence scoring ──────────────────────────────
        confidence, conf_reasons = self._compute_confidence(
            signals, suspicious
        )
        reasoning.extend(conf_reasons)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Rule engine classified {len(suspicious)} suspicious services, "
            f"confidence={confidence:.2f}, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "classification",
                "context": {
                    "suspicious_count": len(suspicious),
                    "confidence": confidence,
                    "reasoning_steps": len(reasoning),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return ClassificationResult(
            suspicious_services=suspicious,
            system_error_summary=summary,
            database_related_errors_detected=db_related,
            confidence_score=confidence,
            classification_source="deterministic",
            classification_latency_ms=round(elapsed_ms, 2),
        )

    # ── suspicious service identification ───────────────────────

    def _is_suspicious(
        self, sig: ServiceSignal
    ) -> Tuple[bool, List[str]]:
        """Apply synthesizer.txt suspicious-service rules.

        A service is suspicious if error_count > 0 AND any of:
          - error_percentage > 10
          - growth_rate > 100
          - critical_keyword = true
          - log_flooding_signal = true

        Returns:
            (is_suspicious, list_of_reasons)
        """
        if sig.error_count <= 0:
            return False, []

        reasons: List[str] = []
        t = self._t

        if sig.error_percentage > t.suspicious_error_pct:
            reasons.append(
                f"error_percentage={sig.error_percentage:.1f}% "
                f"> threshold {t.suspicious_error_pct}%"
            )
        if sig.growth_rate_last_period > t.suspicious_growth_rate:
            reasons.append(
                f"growth_rate={sig.growth_rate_last_period:.1f}% "
                f"> threshold {t.suspicious_growth_rate}%"
            )
        if sig.critical_keyword:
            reasons.append("critical keyword detected")
        if sig.log_flooding_signal:
            reasons.append("log flooding detected")

        return len(reasons) > 0, reasons

    # ── severity classification ─────────────────────────────────

    def _classify_severity(self, sig: ServiceSignal) -> SeverityHint:
        """Apply synthesizer.txt severity rules.

        IF error_percentage > 40 OR critical_keyword → high
        ELIF 10 ≤ error_percentage ≤ 40 OR moderate_keyword → medium
        ELSE → low
        """
        t = self._t

        if sig.error_percentage > t.high_error_pct or sig.critical_keyword:
            return SeverityHint.HIGH
        elif (
            t.medium_error_pct_low <= sig.error_percentage <= t.medium_error_pct_high
            or sig.moderate_keyword
        ):
            return SeverityHint.MEDIUM
        else:
            return SeverityHint.LOW

    # ── confidence scoring ──────────────────────────────────────

    def _compute_confidence(
        self,
        signals: SignalExtractionResult,
        suspicious: List[SuspiciousService],
    ) -> Tuple[float, List[str]]:
        """Compute confidence score using the algorithm from synthesizer.txt.

        base = 0.2
        +0.3 if any dominant_service_signal
        +0.2 if any trend = increasing OR sudden_spike
        +0.2 if any critical_keyword
        +0.1 if cascading_candidate
        -0.2 if affected_service_count > 5
        -0.3 if all growth_rate < 20
        Clamp [0.0, 1.0]
        If no suspicious → confidence < 0.3

        Returns:
            (confidence_score, reasoning_steps)
        """
        t = self._t
        reasons: List[str] = []
        confidence = t.confidence_base
        reasons.append(f"base confidence = {confidence}")

        # +0.3 dominant
        if any(s.dominant_service_signal for s in signals.service_signals):
            confidence += t.confidence_dominant_bonus
            reasons.append(f"+{t.confidence_dominant_bonus} dominant service found")

        # +0.2 trend
        if any(
            s.trend_type in (TrendType.INCREASING, TrendType.SUDDEN_SPIKE)
            for s in signals.service_signals
        ):
            confidence += t.confidence_trend_bonus
            reasons.append(f"+{t.confidence_trend_bonus} increasing/spike trend")

        # +0.2 critical keyword
        if any(s.critical_keyword for s in signals.service_signals):
            confidence += t.confidence_keyword_bonus
            reasons.append(f"+{t.confidence_keyword_bonus} critical keyword")

        # +0.1 cascading
        if signals.system_signals.cascading_candidate:
            confidence += t.confidence_cascading_bonus
            reasons.append(f"+{t.confidence_cascading_bonus} cascading candidate")

        # -0.2 distributed errors
        if (
            signals.system_signals.affected_service_count
            > t.confidence_distributed_threshold
        ):
            confidence -= t.confidence_distributed_penalty
            reasons.append(
                f"-{t.confidence_distributed_penalty} distributed errors "
                f"({signals.system_signals.affected_service_count} services)"
            )

        # -0.3 all stable
        if all(
            s.growth_rate_last_period < t.confidence_stable_growth_max
            for s in signals.service_signals
        ):
            confidence -= t.confidence_stable_penalty
            reasons.append(
                f"-{t.confidence_stable_penalty} all growth rates < "
                f"{t.confidence_stable_growth_max}%"
            )

        # clamp
        confidence = max(0.0, min(1.0, confidence))
        reasons.append(f"clamped confidence = {confidence:.2f}")

        # enforce: no suspicious → confidence < 0.3
        if not suspicious and confidence >= 0.3:
            confidence = 0.0
            reasons.append("forced to 0.0 — no suspicious services")

        return round(confidence, 2), reasons

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _find_dominant_service(
        service_signals: List[ServiceSignal],
    ) -> Optional[str]:
        """Find the dominant service (>60% errors), or None."""
        for sig in service_signals:
            if sig.dominant_service_signal:
                return sig.service
        return None

    @staticmethod
    def _detect_system_wide_spike(
        service_signals: List[ServiceSignal],
        affected_count: int,
    ) -> bool:
        """system_wide_spike = >50% services affected AND any sudden_spike."""
        total_services = len(service_signals) if service_signals else 1
        pct_affected = affected_count / total_services
        has_spike = any(
            s.trend_type == TrendType.SUDDEN_SPIKE for s in service_signals
        )
        return pct_affected > 0.5 and has_spike

    def _detect_database_errors(
        self, keyword_matches: Dict[str, List[str]]
    ) -> bool:
        """Check if any keywords indicate database-related errors."""
        db_kw = self._kw.database_keywords
        for keywords in keyword_matches.values():
            for kw in keywords:
                kw_lower = kw.lower()
                for db in db_kw:
                    if db.lower() in kw_lower or kw_lower in db.lower():
                        return True
        return False
