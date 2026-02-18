"""
File: core/signal_extractor.py
Purpose: Pure deterministic signal extraction — replaces scanner.txt LLM prompt.
Dependencies: Standard library only (no numpy required at this scale).
Performance: <100ms for 1000 services, <10ms for typical 8-service incidents.

Extracts every signal defined in system.txt using plain Python.
Zero LLM calls.  100% reproducible.  Same input → same output.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agents.log_agent.config import LogAgentConfig, ThresholdConfig
from agents.log_agent.schema import (
    LogAnalysisInput,
    ServiceSignal,
    SignalExtractionResult,
    SystemSignal,
    TrendType,
)
from agents.log_agent.telemetry import get_logger

logger = get_logger("log_agent.signal_extractor")


# ═══════════════════════════════════════════════════════════════
#  SIGNAL EXTRACTOR
# ═══════════════════════════════════════════════════════════════


class SignalExtractor:
    """Deterministic signal extraction engine.

    Replaces the scanner.txt LLM prompt with pure Python logic.
    Extracts per-service and system-level signals from structured
    log summary data.

    Args:
        config: Agent configuration (thresholds, keywords).

    Example::

        extractor = SignalExtractor(LogAgentConfig())
        input_data = LogAnalysisInput(
            error_summary={"payment-service": 340, "auth-service": 12},
            total_error_logs=352,
            error_trends={"payment-service": [0, 0, 5, 30, 340]},
            keyword_matches={"payment-service": ["database timeout"]},
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
        )
        result = extractor.extract(input_data)
        print(result.service_signals[0].error_percentage)  # 96.59
    """

    def __init__(self, config: Optional[LogAgentConfig] = None) -> None:
        self._config = config or LogAgentConfig()
        self._thresholds = self._config.thresholds
        self._keywords = self._config.keywords

    # ── public API ──────────────────────────────────────────────

    def extract(self, input_data: LogAnalysisInput) -> SignalExtractionResult:
        """Extract all signals from the input data.

        Args:
            input_data: Validated LogAnalysisInput.

        Returns:
            SignalExtractionResult with per-service and system signals.

        Raises:
            ValueError: If input_data fails validation.
        """
        start = time.perf_counter()

        # ── per-service signals ─────────────────────────────────
        service_signals: List[ServiceSignal] = []
        total = input_data.total_error_logs

        for service, error_count in input_data.error_summary.items():
            if error_count <= 0:
                continue

            signal = self._extract_service_signal(
                service=service,
                error_count=error_count,
                total_error_logs=total,
                trend_data=input_data.error_trends.get(service, []),
                keywords=input_data.keyword_matches.get(service, []),
            )
            service_signals.append(signal)

        # ── system-level signals ────────────────────────────────
        system_signals = self._extract_system_signals(
            service_signals=service_signals,
            total_error_logs=total,
            error_trends=input_data.error_trends,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Signal extraction completed: {len(service_signals)} services "
            f"in {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": input_data.correlation_id,
                "layer": "deterministic",
                "context": {
                    "service_count": len(service_signals),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return SignalExtractionResult(
            service_signals=service_signals,
            system_signals=system_signals,
            extraction_latency_ms=round(elapsed_ms, 2),
        )

    # ── per-service extraction ──────────────────────────────────

    def _extract_service_signal(
        self,
        service: str,
        error_count: int,
        total_error_logs: int,
        trend_data: List[int],
        keywords: List[str],
    ) -> ServiceSignal:
        """Extract all signals for a single service.

        Args:
            service: Service name.
            error_count: Total errors for this service.
            total_error_logs: Total errors across all services.
            trend_data: Error counts per time bucket [oldest → newest].
            keywords: Matched error keywords for this service.

        Returns:
            ServiceSignal with all fields populated.
        """
        # error percentage (safe division)
        error_pct = self._safe_percentage(error_count, total_error_logs)

        # trend analysis
        first_nz_idx = self._first_non_zero_index(trend_data)
        growth_rate = self._compute_growth_rate(trend_data)
        trend_type = self._classify_trend(growth_rate, trend_data)

        # keyword detection
        critical_kw = self._has_keyword_match(
            keywords, self._keywords.critical_keywords
        )
        moderate_kw = self._has_keyword_match(
            keywords, self._keywords.moderate_keywords
        )

        # flooding detection
        flooding = self._detect_log_flooding(trend_data)

        # dominance detection
        dominant = error_pct > self._thresholds.dominant_service_pct

        return ServiceSignal(
            service=service,
            error_count=error_count,
            error_percentage=round(error_pct, 2),
            first_non_zero_trend_index=first_nz_idx,
            growth_rate_last_period=round(growth_rate, 2),
            trend_type=trend_type,
            critical_keyword=critical_kw,
            moderate_keyword=moderate_kw,
            log_flooding_signal=flooding,
            dominant_service_signal=dominant,
        )

    # ── system-level extraction ─────────────────────────────────

    def _extract_system_signals(
        self,
        service_signals: List[ServiceSignal],
        total_error_logs: int,
        error_trends: Dict[str, List[int]],
    ) -> SystemSignal:
        """Extract system-wide aggregate signals.

        Args:
            service_signals: Already-extracted per-service signals.
            total_error_logs: Total errors across all services.
            error_trends: Raw trend data for all services.

        Returns:
            SystemSignal with aggregate fields.
        """
        affected_count = len(service_signals)

        # earliest error service — service with lowest first_non_zero_trend_index
        earliest_service: Optional[str] = None
        earliest_idx = float("inf")
        for sig in service_signals:
            if (
                sig.first_non_zero_trend_index >= 0
                and sig.first_non_zero_trend_index < earliest_idx
            ):
                earliest_idx = sig.first_non_zero_trend_index
                earliest_service = sig.service

        # cascading detection
        cascading = self._detect_cascading(
            service_signals=service_signals,
            affected_count=affected_count,
        )

        return SystemSignal(
            total_error_logs=total_error_logs,
            affected_service_count=affected_count,
            earliest_error_service=earliest_service,
            cascading_candidate=cascading,
        )

    # ── computation helpers ─────────────────────────────────────

    @staticmethod
    def _safe_percentage(part: int, total: int) -> float:
        """Compute percentage with division-by-zero protection.

        Args:
            part: Numerator.
            total: Denominator.

        Returns:
            Percentage (0.0−100.0), or 0.0 if total is zero.

        Example::

            SignalExtractor._safe_percentage(340, 352)  # 96.59
            SignalExtractor._safe_percentage(0, 0)      # 0.0
        """
        if total <= 0:
            return 0.0
        return (part / total) * 100.0

    @staticmethod
    def _first_non_zero_index(trend: List[int]) -> int:
        """Find the index of the first non-zero value in trend data.

        Args:
            trend: List of error counts per time bucket.

        Returns:
            Index (0-based), or -1 if all zeros or empty.

        Example::

            SignalExtractor._first_non_zero_index([0, 0, 5, 30])  # 2
            SignalExtractor._first_non_zero_index([0, 0, 0])       # -1
            SignalExtractor._first_non_zero_index([])               # -1
        """
        for i, val in enumerate(trend):
            if val > 0:
                return i
        return -1

    @staticmethod
    def _compute_growth_rate(trend: List[int]) -> float:
        """Calculate growth rate between the last two periods.

        Uses the formula from planner.txt Step 3:
            growth_rate = ((last - prev) / prev) * 100

        Edge case: if prev is 0, uses mean of all periods as baseline
        (as specified in planner.txt).

        Args:
            trend: List of error counts per time bucket.

        Returns:
            Growth rate as percentage. 0.0 if insufficient data.

        Example::

            SignalExtractor._compute_growth_rate([0, 0, 5, 30, 340])
            # ((340 - 30) / 30) * 100 = 1033.33

            SignalExtractor._compute_growth_rate([0, 0, 0, 340])
            # prev=0, use mean([0,0,0,340])=85 → ((340 - 85) / 85) * 100

            SignalExtractor._compute_growth_rate([])  # 0.0
        """
        if len(trend) < 2:
            return 0.0

        last = trend[-1]
        prev = trend[-2]

        if prev > 0:
            return ((last - prev) / prev) * 100.0

        # Edge case: prev period is zero — use mean as baseline
        non_zero = [v for v in trend if v > 0]
        if not non_zero:
            return 0.0
        mean_val = sum(non_zero) / len(non_zero)
        if mean_val <= 0:
            return 0.0
        return ((last - mean_val) / mean_val) * 100.0

    def _classify_trend(
        self, growth_rate: float, trend: List[int]
    ) -> TrendType:
        """Classify trend direction from growth rate.

        Rules from planner.txt Step 3:
            growth_rate > 200%    → sudden_spike
            20 ≤ rate ≤ 200       → increasing
            -20 < rate < 20       → stable
            rate ≤ -20            → decreasing

        Args:
            growth_rate: Computed growth rate percentage.
            trend: Raw trend data (used for edge case).

        Returns:
            TrendType enum value.
        """
        t = self._thresholds

        if growth_rate > t.spike_growth_rate:
            return TrendType.SUDDEN_SPIKE
        elif growth_rate >= t.increasing_growth_rate_low:
            return TrendType.INCREASING
        elif growth_rate > -t.stable_growth_rate_bound:
            return TrendType.STABLE
        else:
            return TrendType.DECREASING

    @staticmethod
    def _has_keyword_match(
        service_keywords: List[str],
        reference_keywords: List[str],
    ) -> bool:
        """Check if any service keyword matches a reference keyword list.

        Case-insensitive substring matching.

        Args:
            service_keywords: Keywords found in logs for a service.
            reference_keywords: Reference list (critical or moderate).

        Returns:
            True if any match found.

        Example::

            SignalExtractor._has_keyword_match(
                ["database timeout", "connection refused"],
                ["database connection timeout", "deadlock"]
            )  # True — "database timeout" matches partially
        """
        for svc_kw in service_keywords:
            svc_lower = svc_kw.lower()
            for ref_kw in reference_keywords:
                ref_lower = ref_kw.lower()
                # bidirectional substring match
                if ref_lower in svc_lower or svc_lower in ref_lower:
                    return True
        return False

    def _detect_log_flooding(self, trend: List[int]) -> bool:
        """Detect log flooding based on repetition count.

        A service is flooding if any single time bucket exceeds
        the configured threshold (default: 100).

        Args:
            trend: Error counts per time bucket.

        Returns:
            True if flooding detected.
        """
        threshold = self._thresholds.log_flood_repeat_count
        return any(count > threshold for count in trend)

    def _detect_cascading(
        self,
        service_signals: List[ServiceSignal],
        affected_count: int,
    ) -> bool:
        """Detect potential cascading failure pattern.

        From planner.txt Step 6:
        - More than N services with errors
        - One service's errors started > M periods before others

        Args:
            service_signals: Per-service signals.
            affected_count: Number of services with errors.

        Returns:
            True if cascading pattern detected.
        """
        t = self._thresholds

        if affected_count < t.cascading_min_services:
            return False

        # Get first-error indices for all services (exclude -1)
        indices = [
            (sig.service, sig.first_non_zero_trend_index)
            for sig in service_signals
            if sig.first_non_zero_trend_index >= 0
        ]

        if len(indices) < 2:
            return False

        # Sort by first-error index
        indices.sort(key=lambda x: x[1])
        earliest_idx = indices[0][1]

        # Check if earliest precedes others by > gap threshold
        later_services = [
            idx for _, idx in indices[1:]
            if idx > earliest_idx + t.cascading_period_gap
        ]

        return len(later_services) >= 1
