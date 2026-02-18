"""
File: core/trace_analyzer.py
Purpose: Parse distributed traces, calculate critical path, find slow spans.
Dependencies: Standard library only
Performance: O(n log n) where n=spans, <20ms for 1000 spans

Implements Algorithm 4: Critical Path Analysis
Uses span tree traversal to find the longest-duration path
through the distributed trace.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    CriticalPathResult,
    DependencyAnalysisInput,
    DistributedTrace,
    SlowSpan,
    TraceAnalysisResult,
    TraceSpan,
)
from agents.dependency_agent.telemetry import get_logger

logger = get_logger("dependency_agent.trace_analyzer")


class TraceAnalyzer:
    """Analyzes distributed traces to find critical paths and slow spans.

    Builds a span tree from parent-child relationships, then uses
    DFS to find the longest-duration path (critical path).

    Args:
        config: Agent configuration with slow span threshold.

    Example::

        analyzer = TraceAnalyzer(DependencyAgentConfig())
        result = analyzer.analyze(input_data)
        print(result.critical_path)
    """

    def __init__(
        self, config: Optional[DependencyAgentConfig] = None
    ) -> None:
        self._config = config or DependencyAgentConfig()

    def analyze(
        self,
        input_data: DependencyAnalysisInput,
        correlation_id: str = "",
    ) -> TraceAnalysisResult:
        """Analyze all traces in the input.

        Args:
            input_data: Input containing distributed traces.
            correlation_id: Request correlation ID.

        Returns:
            TraceAnalysisResult with critical path and slow spans.
        """
        start = time.perf_counter()

        if not input_data.traces:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return TraceAnalysisResult(
                analysis_latency_ms=round(elapsed_ms, 2),
            )

        # Analyze the first (primary) trace
        trace = input_data.traces[0]

        critical_path = self._find_critical_path(trace)
        slow_spans = self._find_slow_spans(trace)
        latency_contributions = self._calculate_latency_contributions(trace)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Trace analysis complete: "
            f"critical_path={critical_path.path if critical_path else []}, "
            f"slow_spans={len(slow_spans)}, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "trace_analysis",
                "context": {
                    "trace_count": len(input_data.traces),
                    "span_count": len(trace.spans),
                    "slow_span_count": len(slow_spans),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return TraceAnalysisResult(
            critical_path=critical_path,
            slow_spans=slow_spans,
            service_latency_contributions=latency_contributions,
            analysis_latency_ms=round(elapsed_ms, 2),
        )

    # ── Algorithm 4: Critical Path Analysis ─────────────────────

    def _find_critical_path(
        self, trace: DistributedTrace
    ) -> Optional[CriticalPathResult]:
        """Find the critical path (longest duration path) through the trace.

        Builds a span tree from parent-child relationships,
        then DFS to find the path with maximum total duration.

        Complexity: O(n) where n=spans

        Args:
            trace: Distributed trace with spans.

        Returns:
            CriticalPathResult or None if no spans.
        """
        if not trace.spans:
            return None

        # Build span lookup and children map
        span_map: Dict[str, TraceSpan] = {}
        children: Dict[str, List[str]] = {}
        root_span: Optional[TraceSpan] = None

        for span in trace.spans:
            span_map[span.span_id] = span
            children[span.span_id] = []

        for span in trace.spans:
            if span.parent_span_id is None:
                root_span = span
            elif span.parent_span_id in children:
                children[span.parent_span_id].append(span.span_id)

        if root_span is None:
            # Fallback: use first span as root
            root_span = trace.spans[0]

        # DFS to find longest path
        total_duration, path_services = self._dfs_critical_path(
            root_span.span_id, span_map, children
        )

        # Remove consecutive duplicates in path
        deduped_path: List[str] = []
        for svc in path_services:
            if not deduped_path or deduped_path[-1] != svc:
                deduped_path.append(svc)

        # Find bottleneck (span with longest individual duration)
        bottleneck_span = max(trace.spans, key=lambda s: s.duration_ms)
        bottleneck_pct = 0.0
        if total_duration > 0:
            bottleneck_pct = (
                bottleneck_span.duration_ms / total_duration
            ) * 100

        return CriticalPathResult(
            path=deduped_path,
            total_duration_ms=round(total_duration, 2),
            bottleneck_service=bottleneck_span.service_name,
            bottleneck_duration_ms=round(
                bottleneck_span.duration_ms, 2
            ),
            bottleneck_percentage=round(
                min(bottleneck_pct, 100.0), 1
            ),
        )

    def _dfs_critical_path(
        self,
        span_id: str,
        span_map: Dict[str, TraceSpan],
        children: Dict[str, List[str]],
    ) -> Tuple[float, List[str]]:
        """DFS to find the longest-duration path from a span.

        Args:
            span_id: Current span ID.
            span_map: span_id → TraceSpan lookup.
            children: span_id → list of child span IDs.

        Returns:
            Tuple of (total_duration, list_of_service_names).
        """
        span = span_map[span_id]
        child_ids = children.get(span_id, [])

        if not child_ids:
            return span.duration_ms, [span.service_name]

        max_duration = 0.0
        max_path: List[str] = []

        for child_id in child_ids:
            child_dur, child_path = self._dfs_critical_path(
                child_id, span_map, children
            )
            if child_dur > max_duration:
                max_duration = child_dur
                max_path = child_path

        return span.duration_ms + max_duration, [
            span.service_name
        ] + max_path

    # ── Slow Span Detection ─────────────────────────────────────

    def _find_slow_spans(
        self, trace: DistributedTrace
    ) -> List[SlowSpan]:
        """Find spans exceeding the slow threshold.

        Args:
            trace: Distributed trace.

        Returns:
            List of SlowSpan objects.
        """
        threshold = self._config.thresholds.slow_span_threshold_ms
        slow: List[SlowSpan] = []

        for span in trace.spans:
            if span.duration_ms > threshold:
                slow.append(SlowSpan(
                    span_id=span.span_id,
                    service_name=span.service_name,
                    operation=span.operation,
                    duration_ms=span.duration_ms,
                    is_error=span.error,
                ))

        return slow

    # ── Latency Contribution Calculation ────────────────────────

    def _calculate_latency_contributions(
        self, trace: DistributedTrace
    ) -> Dict[str, float]:
        """Calculate each service's contribution to total trace duration.

        Args:
            trace: Distributed trace.

        Returns:
            Dict mapping service_name → total_duration_ms.
        """
        contributions: Dict[str, float] = {}

        for span in trace.spans:
            name = span.service_name
            contributions[name] = contributions.get(name, 0.0) + span.duration_ms

        return {
            k: round(v, 2) for k, v in contributions.items()
        }
