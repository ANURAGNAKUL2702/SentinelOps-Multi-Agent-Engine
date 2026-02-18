"""
File: core/bottleneck_detector.py
Purpose: Detect architectural bottlenecks (fan-in, fan-out, sequential).
Dependencies: Standard library only
Performance: O(V) single pass, <5ms for 1000 services

Implements Algorithm 6: Detect Bottlenecks
Scans in/out degree and critical path percentages to flag bottlenecks.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    Bottleneck,
    BottleneckDetectionResult,
    BottleneckType,
    CriticalPathResult,
    GraphData,
    Severity,
    TraceSpan,
)
from agents.dependency_agent.telemetry import get_logger

logger = get_logger("dependency_agent.bottleneck_detector")


class BottleneckDetector:
    """Identify bottleneck services in the dependency graph.

    Three bottleneck categories:
      - fan_in:  in_degree > threshold (default 3)
      - fan_out: out_degree > threshold (default 5)
      - sequential: duration > bottleneck_pct of critical path

    Args:
        config: Agent configuration with thresholds.
    """

    def __init__(
        self, config: Optional[DependencyAgentConfig] = None
    ) -> None:
        self._config = config or DependencyAgentConfig()

    def detect(
        self,
        graph: GraphData,
        critical_path: Optional[CriticalPathResult] = None,
        spans: Optional[List[TraceSpan]] = None,
        correlation_id: str = "",
    ) -> BottleneckDetectionResult:
        """Detect bottlenecks across all services.

        Args:
            graph: Pre-built graph data with in/out degrees.
            critical_path: Optional critical path result.
            spans: Optional trace spans for sequential detection.
            correlation_id: Request correlation ID.

        Returns:
            BottleneckDetectionResult with list of detected bottlenecks.
        """
        start = time.perf_counter()
        bottlenecks: List[Bottleneck] = []

        thresholds = self._config.thresholds

        # Check every service node in the graph
        all_services = set(graph.in_degree.keys()) | set(
            graph.out_degree.keys()
        )

        for svc in all_services:
            in_deg = graph.in_degree.get(svc, 0)
            out_deg = graph.out_degree.get(svc, 0)

            # Fan-in bottleneck
            if in_deg > thresholds.fan_in_threshold:
                severity = self._severity_from_degree(
                    in_deg, thresholds.fan_in_threshold
                )
                bottlenecks.append(
                    Bottleneck(
                        service_name=svc,
                        bottleneck_type=BottleneckType.FAN_IN,
                        severity=severity,
                        fan_in_count=in_deg,
                        fan_out_count=out_deg,
                        reasoning=(
                            f"High fan-in: {in_deg} upstream "
                            f"dependencies (threshold: "
                            f"{thresholds.fan_in_threshold})"
                        ),
                    )
                )

            # Fan-out bottleneck
            if out_deg > thresholds.fan_out_threshold:
                severity = self._severity_from_degree(
                    out_deg, thresholds.fan_out_threshold
                )
                bottlenecks.append(
                    Bottleneck(
                        service_name=svc,
                        bottleneck_type=BottleneckType.FAN_OUT,
                        severity=severity,
                        fan_in_count=in_deg,
                        fan_out_count=out_deg,
                        reasoning=(
                            f"High fan-out: {out_deg} downstream "
                            f"dependencies (threshold: "
                            f"{thresholds.fan_out_threshold})"
                        ),
                    )
                )

        # Sequential bottlenecks from critical path
        if critical_path and spans:
            seq_bottlenecks = self._detect_sequential(
                critical_path, spans, thresholds.bottleneck_pct_threshold
            )
            bottlenecks.extend(seq_bottlenecks)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Bottleneck detection complete: "
            f"found={len(bottlenecks)}, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "bottleneck_detection",
                "context": {
                    "bottleneck_count": len(bottlenecks),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return BottleneckDetectionResult(
            bottlenecks=bottlenecks,
            detection_latency_ms=round(elapsed_ms, 2),
        )

    def _detect_sequential(
        self,
        critical_path: CriticalPathResult,
        spans: List[TraceSpan],
        pct_threshold: float,
    ) -> List[Bottleneck]:
        """Detect sequential bottlenecks on the critical path.

        A service is sequential bottleneck if its duration is
        > pct_threshold% of the critical path total duration.

        Args:
            critical_path: The critical path result.
            spans: All trace spans.
            pct_threshold: Percentage threshold for bottleneck.

        Returns:
            List of sequential Bottleneck entries.
        """
        if not critical_path.path or critical_path.total_duration_ms <= 0:
            return []

        # Build service → total duration from spans
        svc_duration: Dict[str, float] = {}
        for span in spans:
            svc_duration[span.service_name] = (
                svc_duration.get(span.service_name, 0.0)
                + span.duration_ms
            )

        bottlenecks: List[Bottleneck] = []
        for svc in critical_path.path:
            dur = svc_duration.get(svc, 0.0)
            pct = (dur / critical_path.total_duration_ms) * 100
            if pct > pct_threshold:
                bottlenecks.append(
                    Bottleneck(
                        service_name=svc,
                        bottleneck_type=BottleneckType.SEQUENTIAL,
                        severity=Severity.HIGH,
                        contributing_duration_ms=round(dur, 2),
                        bottleneck_percentage=round(pct, 1),
                        reasoning=(
                            f"Sequential bottleneck: {svc} "
                            f"contributes {pct:.1f}% of critical "
                            f"path duration ({dur:.0f}ms / "
                            f"{critical_path.total_duration_ms:.0f}ms)"
                        ),
                    )
                )

        return bottlenecks

    @staticmethod
    def _severity_from_degree(
        degree: int, threshold: int
    ) -> Severity:
        """Map degree magnitude to severity level.

        Args:
            degree: Actual degree count.
            threshold: Configured threshold.

        Returns:
            Severity enum value.
        """
        ratio = degree / max(threshold, 1)
        if ratio >= 3.0:
            return Severity.CRITICAL
        if ratio >= 2.0:
            return Severity.HIGH
        if ratio >= 1.5:
            return Severity.MEDIUM
        return Severity.LOW
