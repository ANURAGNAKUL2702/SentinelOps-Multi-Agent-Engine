"""
File: fallback.py
Purpose: Rule-based fallback for cascading failure detection + confidence.
Dependencies: Standard library only
Performance: O(V) for scanning, <5ms

Implements:
  Algorithm 7: Detect Cascading Failures
  Algorithm 8: Confidence Score (additive formula, 0–1 range)

Used when LLM is disabled or circuit breaker is open.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    Bottleneck,
    BottleneckDetectionResult,
    CascadePattern,
    CascadingFailureRisk,
    ClassificationResult,
    CriticalPathResult,
    DependencyAnalysisInput,
    DependencyAnalysisSummary,
    FailedServiceInfo,
    GraphBuildResult,
    GraphData,
    HealthStatus,
    ImpactAnalysisResult,
    Severity,
    SinglePointOfFailure,
    TraceAnalysisResult,
)
from agents.dependency_agent.telemetry import get_logger

logger = get_logger("dependency_agent.fallback")


class FallbackClassifier:
    """Rule-based classifier for dependency analysis.

    Deterministic alternative to LLM classification that produces
    cascade detection and confidence scores using preset rules.

    Args:
        config: Agent configuration with thresholds.
    """

    def __init__(
        self, config: Optional[DependencyAgentConfig] = None
    ) -> None:
        self._config = config or DependencyAgentConfig()

    def classify(
        self,
        input_data: DependencyAnalysisInput,
        graph_result: GraphBuildResult,
        impact: Optional[Dict[str, ImpactAnalysisResult]] = None,
        trace_result: Optional[TraceAnalysisResult] = None,
        bottleneck_result: Optional[BottleneckDetectionResult] = None,
        correlation_id: str = "",
    ) -> ClassificationResult:
        """Produce a classification from deterministic rules.

        Args:
            input_data: Raw input data.
            graph_result: Graph build result with graph data.
            impact: Per-service impact results.
            trace_result: Optional trace analysis result.
            bottleneck_result: Optional bottleneck detection result.
            correlation_id: Request correlation ID.

        Returns:
            ClassificationResult with all analysis fields.
        """
        start = time.perf_counter()

        graph = graph_result.graph
        impact = impact or {}

        # Build failed service info
        failed_service = self._failed_service_info(input_data)

        # Cascading failure risk
        cascade_risk = self._detect_cascading(
            input_data, graph, impact
        )

        # SPOFs
        spofs = self._detect_spofs(input_data, graph)

        # Get impact for the failed service
        failed_impact: Optional[ImpactAnalysisResult] = None
        if input_data.current_failure:
            failed_impact = impact.get(
                input_data.current_failure.service_name
            )

        # Get critical path from trace analysis
        critical_path: Optional[CriticalPathResult] = None
        if trace_result:
            critical_path = trace_result.critical_path

        # Get bottlenecks
        bottlenecks: List[Bottleneck] = []
        if bottleneck_result:
            bottlenecks = bottleneck_result.bottlenecks

        # Build summary
        summary = self._build_summary(
            input_data, graph_result
        )

        # Confidence score
        confidence = self._confidence_score(
            graph, impact, trace_result,
            bottleneck_result, graph_result.has_cycles,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Fallback classification complete: "
            f"confidence={confidence}, "
            f"spofs={len(spofs)}, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "classification",
                "context": {
                    "spof_count": len(spofs),
                    "confidence": confidence,
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return ClassificationResult(
            failed_service=failed_service,
            dependency_analysis=summary,
            impact_analysis=failed_impact,
            critical_path=critical_path,
            bottlenecks=bottlenecks,
            cascading_failure_risk=cascade_risk,
            single_points_of_failure=spofs,
            confidence_score=confidence,
            confidence_reasoning=self._confidence_reasoning(
                confidence, graph_result.has_cycles,
                trace_result, bottleneck_result,
            ),
            classification_source="deterministic",
            classification_latency_ms=round(elapsed_ms, 2),
        )

    # ── Algorithm 7: Cascading Failure Detection ────────────────

    def _detect_cascading(
        self,
        input_data: DependencyAnalysisInput,
        graph: GraphData,
        impact: Dict[str, ImpactAnalysisResult],
    ) -> CascadingFailureRisk:
        """Detect cascading failure risk based on current failure.

        Checks if downstream or upstream services of the failure
        are degraded/unhealthy, indicating cascade propagation.

        Args:
            input_data: Input with optional current_failure.
            graph: Pre-built graph data.
            impact: Per-service impact data.

        Returns:
            CascadingFailureRisk assessment.
        """
        if not input_data.current_failure:
            return CascadingFailureRisk()

        svc = input_data.current_failure.service_name

        # Build health lookup
        health_map: Dict[str, HealthStatus] = {}
        for node in input_data.service_graph.nodes:
            health_map[node.service_name] = node.health_status

        downstream = graph.adjacency_list.get(svc, [])
        upstream = graph.reverse_adjacency.get(svc, [])

        affected_downstream: List[str] = []
        affected_upstream: List[str] = []

        for ds in downstream:
            status = health_map.get(ds, HealthStatus.UNKNOWN)
            if status in (
                HealthStatus.DEGRADED,
                HealthStatus.UNHEALTHY,
            ):
                affected_downstream.append(ds)

        for us in upstream:
            status = health_map.get(us, HealthStatus.UNKNOWN)
            if status in (
                HealthStatus.DEGRADED,
                HealthStatus.UNHEALTHY,
            ):
                affected_upstream.append(us)

        all_affected = affected_downstream + affected_upstream
        is_cascading = len(all_affected) > 0

        # Determine cascade pattern
        if affected_downstream and affected_upstream:
            pattern = CascadePattern.BIDIRECTIONAL
        elif affected_downstream:
            pattern = CascadePattern.DOWNSTREAM_PROPAGATION
        elif affected_upstream:
            pattern = CascadePattern.UPSTREAM_PROPAGATION
        else:
            pattern = CascadePattern.ISOLATED

        # Cascade depth from blast radius
        svc_impact = impact.get(svc)
        depth = 0
        if svc_impact:
            depth = (
                len(svc_impact.blast_radius.directly_affected_services)
                + len(
                    svc_impact.blast_radius.indirectly_affected_services
                )
            )

        reasoning = ""
        if is_cascading:
            reasoning = (
                f"{svc} failure affects {len(all_affected)} services "
                f"via {pattern.value}"
            )

        return CascadingFailureRisk(
            is_cascading=is_cascading,
            cascade_pattern=pattern,
            cascade_depth=depth,
            affected_services=all_affected,
            reasoning=reasoning,
        )

    def _detect_spofs(
        self,
        input_data: DependencyAnalysisInput,
        graph: GraphData,
    ) -> List[SinglePointOfFailure]:
        """Detect single points of failure.

        A SPOF: instance_count == 1 AND dependents >= threshold.

        Args:
            input_data: Input with service nodes.
            graph: Pre-built graph data.

        Returns:
            List of SinglePointOfFailure entries.
        """
        threshold = self._config.thresholds.spof_min_dependents
        spofs: List[SinglePointOfFailure] = []

        for node in input_data.service_graph.nodes:
            svc = node.service_name
            if node.instance_count <= 1:
                dependents = graph.in_degree.get(svc, 0)
                if dependents >= threshold:
                    spofs.append(
                        SinglePointOfFailure(
                            service_name=svc,
                            reason=(
                                f"Single instance with "
                                f"{dependents} dependents"
                            ),
                            mitigation=(
                                f"Scale {svc} to at least 2 instances"
                            ),
                        )
                    )

        return spofs

    def _failed_service_info(
        self, input_data: DependencyAnalysisInput
    ) -> Optional[FailedServiceInfo]:
        """Extract info about the currently failing service.

        Args:
            input_data: Input with optional current_failure.

        Returns:
            FailedServiceInfo or None.
        """
        if not input_data.current_failure:
            return None

        svc = input_data.current_failure.service_name

        # Look up health status from nodes
        health = "unknown"
        for node in input_data.service_graph.nodes:
            if node.service_name == svc:
                health = node.health_status.value
                break

        return FailedServiceInfo(
            service_name=svc,
            failure_type=input_data.current_failure.failure_type.value,
            health_status=health,
        )

    def _build_summary(
        self,
        input_data: DependencyAnalysisInput,
        graph_result: GraphBuildResult,
    ) -> DependencyAnalysisSummary:
        """Build the analysis summary.

        Args:
            input_data: Input data.
            graph_result: Graph build result.

        Returns:
            DependencyAnalysisSummary.
        """
        return DependencyAnalysisSummary(
            total_services=len(input_data.service_graph.nodes),
            total_dependencies=len(input_data.service_graph.edges),
            graph_has_cycles=graph_result.has_cycles,
            max_dependency_depth=graph_result.max_depth,
        )

    # ── Algorithm 8: Confidence Score ───────────────────────────

    def _confidence_score(
        self,
        graph: GraphData,
        impact: Dict[str, ImpactAnalysisResult],
        trace_result: Optional[TraceAnalysisResult],
        bottleneck_result: Optional[BottleneckDetectionResult],
        has_cycles: bool,
    ) -> float:
        """Calculate overall confidence score.

        Formula (additive, max 1.0):
          base                     = 0.20
          + graph_built            = 0.15
          + blast_radius_computed  = 0.20
          + critical_path_found    = 0.20
          + bottlenecks_detected   = 0.15
          + no_cycles              = 0.10
                            Total  = 1.00

        Args:
            graph: Pre-built graph data.
            impact: Per-service impact data.
            trace_result: Optional trace analysis.
            bottleneck_result: Optional bottleneck detection.
            has_cycles: Whether cycles exist.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        score = 0.20  # base

        # Graph was built successfully
        if graph.adjacency_list or graph.node_map:
            score += 0.15

        # Blast radius computed
        if impact:
            score += 0.20

        # Critical path found
        if (
            trace_result
            and trace_result.critical_path
            and trace_result.critical_path.path
        ):
            score += 0.20

        # Bottleneck detection ran
        if bottleneck_result is not None:
            score += 0.15

        # No cycles is a positive signal
        if not has_cycles:
            score += 0.10

        return round(min(score, 1.0), 2)

    @staticmethod
    def _confidence_reasoning(
        confidence: float,
        has_cycles: bool,
        trace_result: Optional[TraceAnalysisResult],
        bottleneck_result: Optional[BottleneckDetectionResult],
    ) -> str:
        """Build human-readable confidence reasoning.

        Args:
            confidence: The calculated confidence score.
            has_cycles: Whether cycles were found.
            trace_result: Trace analysis result.
            bottleneck_result: Bottleneck detection result.

        Returns:
            Explanation string.
        """
        parts: List[str] = [
            f"Confidence {confidence:.2f}:"
        ]
        parts.append("graph built")

        if (
            trace_result
            and trace_result.critical_path
            and trace_result.critical_path.path
        ):
            parts.append("critical path identified")

        if bottleneck_result is not None:
            bn_count = len(bottleneck_result.bottlenecks)
            parts.append(f"{bn_count} bottleneck(s) analyzed")

        if has_cycles:
            parts.append("WARNING: cycles detected")
        else:
            parts.append("no cycles")

        return "; ".join(parts)
