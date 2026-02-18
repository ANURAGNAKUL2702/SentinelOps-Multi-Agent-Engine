"""
File: core/evidence_aggregator.py
Purpose: Algorithm 1 — Evidence Aggregation O(n).
Dependencies: Schema models only.
Performance: <10ms for typical input, O(n) where n=total findings.

Collects findings from log, metrics, and dependency agents,
normalizes them into EvidenceItem objects, and detects cross-agent
correlations.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    CrossAgentCorrelation,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    HypothesisAgentInput,
    Severity,
)
from agents.hypothesis_agent.telemetry import get_logger

logger = get_logger("hypothesis_agent.evidence_aggregator")


class EvidenceAggregator:
    """Aggregates evidence from all upstream agent findings.

    Pipeline::

        LogFindings  ──┐
        MetricFindings ──┤──  normalize ──  correlate ──  AggregatedEvidence
        DepFindings  ──┘

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[HypothesisAgentConfig] = None
    ) -> None:
        self._config = config or HypothesisAgentConfig()

    def aggregate(
        self,
        input_data: HypothesisAgentInput,
        correlation_id: str = "",
    ) -> AggregatedEvidence:
        """Aggregate all agent findings into a unified evidence set.

        Args:
            input_data: Input with findings from all agents.
            correlation_id: Request correlation ID.

        Returns:
            AggregatedEvidence with all items and correlations.
        """
        start = time.perf_counter()
        evidence_items: List[EvidenceItem] = []
        sources: Set[EvidenceSource] = set()

        # ── Extract from log findings ───────────────────────────
        log_evidence = self._extract_log_evidence(input_data)
        if log_evidence:
            evidence_items.extend(log_evidence)
            sources.add(EvidenceSource.LOG_AGENT)

        # ── Extract from metric findings ────────────────────────
        metric_evidence = self._extract_metric_evidence(input_data)
        if metric_evidence:
            evidence_items.extend(metric_evidence)
            sources.add(EvidenceSource.METRICS_AGENT)

        # ── Extract from dependency findings ────────────────────
        dep_evidence = self._extract_dependency_evidence(input_data)
        if dep_evidence:
            evidence_items.extend(dep_evidence)
            sources.add(EvidenceSource.DEPENDENCY_AGENT)

        # ── Cap evidence items ──────────────────────────────────
        max_items = self._config.performance.max_evidence_items
        if len(evidence_items) > max_items:
            evidence_items = evidence_items[:max_items]

        # ── Detect cross-agent correlations ─────────────────────
        correlations = self._detect_correlations(
            evidence_items, sources
        )

        # ── Compute summary stats ──────────────────────────────
        strong_count = sum(
            1
            for e in evidence_items
            if e.strength == EvidenceStrength.STRONG
        )
        dominant_severity = self._compute_dominant_severity(
            evidence_items
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Evidence aggregation complete — "
            f"{len(evidence_items)} items, "
            f"{len(correlations)} correlations, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "evidence_aggregation",
                "context": {
                    "total": len(evidence_items),
                    "strong": strong_count,
                    "sources": [s.value for s in sorted(sources, key=lambda x: x.value)],
                },
            },
        )

        return AggregatedEvidence(
            evidence_items=evidence_items,
            correlations=correlations,
            total_evidence_count=len(evidence_items),
            strong_evidence_count=strong_count,
            sources_represented=sorted(sources, key=lambda x: x.value),
            dominant_severity=dominant_severity,
            aggregation_latency_ms=round(elapsed_ms, 2),
        )

    # ── Log evidence extraction ─────────────────────────────────

    def _extract_log_evidence(
        self, input_data: HypothesisAgentInput
    ) -> List[EvidenceItem]:
        """Extract evidence from log agent findings."""
        items: List[EvidenceItem] = []
        log = input_data.log_findings

        # Suspicious services
        for svc in log.suspicious_services:
            svc_name = svc.get("service", "unknown")
            error_count = svc.get("error_count", 0)
            severity_hint = svc.get("severity_hint", "medium")
            keywords = svc.get("error_keywords_detected", [])

            severity = self._map_severity(severity_hint)
            strength = (
                EvidenceStrength.STRONG
                if error_count > 50 or severity in (Severity.CRITICAL, Severity.HIGH)
                else EvidenceStrength.MODERATE
                if error_count > 10
                else EvidenceStrength.WEAK
            )

            desc = (
                f"Service '{svc_name}' has {error_count} errors"
            )
            if keywords:
                desc += f" (keywords: {', '.join(keywords[:5])})"

            items.append(EvidenceItem(
                source=EvidenceSource.LOG_AGENT,
                description=desc,
                severity=severity,
                strength=strength,
                raw_data=svc,
            ))

        # System-wide signals
        if log.system_wide_spike:
            items.append(EvidenceItem(
                source=EvidenceSource.LOG_AGENT,
                description="System-wide error spike detected",
                severity=Severity.HIGH,
                strength=EvidenceStrength.STRONG,
            ))

        if log.potential_upstream_failure:
            items.append(EvidenceItem(
                source=EvidenceSource.LOG_AGENT,
                description="Potential upstream failure detected",
                severity=Severity.HIGH,
                strength=EvidenceStrength.MODERATE,
            ))

        if log.database_errors_detected:
            items.append(EvidenceItem(
                source=EvidenceSource.LOG_AGENT,
                description="Database-related errors detected in logs",
                severity=Severity.HIGH,
                strength=EvidenceStrength.STRONG,
                raw_data={"database_errors": True},
            ))

        return items

    # ── Metric evidence extraction ──────────────────────────────

    def _extract_metric_evidence(
        self, input_data: HypothesisAgentInput
    ) -> List[EvidenceItem]:
        """Extract evidence from metrics agent findings."""
        items: List[EvidenceItem] = []
        metrics = input_data.metric_findings

        # Anomalous metrics
        for metric in metrics.anomalous_metrics:
            metric_name = metric.get("metric_name", "unknown")
            zscore = metric.get("zscore", 0.0)
            severity_raw = metric.get("severity", "medium")
            anomaly_type = metric.get("anomaly_type", "unknown")

            severity = self._map_severity(severity_raw)
            strength = (
                EvidenceStrength.STRONG
                if abs(zscore) > 3.0
                else EvidenceStrength.MODERATE
                if abs(zscore) > 2.0
                else EvidenceStrength.WEAK
            )

            items.append(EvidenceItem(
                source=EvidenceSource.METRICS_AGENT,
                description=(
                    f"Anomalous metric '{metric_name}' "
                    f"(z-score={zscore:.2f}, type={anomaly_type})"
                ),
                severity=severity,
                strength=strength,
                raw_data=metric,
            ))

        # Correlations
        for corr in metrics.correlations:
            m1 = corr.get("metric_1", "?")
            m2 = corr.get("metric_2", "?")
            coeff = corr.get("correlation_coefficient", 0.0)

            if abs(coeff) > 0.7:
                items.append(EvidenceItem(
                    source=EvidenceSource.METRICS_AGENT,
                    description=(
                        f"Strong correlation between "
                        f"'{m1}' and '{m2}' (r={coeff:.2f})"
                    ),
                    severity=Severity.MEDIUM,
                    strength=EvidenceStrength.MODERATE,
                    raw_data=corr,
                ))

        # System-level signals
        if metrics.resource_saturation:
            items.append(EvidenceItem(
                source=EvidenceSource.METRICS_AGENT,
                description="Resource saturation detected",
                severity=Severity.CRITICAL,
                strength=EvidenceStrength.STRONG,
            ))

        if metrics.cascading_degradation:
            items.append(EvidenceItem(
                source=EvidenceSource.METRICS_AGENT,
                description="Cascading degradation detected in metrics",
                severity=Severity.HIGH,
                strength=EvidenceStrength.STRONG,
            ))

        return items

    # ── Dependency evidence extraction ──────────────────────────

    def _extract_dependency_evidence(
        self, input_data: HypothesisAgentInput
    ) -> List[EvidenceItem]:
        """Extract evidence from dependency agent findings."""
        items: List[EvidenceItem] = []
        dep = input_data.dependency_findings

        # Failed service
        if dep.failed_service:
            items.append(EvidenceItem(
                source=EvidenceSource.DEPENDENCY_AGENT,
                description=(
                    f"Primary failed service: '{dep.failed_service}'"
                ),
                severity=Severity.CRITICAL,
                strength=EvidenceStrength.STRONG,
            ))

        # Cascading failure
        if dep.is_cascading:
            items.append(EvidenceItem(
                source=EvidenceSource.DEPENDENCY_AGENT,
                description=(
                    f"Cascading failure detected — "
                    f"pattern={dep.cascade_pattern}, "
                    f"blast_radius={dep.blast_radius_count}"
                ),
                severity=Severity.CRITICAL,
                strength=EvidenceStrength.STRONG,
                raw_data={
                    "cascade_pattern": dep.cascade_pattern,
                    "blast_radius": dep.blast_radius_count,
                },
            ))

        # SPOFs
        for spof in dep.single_points_of_failure:
            items.append(EvidenceItem(
                source=EvidenceSource.DEPENDENCY_AGENT,
                description=(
                    f"Single point of failure: '{spof}'"
                ),
                severity=Severity.HIGH,
                strength=EvidenceStrength.MODERATE,
            ))

        # Bottlenecks
        for bn in dep.bottleneck_services:
            items.append(EvidenceItem(
                source=EvidenceSource.DEPENDENCY_AGENT,
                description=f"Bottleneck service: '{bn}'",
                severity=Severity.MEDIUM,
                strength=EvidenceStrength.MODERATE,
            ))

        # Graph cycles
        if dep.graph_has_cycles:
            items.append(EvidenceItem(
                source=EvidenceSource.DEPENDENCY_AGENT,
                description="Dependency graph contains cycles",
                severity=Severity.MEDIUM,
                strength=EvidenceStrength.WEAK,
            ))

        return items

    # ── Cross-agent correlation ─────────────────────────────────

    def _detect_correlations(
        self,
        items: List[EvidenceItem],
        sources: Set[EvidenceSource],
    ) -> List[CrossAgentCorrelation]:
        """Detect correlations across agent findings.

        Looks for:
        1. Same service mentioned by multiple agents.
        2. Log DB errors + metric anomalies (DB pattern).
        3. Cascading failure in deps + system-wide log spike.
        """
        correlations: List[CrossAgentCorrelation] = []

        if len(sources) < 2:
            return correlations

        # Collect services mentioned per source
        services_by_source: Dict[EvidenceSource, Set[str]] = {}
        for idx, item in enumerate(items):
            # Extract service names from descriptions
            for svc_candidate in self._extract_service_names(
                item.description
            ):
                services_by_source.setdefault(
                    item.source, set()
                ).add(svc_candidate)

        # Find services mentioned by multiple agents
        all_sources_list = list(services_by_source.keys())
        for i in range(len(all_sources_list)):
            for j in range(i + 1, len(all_sources_list)):
                src_a = all_sources_list[i]
                src_b = all_sources_list[j]
                overlap = (
                    services_by_source.get(src_a, set())
                    & services_by_source.get(src_b, set())
                )
                if overlap:
                    correlations.append(CrossAgentCorrelation(
                        sources=[src_a, src_b],
                        description=(
                            f"Services mentioned by both "
                            f"{src_a.value} and {src_b.value}: "
                            f"{', '.join(sorted(overlap))}"
                        ),
                        correlation_score=min(
                            0.9, 0.3 * len(overlap)
                        ),
                    ))

        # Check for DB error + metric anomaly correlation
        has_db_log = any(
            "database" in item.description.lower()
            for item in items
            if item.source == EvidenceSource.LOG_AGENT
        )
        has_metric_anomaly = any(
            item.source == EvidenceSource.METRICS_AGENT
            and item.strength == EvidenceStrength.STRONG
            for item in items
        )
        if has_db_log and has_metric_anomaly:
            correlations.append(CrossAgentCorrelation(
                sources=[
                    EvidenceSource.LOG_AGENT,
                    EvidenceSource.METRICS_AGENT,
                ],
                description=(
                    "Database errors in logs correlate with "
                    "strong metric anomalies"
                ),
                correlation_score=0.8,
            ))

        # Check for cascading failure + system-wide spike
        has_cascade = any(
            "cascading" in item.description.lower()
            for item in items
            if item.source == EvidenceSource.DEPENDENCY_AGENT
        )
        has_spike = any(
            "system-wide" in item.description.lower()
            for item in items
            if item.source == EvidenceSource.LOG_AGENT
        )
        if has_cascade and has_spike:
            correlations.append(CrossAgentCorrelation(
                sources=[
                    EvidenceSource.LOG_AGENT,
                    EvidenceSource.DEPENDENCY_AGENT,
                ],
                description=(
                    "Cascading failure aligns with "
                    "system-wide error spike"
                ),
                correlation_score=0.85,
            ))

        return correlations

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_service_names(description: str) -> List[str]:
        """Extract service names from evidence descriptions.

        Looks for patterns like 'Service X' or "'service_name'".
        """
        names: List[str] = []
        # Match single-quoted words
        import re
        for match in re.finditer(r"'([^']+)'", description):
            names.append(match.group(1))
        return names

    @staticmethod
    def _map_severity(raw: str) -> Severity:
        """Map a raw severity string to Severity enum."""
        raw_lower = raw.lower() if isinstance(raw, str) else "medium"
        if raw_lower in ("critical",):
            return Severity.CRITICAL
        if raw_lower in ("high",):
            return Severity.HIGH
        if raw_lower in ("low",):
            return Severity.LOW
        return Severity.MEDIUM

    @staticmethod
    def _compute_dominant_severity(
        items: List[EvidenceItem],
    ) -> Severity:
        """Find the most common severity across evidence items."""
        if not items:
            return Severity.MEDIUM

        counts: Dict[Severity, int] = {}
        for item in items:
            counts[item.severity] = counts.get(
                item.severity, 0
            ) + 1

        # Priority order: critical > high > medium > low
        priority = [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
        ]

        max_count = max(counts.values())
        for sev in priority:
            if counts.get(sev, 0) == max_count:
                return sev

        return Severity.MEDIUM
