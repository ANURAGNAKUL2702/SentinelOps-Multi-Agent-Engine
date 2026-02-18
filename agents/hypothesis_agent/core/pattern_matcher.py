"""
File: core/pattern_matcher.py
Purpose: Algorithm 2 — Pattern Matching O(p*e).
Dependencies: Schema models only.
Performance: <5ms, O(p*e) where p=patterns, e=evidence.

Matches aggregated evidence against the known failure pattern library
(5 patterns). Returns scored PatternMatch objects.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    IncidentCategory,
    KnownPattern,
    PatternIndicator,
    PatternMatch,
    PatternName,
)
from agents.hypothesis_agent.telemetry import get_logger

logger = get_logger("hypothesis_agent.pattern_matcher")


# ═══════════════════════════════════════════════════════════════
#  PATTERN LIBRARY (5 known failure patterns)
# ═══════════════════════════════════════════════════════════════


def _build_pattern_library() -> List[KnownPattern]:
    """Build the known failure pattern library.

    Each pattern has indicators with POSITIVE weights (must match) and
    patterns are designed to be MUTUALLY EXCLUSIVE — only the correct
    scenario should score significantly above the min_match_score.
    """
    return [
        KnownPattern(
            pattern_name=PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
            description=(
                "Database connection pool reaches capacity, causing "
                "timeouts and cascading failures in dependent services"
            ),
            category=IncidentCategory.DATABASE,
            indicators=[
                PatternIndicator(
                    indicator_name="db_specific_errors",
                    weight=0.30,
                ),
                PatternIndicator(
                    indicator_name="db_pool_keywords",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="db_metric_anomaly",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="no_oom_errors",
                    weight=0.10,
                ),
                PatternIndicator(
                    indicator_name="no_packet_loss",
                    weight=0.10,
                ),
            ],
            typical_mttr_minutes=45.0,
            validation_tests=[
                "Check database connection pool metrics",
                "Review active/idle connection counts",
                "Verify connection timeout settings",
            ],
        ),
        KnownPattern(
            pattern_name=PatternName.MEMORY_LEAK,
            description=(
                "Gradual memory increase until OOM, causing service "
                "restarts and request failures"
            ),
            category=IncidentCategory.APPLICATION,
            indicators=[
                PatternIndicator(
                    indicator_name="oom_keywords_strict",
                    weight=0.30,
                ),
                PatternIndicator(
                    indicator_name="memory_heap_anomaly",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="gc_overhead_keywords",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="no_cpu_saturation",
                    weight=0.10,
                ),
                PatternIndicator(
                    indicator_name="no_packet_loss",
                    weight=0.10,
                ),
            ],
            typical_mttr_minutes=60.0,
            validation_tests=[
                "Check heap memory trends over time",
                "Look for monotonically increasing allocations",
                "Review recent code deployments for leak patterns",
            ],
        ),
        KnownPattern(
            pattern_name=PatternName.NETWORK_PARTITION,
            description=(
                "Network segmentation causing connectivity loss "
                "between service groups"
            ),
            category=IncidentCategory.NETWORK,
            indicators=[
                PatternIndicator(
                    indicator_name="network_specific_errors",
                    weight=0.30,
                ),
                PatternIndicator(
                    indicator_name="packet_loss_detected",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="distributed_failure",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="no_oom_errors",
                    weight=0.10,
                ),
                PatternIndicator(
                    indicator_name="no_db_specific_errors",
                    weight=0.10,
                ),
            ],
            typical_mttr_minutes=30.0,
            validation_tests=[
                "Run network connectivity checks between AZs",
                "Check for recent network config changes",
                "Verify DNS resolution across services",
            ],
        ),
        KnownPattern(
            pattern_name=PatternName.CPU_SPIKE,
            description=(
                "Sudden CPU saturation causing thread pool exhaustion, "
                "request timeouts, and cascading failures"
            ),
            category=IncidentCategory.INFRASTRUCTURE,
            indicators=[
                PatternIndicator(
                    indicator_name="cpu_saturation_keywords",
                    weight=0.30,
                ),
                PatternIndicator(
                    indicator_name="thread_pool_keywords",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="cpu_metric_anomaly",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="no_oom_errors",
                    weight=0.10,
                ),
                PatternIndicator(
                    indicator_name="no_packet_loss",
                    weight=0.10,
                ),
            ],
            typical_mttr_minutes=20.0,
            validation_tests=[
                "Check CPU utilization and load average",
                "Review thread pool metrics and queue depth",
                "Identify CPU-intensive batch jobs or runaway processes",
            ],
        ),
        KnownPattern(
            pattern_name=PatternName.DEPLOYMENT_ISSUE,
            description=(
                "Recent deployment introduced a bug or misconfiguration "
                "causing service failures"
            ),
            category=IncidentCategory.DEPLOYMENT,
            indicators=[
                PatternIndicator(
                    indicator_name="error_spike_after_deploy",
                    weight=0.35,
                ),
                PatternIndicator(
                    indicator_name="single_service_errors",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="high_error_rate",
                    weight=0.2,
                ),
                PatternIndicator(
                    indicator_name="no_infrastructure_issues",
                    weight=0.2,
                ),
            ],
            typical_mttr_minutes=20.0,
            validation_tests=[
                "Check recent deployment timestamps",
                "Compare error rates before/after deploy",
                "Review deployment changelog for breaking changes",
            ],
        ),
        KnownPattern(
            pattern_name=PatternName.CONFIGURATION_ERROR,
            description=(
                "Misconfiguration of service parameters causing "
                "unexpected behavior or failures"
            ),
            category=IncidentCategory.CONFIGURATION,
            indicators=[
                PatternIndicator(
                    indicator_name="config_keywords",
                    weight=0.3,
                ),
                PatternIndicator(
                    indicator_name="sudden_error_onset",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="specific_error_pattern",
                    weight=0.25,
                ),
                PatternIndicator(
                    indicator_name="limited_blast_radius",
                    weight=0.2,
                ),
            ],
            typical_mttr_minutes=15.0,
            validation_tests=[
                "Review recent config changes",
                "Check environment variable settings",
                "Compare config against known-good baseline",
            ],
        ),
    ]


PATTERN_LIBRARY: List[KnownPattern] = _build_pattern_library()


class PatternMatcher:
    """Matches evidence against the known failure pattern library.

    For each pattern, checks evidence items against indicators
    and produces a weighted match score.

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[HypothesisAgentConfig] = None
    ) -> None:
        self._config = config or HypothesisAgentConfig()
        self._patterns = PATTERN_LIBRARY

    def match(
        self,
        evidence: AggregatedEvidence,
        correlation_id: str = "",
    ) -> List[PatternMatch]:
        """Match evidence against all known patterns.

        Args:
            evidence: Aggregated evidence from all agents.
            correlation_id: Request correlation ID.

        Returns:
            List of PatternMatch sorted by match_score descending.
        """
        start = time.perf_counter()
        matches: List[PatternMatch] = []

        for pattern in self._patterns:
            match = self._match_pattern(pattern, evidence)
            if (
                match.match_score
                >= self._config.patterns.min_match_score
                and match.matched_indicators
                >= self._config.patterns.min_indicators_matched
            ):
                matches.append(match)

        # Sort by score descending
        matches.sort(key=lambda m: m.match_score, reverse=True)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Pattern matching complete — "
            f"{len(matches)}/{len(self._patterns)} patterns matched, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "pattern_matching",
                "context": {
                    "matches": len(matches),
                    "total_patterns": len(self._patterns),
                },
            },
        )

        return matches

    def _match_pattern(
        self,
        pattern: KnownPattern,
        evidence: AggregatedEvidence,
    ) -> PatternMatch:
        """Match a single pattern against evidence.

        Args:
            pattern: Known pattern to check.
            evidence: Aggregated evidence.

        Returns:
            PatternMatch with score and indicator counts.
        """
        matched_count = 0
        total_weight = 0.0
        matched_weight = 0.0

        for indicator in pattern.indicators:
            total_weight += indicator.weight
            if self._check_indicator(
                indicator.indicator_name, evidence
            ):
                indicator.matched = True
                matched_count += 1
                matched_weight += indicator.weight

        score = (
            matched_weight / total_weight
            if total_weight > 0
            else 0.0
        )

        description = ""
        if score >= self._config.patterns.high_match_threshold:
            description = (
                f"Strong match for {pattern.pattern_name.value}: "
                f"{matched_count}/{len(pattern.indicators)} indicators"
            )
        elif score >= self._config.patterns.min_match_score:
            description = (
                f"Partial match for {pattern.pattern_name.value}: "
                f"{matched_count}/{len(pattern.indicators)} indicators"
            )

        return PatternMatch(
            pattern_name=pattern.pattern_name,
            match_score=round(score, 3),
            matched_indicators=matched_count,
            total_indicators=len(pattern.indicators),
            category=pattern.category,
            description=description,
        )

    def _check_indicator(
        self,
        indicator_name: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """Check if an indicator is present in the evidence.

        Args:
            indicator_name: Name of the indicator to check.
            evidence: Aggregated evidence.

        Returns:
            True if indicator found in evidence.
        """
        items = evidence.evidence_items
        descriptions_lower = [
            e.description.lower() for e in items
        ]
        all_text = " ".join(descriptions_lower)

        checker = self._INDICATOR_CHECKERS.get(indicator_name)
        if checker:
            return checker(self, items, all_text, evidence)
        return False

    # ── Indicator checker methods ───────────────────────────────
    #
    # Each checker matches a SPECIFIC failure signature.
    # Indicators are designed to be MUTUALLY EXCLUSIVE so that
    # only the correct scenario pattern scores highly.
    #

    # ── DATABASE-SPECIFIC checkers ──────────────────────────────

    def _check_db_specific_errors(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if DB-specific error keywords appear (NOT generic 'timeout')."""
        db_keywords = [
            "database connection", "connection pool exhausted",
            "query timeout", "query execution", "slow query",
            "deadlock", "hikaricp", "max_connections",
            "lock wait", "sqlexception", "sqltransient",
        ]
        return any(kw in all_text for kw in db_keywords)

    def _check_db_pool_keywords(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if DB connection pool keywords appear."""
        pool_kw = [
            "connection pool", "pool exhausted", "pool wait",
            "connection acquisition", "hikaricp",
        ]
        return any(kw in all_text for kw in pool_kw)

    def _check_db_metric_anomaly(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if DB-specific metrics show anomalies (query_duration, db_connections)."""
        db_metrics = {"db_query_duration_ms", "db_active_connections", "db_pool_wait_ms"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in db_metrics:
                    return True
                # Also check description for DB metric references
                desc = e.description.lower()
                if any(m in desc for m in ["db_query", "db_active", "db_pool"]):
                    return True
        return False

    # ── MEMORY LEAK-SPECIFIC checkers ───────────────────────────

    def _check_oom_keywords_strict(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE only if OOM-specific keywords appear (OutOfMemoryError, OOMKilled).

        This is UNIQUE to memory leaks — OOM errors never appear in
        CPU spike, network, or database scenarios.
        """
        oom_kw = [
            "outofmemoryerror", "oomkilled", "oom killer",
            "out of memory", "exit code 137",
        ]
        return any(kw in all_text for kw in oom_kw)

    def _check_memory_heap_anomaly(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if heap/memory-specific metrics are anomalous."""
        memory_metrics = {"heap_used_mb", "memory_percent", "gc_overhead_percent", "gc_pause_ms"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in memory_metrics:
                    return True
        # Also check for heap/memory keywords in log evidence
        heap_kw = ["heap", "gc overhead", "memory pool", "old gen"]
        return any(kw in all_text for kw in heap_kw)

    def _check_gc_overhead_keywords(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if GC overhead keywords appear (unique to memory leaks)."""
        gc_kw = ["gc overhead", "garbage collection", "major collection", "gc_overhead"]
        return any(kw in all_text for kw in gc_kw)

    # ── NETWORK-SPECIFIC checkers ───────────────────────────────

    def _check_network_specific_errors(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if network-specific error patterns appear.

        Matches 'connection reset', 'no route', 'unreachable',
        'socket timeout', 'DNS' — NOT generic 'connection timeout'
        which also appears in DB scenarios.
        """
        net_kw = [
            "connection reset by peer", "no route to host",
            "unreachable", "socket timeout", "broken pipe",
            "dns resolution", "tcp retransmission",
            "packet loss", "network partition",
        ]
        return any(kw in all_text for kw in net_kw)

    def _check_packet_loss_detected(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if packet_loss or tcp_retransmissions metrics are anomalous.

        This is UNIQUE to network issues — packet loss never appears
        in memory leak, CPU spike, or database scenarios.
        """
        net_metrics = {"packet_loss_percent", "tcp_retransmissions"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in net_metrics:
                    return True
        return "packet" in all_text or "retransmission" in all_text

    def _check_distributed_failure(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if multiple services show SIMULTANEOUS errors (distributed)."""
        services = set()
        for e in items:
            if e.source == EvidenceSource.LOG_AGENT:
                svc = e.raw_data.get("service", "")
                if svc:
                    services.add(svc)
        return len(services) >= 3  # 3+ services affected simultaneously

    # ── CPU SPIKE-SPECIFIC checkers ─────────────────────────────

    def _check_cpu_saturation_keywords(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if CPU-saturation-specific keywords appear.

        Matches 'CPU at 100%', 'high CPU', 'CPU saturated',
        'cpu-bound', 'worker threads blocked'.
        """
        cpu_kw = [
            "cpu", "cpu utilisation", "cpu saturat",
            "high cpu", "cpu-bound", "worker threads blocked",
            "context switch", "load average",
        ]
        return any(kw in all_text for kw in cpu_kw)

    def _check_thread_pool_keywords(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if thread-pool-specific keywords appear (unique to CPU spikes)."""
        tp_kw = [
            "thread pool", "thread starvation", "rejected execution",
            "queue full", "queue depth", "0 idle workers",
        ]
        return any(kw in all_text for kw in tp_kw)

    def _check_cpu_metric_anomaly(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if CPU or thread_pool metrics show anomalies."""
        cpu_metrics = {"cpu_percent", "thread_pool_active_pct"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in cpu_metrics:
                    # Check that it's a significant anomaly
                    z_score = abs(e.raw_data.get("zscore", 0))
                    if z_score >= 2.0:
                        return True
        return False

    # ── NEGATIVE (anti-) indicators ─────────────────────────────
    # These CONFIRM a hypothesis by ruling out OTHER scenarios.

    def _check_no_oom_errors(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if no OOM-related METRICS are anomalous → NOT a memory leak.

        Uses metric evidence only (root-service metrics) to avoid
        false negatives from cascade services' symptomatic log text.
        """
        oom_metrics = {"heap_used_mb", "gc_overhead_percent", "gc_pause_ms"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in oom_metrics:
                    zscore = abs(e.raw_data.get("zscore", 0))
                    if zscore >= 2.0:
                        return False  # Memory metric IS anomalous
        return True

    def _check_no_packet_loss(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if no network METRICS are anomalous → NOT a network partition.

        Uses metric evidence only (root-service metrics) to avoid
        false negatives from cascade services' symptomatic network logs.
        """
        net_metrics = {"packet_loss_percent", "tcp_retransmissions"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in net_metrics:
                    zscore = abs(e.raw_data.get("zscore", 0))
                    if zscore >= 2.0:
                        return False  # Network metric IS anomalous
        return True

    def _check_no_cpu_saturation(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if no CPU METRICS are anomalous → NOT a CPU spike.

        Uses metric evidence only (root-service metrics).
        """
        cpu_metrics = {"cpu_percent", "thread_pool_active_pct"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in cpu_metrics:
                    zscore = abs(e.raw_data.get("zscore", 0))
                    if zscore >= 2.0:
                        return False  # CPU metric IS anomalous
        return True

    def _check_no_db_specific_errors(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        """TRUE if no DB METRICS are anomalous → NOT a database issue.

        Uses metric evidence only (root-service metrics).
        """
        db_metrics = {"db_query_duration_ms", "db_active_connections", "db_pool_wait_ms"}
        for e in items:
            if e.source == EvidenceSource.METRICS_AGENT:
                metric_name = e.raw_data.get("metric_name", "")
                if metric_name in db_metrics:
                    zscore = abs(e.raw_data.get("zscore", 0))
                    if zscore >= 2.0:
                        return False  # DB metric IS anomalous
        return True

    # ── DEPLOYMENT / CONFIG checkers (kept from original) ───────

    def _check_error_spike_after_deploy(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        return "spike" in all_text or "deploy" in all_text

    def _check_single_service_errors(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        svc_set = set()
        for e in items:
            if e.source == EvidenceSource.LOG_AGENT:
                svc = e.raw_data.get("service", "")
                if svc:
                    svc_set.add(svc)
        return len(svc_set) == 1

    def _check_high_error_rate(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        return any(
            e.raw_data.get("error_percentage", 0) > 10
            or "error" in e.description.lower()
            for e in items
            if e.source == EvidenceSource.LOG_AGENT
        )

    def _check_no_infrastructure_issues(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        dep_issues = any(
            e.source == EvidenceSource.DEPENDENCY_AGENT
            and e.strength == EvidenceStrength.STRONG
            for e in items
        )
        return not dep_issues

    def _check_config_keywords(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        return any(
            kw in all_text
            for kw in [
                "config", "configuration", "setting",
                "environment", "variable", "parameter",
            ]
        )

    def _check_sudden_error_onset(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        return "spike" in all_text or "sudden" in all_text

    def _check_specific_error_pattern(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        return any(
            len(e.raw_data.get("error_keywords_detected", [])) > 0
            for e in items
            if e.source == EvidenceSource.LOG_AGENT
        )

    def _check_limited_blast_radius(
        self,
        items: List[EvidenceItem],
        all_text: str,
        evidence: AggregatedEvidence,
    ) -> bool:
        for e in items:
            if e.source == EvidenceSource.DEPENDENCY_AGENT:
                raw = e.raw_data
                if raw.get("blast_radius", 0) > 5:
                    return False
        return True

    # ── Indicator dispatch table ────────────────────────────────

    _INDICATOR_CHECKERS = {
        # Database-specific
        "db_specific_errors": _check_db_specific_errors,
        "db_pool_keywords": _check_db_pool_keywords,
        "db_metric_anomaly": _check_db_metric_anomaly,
        # Memory-leak-specific
        "oom_keywords_strict": _check_oom_keywords_strict,
        "memory_heap_anomaly": _check_memory_heap_anomaly,
        "gc_overhead_keywords": _check_gc_overhead_keywords,
        # Network-specific
        "network_specific_errors": _check_network_specific_errors,
        "packet_loss_detected": _check_packet_loss_detected,
        "distributed_failure": _check_distributed_failure,
        # CPU-spike-specific
        "cpu_saturation_keywords": _check_cpu_saturation_keywords,
        "thread_pool_keywords": _check_thread_pool_keywords,
        "cpu_metric_anomaly": _check_cpu_metric_anomaly,
        # Negative (anti-) indicators
        "no_oom_errors": _check_no_oom_errors,
        "no_packet_loss": _check_no_packet_loss,
        "no_cpu_saturation": _check_no_cpu_saturation,
        "no_db_specific_errors": _check_no_db_specific_errors,
        # Deployment / config (kept from original)
        "error_spike_after_deploy": _check_error_spike_after_deploy,
        "single_service_errors": _check_single_service_errors,
        "high_error_rate": _check_high_error_rate,
        "no_infrastructure_issues": _check_no_infrastructure_issues,
        "config_keywords": _check_config_keywords,
        "sudden_error_onset": _check_sudden_error_onset,
        "specific_error_pattern": _check_specific_error_pattern,
        "limited_blast_radius": _check_limited_blast_radius,
    }
