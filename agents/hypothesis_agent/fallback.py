"""
File: fallback.py
Purpose: Rule-based fallback hypothesis generation.
Dependencies: Schema models only.
Performance: <10ms, O(e) where e=evidence items.

Generates deterministic hypotheses when LLM is disabled or
circuit breaker is open. Uses pattern matches and evidence
to construct reasonable hypotheses without LLM calls.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    CausalChain,
    CausalChainLink,
    CausalRelationship,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    Hypothesis,
    HypothesisStatus,
    IncidentCategory,
    PatternMatch,
    PatternName,
    Severity,
)
from agents.hypothesis_agent.telemetry import get_logger

logger = get_logger("hypothesis_agent.fallback")


# ═══════════════════════════════════════════════════════════════
#  PATTERN → HYPOTHESIS TEMPLATES
# ═══════════════════════════════════════════════════════════════

_PATTERN_TEMPLATES: Dict[PatternName, Dict] = {
    PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION: {
        "theory": (
            "Database connection pool exhaustion caused cascading "
            "timeouts across dependent services"
        ),
        "category": IncidentCategory.DATABASE,
        "severity": Severity.CRITICAL,
        "mttr": 45.0,
    },
    PatternName.MEMORY_LEAK: {
        "theory": (
            "Memory leak in application service causing progressive "
            "performance degradation and eventual OOM failures"
        ),
        "category": IncidentCategory.APPLICATION,
        "severity": Severity.HIGH,
        "mttr": 60.0,
    },
    PatternName.NETWORK_PARTITION: {
        "theory": (
            "Network partition or connectivity loss causing "
            "service communication failures"
        ),
        "category": IncidentCategory.NETWORK,
        "severity": Severity.HIGH,
        "mttr": 30.0,
    },
    PatternName.CPU_SPIKE: {
        "theory": (
            "CPU saturation in application service causing thread "
            "pool exhaustion and cascading request failures"
        ),
        "category": IncidentCategory.INFRASTRUCTURE,
        "severity": Severity.HIGH,
        "mttr": 20.0,
    },
    PatternName.DEPLOYMENT_ISSUE: {
        "theory": (
            "Recent deployment introduced a breaking change "
            "causing service failures"
        ),
        "category": IncidentCategory.DEPLOYMENT,
        "severity": Severity.MEDIUM,
        "mttr": 20.0,
    },
    PatternName.CONFIGURATION_ERROR: {
        "theory": (
            "Configuration error causing unexpected behavior "
            "or service failures"
        ),
        "category": IncidentCategory.CONFIGURATION,
        "severity": Severity.MEDIUM,
        "mttr": 15.0,
    },
}


class FallbackGenerator:
    """Rule-based hypothesis generator — deterministic fallback.

    Generates hypotheses from:
    1. Pattern matches (highest priority).
    2. Evidence-based inference.
    3. Generic catch-all hypothesis.

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[HypothesisAgentConfig] = None
    ) -> None:
        self._config = config or HypothesisAgentConfig()

    def generate(
        self,
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
        correlation_id: str = "",
    ) -> List[Hypothesis]:
        """Generate hypotheses using deterministic rules.

        Args:
            evidence: Aggregated evidence.
            pattern_matches: Known pattern matches.
            correlation_id: Request correlation ID.

        Returns:
            List of hypotheses (3-5).
        """
        start = time.perf_counter()
        hypotheses: List[Hypothesis] = []

        # 1. Generate from pattern matches
        for pm in pattern_matches:
            template = _PATTERN_TEMPLATES.get(pm.pattern_name)
            if template:
                # Collect supporting evidence descriptions
                supporting = self._find_supporting(
                    evidence, pm.pattern_name
                )
                hypotheses.append(Hypothesis(
                    theory=template["theory"],
                    category=template["category"],
                    severity=template["severity"],
                    likelihood_score=round(pm.match_score * 0.8, 4),
                    evidence_supporting=supporting,
                    evidence_contradicting=[],
                    pattern_match=pm,
                    estimated_mttr_minutes=template["mttr"],
                    status=HypothesisStatus.ACTIVE,
                    reasoning=(
                        f"Pattern match: {pm.pattern_name.value} "
                        f"(score={pm.match_score:.2f})"
                    ),
                ))

        # 2. Generate from strong evidence if needed
        if len(hypotheses) < self._config.limits.min_hypotheses:
            evidence_hypotheses = self._from_evidence(evidence)
            for h in evidence_hypotheses:
                if len(hypotheses) >= self._config.limits.max_hypotheses:
                    break
                # Avoid duplicates by category
                if not any(
                    eh.category == h.category for eh in hypotheses
                ):
                    hypotheses.append(h)

        # 3. Add catch-all if still below minimum
        if len(hypotheses) < self._config.limits.min_hypotheses:
            hypotheses.append(Hypothesis(
                theory=(
                    "Transient infrastructure issue causing "
                    "intermittent service degradation"
                ),
                category=IncidentCategory.INFRASTRUCTURE,
                severity=Severity.MEDIUM,
                likelihood_score=0.15,
                evidence_supporting=[],
                evidence_contradicting=[],
                estimated_mttr_minutes=30.0,
                status=HypothesisStatus.ACTIVE,
                reasoning="Catch-all hypothesis for unmatched evidence",
            ))

        # Ensure minimum and cap
        while len(hypotheses) < self._config.limits.min_hypotheses:
            hypotheses.append(Hypothesis(
                theory=(
                    "Unknown root cause — further investigation needed"
                ),
                category=IncidentCategory.UNKNOWN,
                severity=Severity.LOW,
                likelihood_score=0.1,
                status=HypothesisStatus.ACTIVE,
                reasoning="Minimum hypothesis requirement",
            ))

        hypotheses = hypotheses[
            : self._config.limits.max_hypotheses
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Fallback generation complete — "
            f"{len(hypotheses)} hypotheses in {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "fallback",
                "context": {
                    "hypothesis_count": len(hypotheses),
                    "from_patterns": sum(
                        1
                        for h in hypotheses
                        if h.pattern_match is not None
                    ),
                },
            },
        )

        return hypotheses

    def _from_evidence(
        self, evidence: AggregatedEvidence
    ) -> List[Hypothesis]:
        """Generate hypotheses from evidence signals."""
        hypotheses: List[Hypothesis] = []

        # Check for database signals
        has_db = any(
            "database" in e.description.lower()
            for e in evidence.evidence_items
        )
        if has_db:
            hypotheses.append(Hypothesis(
                theory=(
                    "Database-related issue causing service failures"
                ),
                category=IncidentCategory.DATABASE,
                severity=Severity.HIGH,
                likelihood_score=0.4,
                evidence_supporting=[
                    e.description
                    for e in evidence.evidence_items
                    if "database" in e.description.lower()
                ],
                status=HypothesisStatus.ACTIVE,
                reasoning="Database-related evidence detected",
            ))

        # Check for cascading signals
        has_cascade = any(
            "cascading" in e.description.lower()
            for e in evidence.evidence_items
        )
        if has_cascade:
            hypotheses.append(Hypothesis(
                theory=(
                    "Cascading failure propagation from a "
                    "single service failure"
                ),
                category=IncidentCategory.INFRASTRUCTURE,
                severity=Severity.CRITICAL,
                likelihood_score=0.45,
                evidence_supporting=[
                    e.description
                    for e in evidence.evidence_items
                    if "cascading" in e.description.lower()
                    or "blast" in e.description.lower()
                ],
                status=HypothesisStatus.ACTIVE,
                reasoning="Cascading failure evidence detected",
            ))

        # Check for resource signals
        has_resource = any(
            "saturation" in e.description.lower()
            or "resource" in e.description.lower()
            or "memory" in e.description.lower()
            for e in evidence.evidence_items
        )
        if has_resource:
            hypotheses.append(Hypothesis(
                theory=(
                    "Resource exhaustion causing service degradation"
                ),
                category=IncidentCategory.APPLICATION,
                severity=Severity.HIGH,
                likelihood_score=0.35,
                evidence_supporting=[
                    e.description
                    for e in evidence.evidence_items
                    if any(
                        kw in e.description.lower()
                        for kw in [
                            "saturation", "resource", "memory",
                        ]
                    )
                ],
                status=HypothesisStatus.ACTIVE,
                reasoning="Resource-related evidence detected",
            ))

        # Check for network signals
        has_network = any(
            "network" in e.description.lower()
            or "connection" in e.description.lower()
            for e in evidence.evidence_items
        )
        if has_network:
            hypotheses.append(Hypothesis(
                theory=(
                    "Network connectivity issue affecting "
                    "service communication"
                ),
                category=IncidentCategory.NETWORK,
                severity=Severity.MEDIUM,
                likelihood_score=0.3,
                evidence_supporting=[
                    e.description
                    for e in evidence.evidence_items
                    if "network" in e.description.lower()
                    or "connection" in e.description.lower()
                ],
                status=HypothesisStatus.ACTIVE,
                reasoning="Network-related evidence detected",
            ))

        return hypotheses

    def _find_supporting(
        self,
        evidence: AggregatedEvidence,
        pattern_name: PatternName,
    ) -> List[str]:
        """Find evidence descriptions supporting a pattern."""
        keywords: Dict[PatternName, List[str]] = {
            PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION: [
                "database", "db_query", "db_active", "db_pool",
                "hikaricp", "deadlock", "sql",
                "connection pool exhausted", "max_connections",
            ],
            PatternName.MEMORY_LEAK: [
                "memory", "heap", "oom", "leak",
                "outofmemoryerror", "gc overhead", "gc_overhead",
            ],
            PatternName.NETWORK_PARTITION: [
                "packet loss", "packet_loss", "retransmission",
                "tcp_retransmission", "no route", "unreachable",
                "network partition", "dns",
            ],
            PatternName.CPU_SPIKE: [
                "cpu", "thread pool", "thread_pool",
                "thread starvation", "worker", "saturat",
                "context switch", "queue depth",
            ],
            PatternName.DEPLOYMENT_ISSUE: [
                "deploy", "release", "version", "rollback",
            ],
            PatternName.CONFIGURATION_ERROR: [
                "config", "setting", "environment", "parameter",
            ],
        }

        kws = keywords.get(pattern_name, [])
        return [
            e.description
            for e in evidence.evidence_items
            if any(kw in e.description.lower() for kw in kws)
        ][:5]
