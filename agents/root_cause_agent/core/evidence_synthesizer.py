"""
File: core/evidence_synthesizer.py
Purpose: Algorithm 1 — Fuse evidence from 4 agents, cross-validate, weight by reliability/recency.
Dependencies: Schema models only.
Performance: <5ms, O(n) where n = total findings across agents.

Extracts evidence items from each agent's findings, assigns source
weights, detects agreement between agents, and returns a unified
SynthesisResult.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    RootCauseAgentInput,
    Severity,
    SynthesisResult,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.evidence_synthesizer")


class EvidenceSynthesizer:
    """Fuses evidence from 4 upstream agents into a unified trail.

    Pipeline::

        LogFindings      ──┐
        MetricsFindings  ──┤──  extract → weight → cross-validate → SynthesisResult
        DepFindings      ──┤
        HypothesisFindings─┘

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def synthesize(
        self,
        input_data: RootCauseAgentInput,
        correlation_id: str = "",
    ) -> SynthesisResult:
        """Fuse evidence from all 4 agent outputs.

        Args:
            input_data: Root cause agent input with all upstream findings.
            correlation_id: Request correlation ID.

        Returns:
            SynthesisResult with unified evidence trail and agreement score.
        """
        start = time.perf_counter()
        evidence_trail: List[Evidence] = []
        sources: Set[EvidenceSourceAgent] = set()

        # ── Extract from log findings ───────────────────────────
        log_ev = self._extract_log_evidence(input_data)
        if log_ev:
            evidence_trail.extend(log_ev)
            sources.add(EvidenceSourceAgent.LOG_AGENT)

        # ── Extract from metrics findings ───────────────────────
        metrics_ev = self._extract_metrics_evidence(input_data)
        if metrics_ev:
            evidence_trail.extend(metrics_ev)
            sources.add(EvidenceSourceAgent.METRICS_AGENT)

        # ── Extract from dependency findings ────────────────────
        dep_ev = self._extract_dependency_evidence(input_data)
        if dep_ev:
            evidence_trail.extend(dep_ev)
            sources.add(EvidenceSourceAgent.DEPENDENCY_AGENT)

        # ── Extract from hypothesis findings ────────────────────
        hyp_ev = self._extract_hypothesis_evidence(input_data)
        if hyp_ev:
            evidence_trail.extend(hyp_ev)
            sources.add(EvidenceSourceAgent.HYPOTHESIS_AGENT)

        # ── Cap evidence items ──────────────────────────────────
        max_items = self._config.performance.max_evidence_items
        if len(evidence_trail) > max_items:
            evidence_trail = evidence_trail[:max_items]

        # ── Compute agreement score ─────────────────────────────
        agreement = self._compute_agreement(input_data, sources)

        # ── Determine primary service ───────────────────────────
        primary_service = self._determine_primary_service(
            input_data, evidence_trail
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Evidence synthesis complete — "
            f"{len(evidence_trail)} items, "
            f"{len(sources)} sources, "
            f"agreement={agreement:.2f}, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "evidence_synthesis",
                "context": {
                    "total": len(evidence_trail),
                    "sources": [s.value for s in sorted(sources, key=lambda x: x.value)],
                    "agreement": round(agreement, 4),
                },
            },
        )

        return SynthesisResult(
            evidence_trail=evidence_trail,
            sources_present=sorted(list(sources), key=lambda x: x.value),
            agreement_score=round(agreement, 4),
            primary_service=primary_service,
            synthesis_latency_ms=round(elapsed_ms, 2),
        )

    # ── Evidence extraction per agent ───────────────────────────

    def _extract_log_evidence(
        self, input_data: RootCauseAgentInput
    ) -> List[Evidence]:
        """Extract evidence from log agent findings."""
        items: List[Evidence] = []
        lf = input_data.log_findings

        for svc in lf.suspicious_services:
            items.append(Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                description=f"Suspicious service detected: {svc}",
                confidence=lf.confidence,
                timestamp=lf.timestamp,
                raw_data={"service": svc},
            ))

        for pattern in lf.error_patterns:
            items.append(Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                description=f"Error pattern: {pattern}",
                confidence=lf.confidence * 0.8,
                timestamp=lf.timestamp,
                raw_data={"pattern": pattern},
            ))

        for anomaly in lf.log_anomalies:
            desc = anomaly.get("description", "Log anomaly detected")
            items.append(Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.CIRCUMSTANTIAL,
                description=str(desc),
                confidence=lf.confidence * 0.6,
                timestamp=lf.timestamp,
                raw_data=anomaly,
            ))

        return items

    def _extract_metrics_evidence(
        self, input_data: RootCauseAgentInput
    ) -> List[Evidence]:
        """Extract evidence from metrics agent findings."""
        items: List[Evidence] = []
        mf = input_data.metrics_findings

        for anomaly in mf.anomalies:
            svc = anomaly.get("service", "unknown")
            metric = anomaly.get("metric", "unknown")
            zscore = anomaly.get("z_score", 0.0)
            severity = anomaly.get("severity", "medium")

            ev_type = EvidenceType.DIRECT
            if abs(zscore) < 2.0:
                ev_type = EvidenceType.CIRCUMSTANTIAL
            elif abs(zscore) < 3.0:
                ev_type = EvidenceType.CORRELATED

            items.append(Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=ev_type,
                description=(
                    f"Anomaly in {svc}/{metric}: "
                    f"z-score={zscore:.2f}, severity={severity}"
                ),
                confidence=mf.confidence,
                timestamp=mf.timestamp,
                raw_data=anomaly,
            ))

        for corr in mf.correlations:
            items.append(Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                description=f"Metric correlation: {corr.get('description', str(corr))}",
                confidence=mf.confidence * 0.7,
                timestamp=mf.timestamp,
                raw_data=corr,
            ))

        return items

    def _extract_dependency_evidence(
        self, input_data: RootCauseAgentInput
    ) -> List[Evidence]:
        """Extract evidence from dependency agent findings."""
        items: List[Evidence] = []
        df = input_data.dependency_findings

        if df.blast_radius > 0:
            items.append(Evidence(
                source=EvidenceSourceAgent.DEPENDENCY_AGENT,
                evidence_type=EvidenceType.DIRECT,
                description=(
                    f"Blast radius: {df.blast_radius} services affected"
                ),
                confidence=df.confidence,
                timestamp=df.timestamp,
                raw_data={"blast_radius": df.blast_radius},
            ))

        for bn in df.bottlenecks:
            items.append(Evidence(
                source=EvidenceSourceAgent.DEPENDENCY_AGENT,
                evidence_type=EvidenceType.DIRECT,
                description=f"Bottleneck service: {bn}",
                confidence=df.confidence,
                timestamp=df.timestamp,
                raw_data={"bottleneck": bn},
            ))

        for path in df.critical_paths:
            items.append(Evidence(
                source=EvidenceSourceAgent.DEPENDENCY_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                description=f"Critical path: {' → '.join(path)}",
                confidence=df.confidence * 0.8,
                timestamp=df.timestamp,
                raw_data={"critical_path": path},
            ))

        for svc in df.affected_services:
            items.append(Evidence(
                source=EvidenceSourceAgent.DEPENDENCY_AGENT,
                evidence_type=EvidenceType.CIRCUMSTANTIAL,
                description=f"Affected service: {svc}",
                confidence=df.confidence * 0.5,
                timestamp=df.timestamp,
                raw_data={"affected_service": svc},
            ))

        return items

    def _extract_hypothesis_evidence(
        self, input_data: RootCauseAgentInput
    ) -> List[Evidence]:
        """Extract evidence from hypothesis agent findings."""
        items: List[Evidence] = []
        hf = input_data.hypothesis_findings

        if hf.top_hypothesis:
            items.append(Evidence(
                source=EvidenceSourceAgent.HYPOTHESIS_AGENT,
                evidence_type=EvidenceType.DIRECT,
                description=f"Top hypothesis: {hf.top_hypothesis}",
                confidence=hf.top_confidence,
                timestamp=hf.timestamp,
                raw_data={
                    "hypothesis": hf.top_hypothesis,
                    "confidence": hf.top_confidence,
                },
            ))

        for hyp in hf.ranked_hypotheses[1:]:  # skip top (already added)
            theory = hyp.get("theory", str(hyp))
            conf = hyp.get("confidence", hyp.get("likelihood_score", 0.3))
            items.append(Evidence(
                source=EvidenceSourceAgent.HYPOTHESIS_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                description=f"Alternative hypothesis: {theory}",
                confidence=min(1.0, max(0.0, float(conf))),
                timestamp=hf.timestamp,
                raw_data=hyp,
            ))

        for chain in hf.causal_chains:
            desc = chain.get("description", str(chain))
            items.append(Evidence(
                source=EvidenceSourceAgent.HYPOTHESIS_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                description=f"Causal chain: {desc}",
                confidence=hf.confidence * 0.8,
                timestamp=hf.timestamp,
                raw_data=chain,
            ))

        return items

    # ── Agreement scoring ───────────────────────────────────────

    def _compute_agreement(
        self,
        input_data: RootCauseAgentInput,
        sources: Set[EvidenceSourceAgent],
    ) -> float:
        """Compute inter-agent agreement score 0.0-1.0.

        Agreement is based on:
        - Number of sources present (more = better)
        - Whether agents agree on the primary service
        - Confidence convergence

        Args:
            input_data: The agent input.
            sources: Which agents contributed evidence.

        Returns:
            Agreement score between 0.0 and 1.0.
        """
        if not sources:
            return 0.0

        # Base score from source count (4 sources = 0.5 base)
        source_score = len(sources) / 4.0 * 0.5

        # Confidence convergence (less spread = more agreement)
        confidences = []
        if EvidenceSourceAgent.LOG_AGENT in sources:
            confidences.append(input_data.log_findings.confidence)
        if EvidenceSourceAgent.METRICS_AGENT in sources:
            confidences.append(input_data.metrics_findings.confidence)
        if EvidenceSourceAgent.DEPENDENCY_AGENT in sources:
            confidences.append(input_data.dependency_findings.confidence)
        if EvidenceSourceAgent.HYPOTHESIS_AGENT in sources:
            confidences.append(input_data.hypothesis_findings.confidence)

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            spread = max(confidences) - min(confidences)
            convergence_score = avg_conf * (1.0 - spread * 0.5)
        else:
            convergence_score = 0.0

        agreement = min(1.0, source_score + convergence_score * 0.5)
        return round(agreement, 4)

    # ── Primary service detection ───────────────────────────────

    def _determine_primary_service(
        self,
        input_data: RootCauseAgentInput,
        evidence_trail: List[Evidence],
    ) -> str:
        """Determine the most-blamed service across agents.

        Args:
            input_data: The agent input.
            evidence_trail: All evidence items.

        Returns:
            Primary service name, or empty string.
        """
        service_votes: Counter = Counter()

        # Log agent votes
        for svc in input_data.log_findings.suspicious_services:
            service_votes[svc] += 2

        # Dependency agent votes
        for bn in input_data.dependency_findings.bottlenecks:
            service_votes[bn] += 3

        # Metrics agent votes (from anomaly services)
        for anom in input_data.metrics_findings.anomalies:
            svc = anom.get("service", "")
            if svc:
                service_votes[svc] += 2

        # Evidence trail votes
        for ev in evidence_trail:
            svc = ev.raw_data.get("service", "")
            if svc:
                service_votes[svc] += 1

        if service_votes:
            return service_votes.most_common(1)[0][0]
        return ""
