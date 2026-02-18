"""
File: core/prevention_advisor.py
Purpose: Algorithm 7 – Generate long-term prevention recommendations.
Dependencies: Standard library only + schema.
Performance: <1ms per call, pure function.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from agents.incident_commander_agent.schema import (
    ActionPriority,
    CausalLink,
    Prevention,
    PreventionCategory,
    ValidationReport,
)


# (category, title, description, effort, priority)
_BASE_RECOMMENDATIONS: List[Tuple[PreventionCategory, str, str, str, ActionPriority]] = [
    (
        PreventionCategory.MONITORING,
        "Add anomaly detection alerts",
        "Implement automated anomaly detection on key SLI metrics "
        "(latency p99, error rate, saturation) to catch regressions earlier.",
        "1 sprint",
        ActionPriority.P1,
    ),
    (
        PreventionCategory.PROCESS,
        "Conduct blameless post-mortem",
        "Schedule a blameless post-mortem within 48 hours to identify "
        "process gaps and update runbooks with lessons learned.",
        "2 days",
        ActionPriority.P1,
    ),
]

# Root-cause-keyword → extra recommendations
_KEYWORD_RECS: Dict[str, List[Tuple[PreventionCategory, str, str, str, ActionPriority]]] = {
    "connection_pool": [
        (
            PreventionCategory.CONFIGURATION,
            "Tune connection pool limits",
            "Review and right-size connection pool max/min settings. "
            "Add pool exhaustion alerts at 80% threshold.",
            "1 day",
            ActionPriority.P1,
        ),
    ],
    "memory_leak": [
        (
            PreventionCategory.ARCHITECTURE,
            "Implement memory profiling in CI",
            "Add heap snapshot comparison tests in CI/CD pipeline "
            "to catch memory leaks before production deployment.",
            "2 sprints",
            ActionPriority.P2,
        ),
    ],
    "cascad": [
        (
            PreventionCategory.ARCHITECTURE,
            "Add circuit breakers between services",
            "Implement circuit breaker pattern (e.g., resilience4j) "
            "on all inter-service calls to prevent cascading failures.",
            "2 sprints",
            ActionPriority.P0,
        ),
        (
            PreventionCategory.ARCHITECTURE,
            "Implement bulkhead isolation",
            "Isolate critical paths with separate thread/connection pools "
            "so failures in one path don't exhaust shared resources.",
            "3 sprints",
            ActionPriority.P1,
        ),
    ],
    "network": [
        (
            PreventionCategory.ARCHITECTURE,
            "Implement retry with exponential backoff",
            "Add idempotent retry logic with jitter on all network calls "
            "to handle transient failures gracefully.",
            "1 sprint",
            ActionPriority.P1,
        ),
    ],
    "deploy": [
        (
            PreventionCategory.PROCESS,
            "Implement canary deployments",
            "Roll out changes to a canary cohort first (5-10%) with "
            "automated rollback on SLO violation.",
            "2 sprints",
            ActionPriority.P1,
        ),
    ],
    "config": [
        (
            PreventionCategory.CONFIGURATION,
            "Add config validation in CI",
            "Implement schema validation for all configuration changes "
            "with dry-run verification before apply.",
            "1 sprint",
            ActionPriority.P1,
        ),
    ],
    "database": [
        (
            PreventionCategory.MONITORING,
            "Add database health dashboard",
            "Create comprehensive database monitoring including "
            "connection count, query latency p99, replication lag, "
            "and deadlock frequency.",
            "1 sprint",
            ActionPriority.P1,
        ),
    ],
}


def _match_keywords(root_cause: str) -> Set[str]:
    """Find keywords matching the root cause."""
    normalized = root_cause.lower().replace("-", "_").replace(" ", "_")
    matched: Set[str] = set()
    for keyword in _KEYWORD_RECS:
        if keyword in normalized:
            matched.add(keyword)
    return matched


def _extract_validation_recs(
    validation_report: ValidationReport | None,
) -> List[Tuple[PreventionCategory, str, str, str, ActionPriority]]:
    """Add recommendations based on validation report findings."""
    if validation_report is None:
        return []

    recs: List[Tuple[PreventionCategory, str, str, str, ActionPriority]] = []

    if validation_report.hallucinations:
        recs.append((
            PreventionCategory.MONITORING,
            "Improve observability coverage",
            "Add structured logging and distributed tracing to services "
            "with insufficient evidence, reducing hallucination risk "
            "in automated analysis.",
            "2 sprints",
            ActionPriority.P2,
        ))

    if validation_report.accuracy_score < 0.7:
        recs.append((
            PreventionCategory.PROCESS,
            "Review root cause analysis tooling",
            f"Accuracy score ({validation_report.accuracy_score:.0%}) is below "
            "threshold. Evaluate and improve automated root cause "
            "analysis pipelines.",
            "1 quarter",
            ActionPriority.P2,
        ))

    return recs


def generate_prevention_recommendations(
    root_cause: str,
    causal_chain: List[CausalLink] | None = None,
    validation_report: ValidationReport | None = None,
) -> List[Prevention]:
    """Generate prevention recommendations.

    Args:
        root_cause: Root cause description.
        causal_chain: Causal chain (used for keyword scanning).
        validation_report: Validation report for quality insights.

    Returns:
        List of Prevention recommendations.
    """
    chain = causal_chain or []

    # Gather keyword matches from root_cause + causal chain
    keywords = _match_keywords(root_cause)
    for link in chain:
        keywords |= _match_keywords(f"{link.cause} {link.effect}")

    # Build recommendations list: base + keyword-specific + validation
    raw: List[Tuple[PreventionCategory, str, str, str, ActionPriority]] = list(
        _BASE_RECOMMENDATIONS
    )
    for kw in sorted(keywords):
        raw.extend(_KEYWORD_RECS.get(kw, []))
    raw.extend(_extract_validation_recs(validation_report))

    # Deduplicate by title
    seen: Set[str] = set()
    result: List[Prevention] = []
    for cat, title, desc, effort, prio in raw:
        if title in seen:
            continue
        seen.add(title)
        result.append(
            Prevention(
                category=cat,
                title=title,
                description=desc,
                effort_estimate=effort,
                priority=prio,
            )
        )

    return result
