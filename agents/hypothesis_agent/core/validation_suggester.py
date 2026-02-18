"""
File: core/validation_suggester.py
Purpose: Algorithm 7 — Validation Test Suggestion O(n).
Dependencies: Schema models only.
Performance: <5ms, O(n) where n=hypotheses.

Generates actionable validation tests for each hypothesis
to help operators verify or invalidate theories.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    Hypothesis,
    IncidentCategory,
    PatternMatch,
    ValidationTest,
)
from agents.hypothesis_agent.telemetry import get_logger

logger = get_logger("hypothesis_agent.validation_suggester")


# ═══════════════════════════════════════════════════════════════
#  CATEGORY-BASED TEST TEMPLATES
# ═══════════════════════════════════════════════════════════════

_CATEGORY_TESTS: Dict[IncidentCategory, List[ValidationTest]] = {
    IncidentCategory.DATABASE: [
        ValidationTest(
            test_name="check_connection_pool",
            description="Check database connection pool utilization",
            expected_outcome="Pool usage near 100% capacity",
            priority=1,
        ),
        ValidationTest(
            test_name="check_slow_queries",
            description="Review slow query log for recent entries",
            expected_outcome="Unusually slow or locked queries",
            priority=2,
        ),
        ValidationTest(
            test_name="check_db_replication_lag",
            description="Verify replication lag if applicable",
            expected_outcome="Increased lag correlating with incident",
            priority=3,
        ),
    ],
    IncidentCategory.APPLICATION: [
        ValidationTest(
            test_name="check_heap_memory",
            description="Check heap memory usage trends",
            expected_outcome="Monotonically increasing memory",
            priority=1,
        ),
        ValidationTest(
            test_name="check_gc_activity",
            description="Review garbage collection frequency/duration",
            expected_outcome="Excessive GC pauses or full GC events",
            priority=2,
        ),
        ValidationTest(
            test_name="check_thread_dumps",
            description="Capture thread dumps for deadlock analysis",
            expected_outcome="Blocked or waiting threads",
            priority=3,
        ),
    ],
    IncidentCategory.NETWORK: [
        ValidationTest(
            test_name="check_connectivity",
            description="Run connectivity checks between AZs",
            expected_outcome="Packet loss or unreachable hosts",
            priority=1,
        ),
        ValidationTest(
            test_name="check_dns",
            description="Verify DNS resolution across services",
            expected_outcome="Stale or incorrect DNS records",
            priority=2,
        ),
        ValidationTest(
            test_name="check_network_config",
            description="Review recent network configuration changes",
            expected_outcome="Recent change correlating with incident",
            priority=3,
        ),
    ],
    IncidentCategory.DEPLOYMENT: [
        ValidationTest(
            test_name="check_deploy_timeline",
            description="Compare deployment time vs error onset",
            expected_outcome="Errors started after deployment",
            priority=1,
        ),
        ValidationTest(
            test_name="check_rollback_option",
            description="Verify rollback is available",
            expected_outcome="Previous version available for rollback",
            priority=2,
        ),
        ValidationTest(
            test_name="check_canary_metrics",
            description="Review canary deployment metrics",
            expected_outcome="Canary showing elevated errors",
            priority=3,
        ),
    ],
    IncidentCategory.CONFIGURATION: [
        ValidationTest(
            test_name="check_config_diff",
            description="Diff current config against last known good",
            expected_outcome="Changed values identified",
            priority=1,
        ),
        ValidationTest(
            test_name="check_env_vars",
            description="Verify environment variable values",
            expected_outcome="Misconfigured or missing variables",
            priority=2,
        ),
        ValidationTest(
            test_name="check_feature_flags",
            description="Review feature flag changes",
            expected_outcome="Recently toggled flags",
            priority=3,
        ),
    ],
    IncidentCategory.INFRASTRUCTURE: [
        ValidationTest(
            test_name="check_resource_utilization",
            description="Check CPU, memory, disk utilization",
            expected_outcome="Resource exhaustion on affected nodes",
            priority=1,
        ),
        ValidationTest(
            test_name="check_cloud_status",
            description="Verify cloud provider status page",
            expected_outcome="Reported incidents in region/AZ",
            priority=2,
        ),
    ],
    IncidentCategory.SECURITY: [
        ValidationTest(
            test_name="check_auth_logs",
            description="Review authentication/authorization logs",
            expected_outcome="Unusual access patterns or failures",
            priority=1,
        ),
        ValidationTest(
            test_name="check_cert_validity",
            description="Verify TLS certificate validity",
            expected_outcome="Expired or misconfigured certificates",
            priority=2,
        ),
    ],
}


class ValidationSuggester:
    """Generates validation tests for hypotheses.

    For each hypothesis, suggests tests from:
    1. Category-based templates.
    2. Pattern-specific tests.
    3. Evidence-derived checks.

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[HypothesisAgentConfig] = None
    ) -> None:
        self._config = config or HypothesisAgentConfig()

    def suggest(
        self,
        hypotheses: List[Hypothesis],
        pattern_matches: List[PatternMatch],
        correlation_id: str = "",
    ) -> List[Hypothesis]:
        """Add validation tests to each hypothesis.

        Args:
            hypotheses: Hypotheses to enrich.
            pattern_matches: Matched patterns for extra tests.
            correlation_id: Request correlation ID.

        Returns:
            Hypotheses with validation_tests populated.
        """
        start = time.perf_counter()
        enriched: List[Hypothesis] = []

        # Build pattern test lookup
        pattern_tests = self._collect_pattern_tests(
            pattern_matches
        )

        for h in hypotheses:
            tests = self._generate_tests(h, pattern_tests)
            enriched.append(
                h.model_copy(update={"validation_tests": tests})
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Validation suggestion complete — "
            f"{sum(len(h.validation_tests) for h in enriched)} "
            f"tests for {len(enriched)} hypotheses, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "validation_suggestion",
            },
        )

        return enriched

    def _generate_tests(
        self,
        hypothesis: Hypothesis,
        pattern_tests: Dict[str, List[str]],
    ) -> List[ValidationTest]:
        """Generate tests for a single hypothesis.

        Args:
            hypothesis: The hypothesis.
            pattern_tests: Pattern-specific test descriptions.

        Returns:
            List of validation tests.
        """
        tests: List[ValidationTest] = []
        seen_names: set = set()

        # 1. Category-based templates
        category_tests = _CATEGORY_TESTS.get(
            hypothesis.category, []
        )
        for t in category_tests:
            if t.test_name not in seen_names:
                tests.append(t)
                seen_names.add(t.test_name)

        # 2. Pattern-specific tests
        if hypothesis.pattern_match:
            pn = hypothesis.pattern_match.pattern_name.value
            for desc in pattern_tests.get(pn, []):
                name = desc.lower().replace(" ", "_")[:40]
                if name not in seen_names:
                    tests.append(ValidationTest(
                        test_name=name,
                        description=desc,
                        expected_outcome="Finding correlating with hypothesis",
                        priority=2,
                    ))
                    seen_names.add(name)

        # 3. Generic evidence verification test
        if hypothesis.evidence_supporting:
            tests.append(ValidationTest(
                test_name="verify_supporting_evidence",
                description=(
                    "Independently verify the supporting evidence "
                    "for this hypothesis"
                ),
                expected_outcome="Evidence confirmed by manual inspection",
                priority=4,
            ))

        return tests

    @staticmethod
    def _collect_pattern_tests(
        pattern_matches: List[PatternMatch],
    ) -> Dict[str, List[str]]:
        """Collect test descriptions from the pattern library.

        Args:
            pattern_matches: Matched patterns.

        Returns:
            Dict mapping pattern name to test descriptions.
        """
        from agents.hypothesis_agent.core.pattern_matcher import (
            PATTERN_LIBRARY,
        )

        tests: Dict[str, List[str]] = {}
        matched_names = {
            pm.pattern_name for pm in pattern_matches
        }

        for pattern in PATTERN_LIBRARY:
            if pattern.pattern_name in matched_names:
                tests[pattern.pattern_name.value] = (
                    pattern.validation_tests
                )

        return tests
