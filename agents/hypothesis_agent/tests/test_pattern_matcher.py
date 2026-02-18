"""
Tests for core/pattern_matcher.py — Algorithm 2: Pattern Matching O(p*e).

Covers:
  - Pattern library structure (5 patterns)
  - Individual indicator checkers
  - Full pattern matching with rich evidence
  - Score sorting
  - Thresholds and filtering
"""

from __future__ import annotations

import pytest

from agents.hypothesis_agent.config import (
    HypothesisAgentConfig,
    PatternThresholds,
)
from agents.hypothesis_agent.core.pattern_matcher import (
    PATTERN_LIBRARY,
    PatternMatcher,
)
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    IncidentCategory,
    PatternName,
    Severity,
)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _evidence(*items: EvidenceItem) -> AggregatedEvidence:
    """Build AggregatedEvidence from evidence items."""
    sources = list(set(e.source for e in items))
    return AggregatedEvidence(
        evidence_items=list(items),
        total_evidence_count=len(items),
        strong_evidence_count=sum(
            1 for e in items
            if e.strength == EvidenceStrength.STRONG
        ),
        sources_represented=sources,
    )


def _item(
    desc: str,
    source: EvidenceSource = EvidenceSource.LOG_AGENT,
    strength: EvidenceStrength = EvidenceStrength.MODERATE,
    severity: Severity = Severity.MEDIUM,
    raw_data: dict | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        source=source,
        description=desc,
        strength=strength,
        severity=severity,
        raw_data=raw_data or {},
    )


@pytest.fixture
def matcher() -> PatternMatcher:
    return PatternMatcher()


# ═══════════════════════════════════════════════════════════════
#  TESTS: Pattern Library
# ═══════════════════════════════════════════════════════════════


class TestPatternLibrary:
    """Test pattern library structure."""

    def test_library_has_5_patterns(self):
        assert len(PATTERN_LIBRARY) == 6

    def test_all_pattern_names_represented(self):
        names = {p.pattern_name for p in PATTERN_LIBRARY}
        expected = {
            PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
            PatternName.MEMORY_LEAK,
            PatternName.NETWORK_PARTITION,
            PatternName.CPU_SPIKE,
            PatternName.DEPLOYMENT_ISSUE,
            PatternName.CONFIGURATION_ERROR,
        }
        assert names == expected

    def test_each_pattern_has_indicators(self):
        for p in PATTERN_LIBRARY:
            assert len(p.indicators) >= 3, (
                f"{p.pattern_name} has too few indicators"
            )

    def test_each_pattern_has_category(self):
        for p in PATTERN_LIBRARY:
            assert p.category != IncidentCategory.UNKNOWN

    def test_indicator_weights_sum_roughly_one(self):
        for p in PATTERN_LIBRARY:
            total = sum(ind.weight for ind in p.indicators)
            assert 0.8 <= total <= 1.2, (
                f"{p.pattern_name} weights sum to {total}"
            )


# ═══════════════════════════════════════════════════════════════
#  TESTS: Database Pattern Matching
# ═══════════════════════════════════════════════════════════════


class TestDatabasePatternMatching:
    """Test matching the database_connection_pool_exhaustion pattern."""

    def test_strong_db_match(self, matcher: PatternMatcher):
        evidence = _evidence(
            _item("Database connection pool exhausted — query timeout errors"),
            _item("Connection pool wait time elevated — HikariCP pool full"),
            _item(
                "DB query duration anomaly detected",
                source=EvidenceSource.METRICS_AGENT,
                raw_data={"metric_name": "db_query_duration_ms"},
            ),
        )
        matches = matcher.match(evidence)
        db_matches = [
            m for m in matches
            if m.pattern_name
            == PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION
        ]
        assert len(db_matches) == 1
        assert db_matches[0].match_score >= 0.5
        assert db_matches[0].category == IncidentCategory.DATABASE

    def test_no_db_match_without_indicators(
        self, matcher: PatternMatcher
    ):
        evidence = _evidence(
            _item("Everything looks normal"),
        )
        matches = matcher.match(evidence)
        db_matches = [
            m for m in matches
            if m.pattern_name
            == PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION
        ]
        assert len(db_matches) == 0


# ═══════════════════════════════════════════════════════════════
#  TESTS: Memory Leak Pattern
# ═══════════════════════════════════════════════════════════════


class TestMemoryLeakPattern:
    """Test matching the memory_leak pattern."""

    def test_memory_leak_match(self, matcher: PatternMatcher):
        evidence = _evidence(
            _item("Heap memory increasing — GC overhead limit exceeded",
                  raw_data={"metric_name": "heap_used_mb"}),
            _item("OutOfMemoryError: Java heap space — OOMKilled"),
            _item("GC overhead at 80% — gc_overhead anomaly",
                  source=EvidenceSource.METRICS_AGENT,
                  raw_data={"metric_name": "gc_overhead_percent"}),
        )
        matches = matcher.match(evidence)
        ml_matches = [
            m for m in matches
            if m.pattern_name == PatternName.MEMORY_LEAK
        ]
        assert len(ml_matches) == 1
        assert ml_matches[0].match_score >= 0.5


# ═══════════════════════════════════════════════════════════════
#  TESTS: Network Partition Pattern
# ═══════════════════════════════════════════════════════════════


class TestNetworkPartitionPattern:
    """Test matching the network_partition pattern."""

    def test_network_partition_match(self, matcher: PatternMatcher):
        evidence = _evidence(
            _item("Connection reset by peer — no route to host",
                  raw_data={"service": "svc-a"}),
            _item("Packet loss detected — TCP retransmissions elevated",
                  raw_data={"service": "svc-b"}),
            _item(
                "Packet loss anomaly detected",
                source=EvidenceSource.METRICS_AGENT,
                raw_data={"metric_name": "packet_loss_percent"},
            ),
            _item("Network errors on svc-c",
                  raw_data={"service": "svc-c"}),
        )
        matches = matcher.match(evidence)
        net_matches = [
            m for m in matches
            if m.pattern_name == PatternName.NETWORK_PARTITION
        ]
        assert len(net_matches) == 1


# ═══════════════════════════════════════════════════════════════
#  TESTS: Sorting and Thresholds
# ═══════════════════════════════════════════════════════════════


class TestSortingAndThresholds:
    """Test match sorting and threshold filtering."""

    def test_matches_sorted_by_score_descending(
        self, matcher: PatternMatcher
    ):
        evidence = _evidence(
            _item("Database connection pool exhausted — query timeout"),
            _item("Connection pool wait time — HikariCP"),
            _item("Heap memory increasing over time — GC overhead",
                  raw_data={"trend": "increasing"}),
            _item("OutOfMemoryError detected"),
            _item(
                "DB query anomaly",
                source=EvidenceSource.METRICS_AGENT,
                raw_data={"metric_name": "db_query_duration_ms"},
            ),
        )
        matches = matcher.match(evidence)
        if len(matches) >= 2:
            scores = [m.match_score for m in matches]
            assert scores == sorted(scores, reverse=True)

    def test_high_threshold_filters_weak_matches(self):
        config = HypothesisAgentConfig(
            patterns=PatternThresholds(
                min_match_score=0.9,
                min_indicators_matched=4,
            )
        )
        matcher = PatternMatcher(config)
        evidence = _evidence(
            _item("Database errors in logs"),
        )
        matches = matcher.match(evidence)
        # Should filter out weak matches
        assert len(matches) == 0

    def test_empty_evidence_no_matches(self, matcher: PatternMatcher):
        evidence = _evidence()
        matches = matcher.match(evidence)
        assert len(matches) == 0
