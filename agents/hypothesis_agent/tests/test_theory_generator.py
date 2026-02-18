"""
Tests for llm/theory_generator.py — Algorithm 3: LLM Hypothesis Generation.

Covers:
  - CircuitBreaker state transitions
  - ResponseCache get/put/TTL/eviction
  - MockLLMProvider success and failure modes
  - TheoryGenerator full pipeline: cache → circuit → LLM → parse
  - Response parsing robustness
  - Retry logic
  - Telemetry recording
"""

from __future__ import annotations

import time
import pytest

from agents.hypothesis_agent.config import (
    HypothesisAgentConfig,
    FeatureFlags,
    LLMConfig,
)
from agents.hypothesis_agent.llm.theory_generator import (
    CircuitBreaker,
    CircuitState,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
    ResponseCache,
    TheoryGenerator,
)
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    Hypothesis,
    IncidentCategory,
    PatternMatch,
    PatternName,
    Severity,
)
from agents.hypothesis_agent.telemetry import TelemetryCollector


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _make_evidence() -> AggregatedEvidence:
    return AggregatedEvidence(
        evidence_items=[
            EvidenceItem(
                source=EvidenceSource.LOG_AGENT,
                description="Database errors detected",
                strength=EvidenceStrength.STRONG,
                severity=Severity.HIGH,
            ),
        ],
        total_evidence_count=1,
        strong_evidence_count=1,
        sources_represented=[EvidenceSource.LOG_AGENT],
        dominant_severity=Severity.HIGH,
    )


def _make_pattern_matches() -> list[PatternMatch]:
    return [
        PatternMatch(
            pattern_name=PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
            match_score=0.7,
            matched_indicators=3,
            total_indicators=5,
            category=IncidentCategory.DATABASE,
        )
    ]


def _make_config(**overrides) -> HypothesisAgentConfig:
    flags = overrides.pop("features", FeatureFlags(use_llm=True))
    llm = overrides.pop("llm", LLMConfig(
        max_retries=1,
        retry_base_delay=0.001,
        cache_ttl_seconds=60,
    ))
    return HypothesisAgentConfig(features=flags, llm=llm, **overrides)


# ═══════════════════════════════════════════════════════════════
#  TESTS: CircuitBreaker
# ═══════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    """Test circuit breaker state machine."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute()

    def test_closes_after_success_threshold(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
            success_threshold=2,
        )
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
        )
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


# ═══════════════════════════════════════════════════════════════
#  TESTS: ResponseCache
# ═══════════════════════════════════════════════════════════════


class TestResponseCache:
    """Test LLM response cache."""

    def test_put_and_get(self):
        cache = ResponseCache(ttl_seconds=60)
        h = Hypothesis(theory="test")
        cache.put("key1", [h])
        result = cache.get("key1")
        assert result is not None
        assert len(result) == 1
        assert result[0].theory == "test"

    def test_get_returns_none_for_missing_key(self):
        cache = ResponseCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = ResponseCache(ttl_seconds=0)
        cache.put("key1", [Hypothesis(theory="old")])
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_max_entries_eviction(self):
        cache = ResponseCache(ttl_seconds=60, max_entries=2)
        cache.put("a", [Hypothesis(theory="a")])
        time.sleep(0.001)
        cache.put("b", [Hypothesis(theory="b")])
        time.sleep(0.001)
        cache.put("c", [Hypothesis(theory="c")])
        # "a" should have been evicted (oldest)
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_compute_key_deterministic(self):
        signals = {"evidence_count": 5, "sources": ["log"]}
        k1 = ResponseCache.compute_key(signals)
        k2 = ResponseCache.compute_key(signals)
        assert k1 == k2

    def test_compute_key_different_for_different_signals(self):
        k1 = ResponseCache.compute_key({"a": 1})
        k2 = ResponseCache.compute_key({"a": 2})
        assert k1 != k2

    def test_clear(self):
        cache = ResponseCache()
        cache.put("x", [Hypothesis(theory="x")])
        cache.clear()
        assert cache.get("x") is None
        assert cache.size == 0


# ═══════════════════════════════════════════════════════════════
#  TESTS: MockLLMProvider
# ═══════════════════════════════════════════════════════════════


class TestMockLLMProvider:
    """Test mock LLM provider."""

    def test_success_returns_hypotheses(self):
        provider = MockLLMProvider()
        response = provider.call("system", "user")
        assert "hypotheses" in response
        assert len(response["hypotheses"]) == 3

    def test_should_fail_raises(self):
        provider = MockLLMProvider(should_fail=True)
        with pytest.raises(LLMProviderError):
            provider.call("system", "user")

    def test_failure_count_then_succeed(self):
        provider = MockLLMProvider(failure_count=2)
        with pytest.raises(LLMProviderError):
            provider.call("system", "user")
        with pytest.raises(LLMProviderError):
            provider.call("system", "user")
        # Third call succeeds
        response = provider.call("system", "user")
        assert "hypotheses" in response

    def test_count_tokens(self):
        provider = MockLLMProvider()
        assert provider.count_tokens("test") >= 1


# ═══════════════════════════════════════════════════════════════
#  TESTS: TheoryGenerator
# ═══════════════════════════════════════════════════════════════


class TestTheoryGenerator:
    """Test the full theory generation pipeline."""

    def test_generate_returns_hypotheses(self):
        config = _make_config()
        gen = TheoryGenerator(
            config=config, provider=MockLLMProvider()
        )
        result = gen.generate(
            evidence=_make_evidence(),
            pattern_matches=_make_pattern_matches(),
        )
        assert isinstance(result, list)
        assert len(result) >= 1
        for h in result:
            assert isinstance(h, Hypothesis)
            assert h.theory

    def test_cache_hit_avoids_llm_call(self):
        config = _make_config()
        telemetry = TelemetryCollector()
        gen = TheoryGenerator(
            config=config,
            provider=MockLLMProvider(),
            telemetry=telemetry,
        )
        evidence = _make_evidence()
        patterns = _make_pattern_matches()

        # First call → cache miss
        result1 = gen.generate(evidence, patterns)
        assert telemetry.cache_misses.value == 1

        # Second call → cache hit
        result2 = gen.generate(evidence, patterns)
        assert telemetry.cache_hits.value == 1
        assert len(result2) == len(result1)

    def test_circuit_breaker_integration(self):
        config = _make_config(
            llm=LLMConfig(
                max_retries=1,
                retry_base_delay=0.001,
                circuit_failure_threshold=1,
                cache_ttl_seconds=60,
            ),
            features=FeatureFlags(use_llm=True, enable_caching=False),
        )
        provider = MockLLMProvider(should_fail=True)
        gen = TheoryGenerator(
            config=config, provider=provider
        )

        # First call fails → circuit opens
        with pytest.raises(LLMProviderError):
            gen.generate(_make_evidence(), [])

        assert gen.circuit_state == CircuitState.OPEN

        # Next call rejected by circuit
        with pytest.raises(LLMProviderError, match="Circuit breaker"):
            gen.generate(_make_evidence(), [])

    def test_telemetry_recorded_on_success(self):
        config = _make_config()
        telemetry = TelemetryCollector()
        gen = TheoryGenerator(
            config=config,
            provider=MockLLMProvider(),
            telemetry=telemetry,
        )
        gen.generate(_make_evidence(), [])
        assert telemetry.llm_calls_total.value >= 1

    def test_parse_handles_invalid_response(self):
        """Custom provider that returns malformed data."""

        class BadProvider(LLMProvider):
            def call(self, system_prompt, user_prompt, timeout_seconds=15):
                return {"hypotheses": "not_a_list"}

        config = _make_config(
            features=FeatureFlags(use_llm=True, enable_caching=False)
        )
        gen = TheoryGenerator(
            config=config, provider=BadProvider()
        )
        result = gen.generate(_make_evidence(), [])
        # Should still return at least 1 fallback hypothesis
        assert len(result) >= 1
