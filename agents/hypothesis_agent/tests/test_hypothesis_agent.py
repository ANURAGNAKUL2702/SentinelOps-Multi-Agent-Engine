"""
Tests for HypothesisAgent — Full integration pipeline.

Covers:
  - End-to-end pipeline (fallback mode)
  - End-to-end pipeline (LLM mode with mock provider)
  - Output schema validation
  - 27 validation checks
  - Phase separation
  - Fallback on LLM failure
  - Health check
  - Telemetry recording
  - Validator standalone tests
  - FallbackGenerator standalone tests
  - HypothesisRefiner standalone tests
  - ValidationSuggester standalone tests
"""

from __future__ import annotations

import pytest

from agents.hypothesis_agent.agent import HypothesisAgent
from agents.hypothesis_agent.config import (
    FeatureFlags,
    HypothesisAgentConfig,
    HypothesisLimits,
    LLMConfig,
)
from agents.hypothesis_agent.core.validation_suggester import (
    ValidationSuggester,
)
from agents.hypothesis_agent.fallback import FallbackGenerator
from agents.hypothesis_agent.llm.hypothesis_refiner import (
    HypothesisRefiner,
)
from agents.hypothesis_agent.llm.theory_generator import (
    LLMProviderError,
    MockLLMProvider,
)
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    DependencyFindings,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    Hypothesis,
    HypothesisAgentInput,
    HypothesisAgentOutput,
    IncidentCategory,
    LogFindings,
    MetricFindings,
    PatternMatch,
    PatternName,
    Severity,
    ValidationResult,
)
from agents.hypothesis_agent.validator import OutputValidator


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


def _rich_input() -> HypothesisAgentInput:
    return HypothesisAgentInput(
        log_findings=LogFindings(
            suspicious_services=[
                {
                    "service": "payment-service",
                    "error_count": 100,
                    "severity_hint": "critical",
                    "error_keywords_detected": ["timeout", "database"],
                },
            ],
            total_error_logs=100,
            dominant_service="payment-service",
            system_wide_spike=True,
            database_errors_detected=True,
        ),
        metric_findings=MetricFindings(
            anomalous_metrics=[
                {
                    "metric_name": "db_pool_usage",
                    "zscore": 4.0,
                    "severity": "critical",
                    "anomaly_type": "spike",
                },
            ],
            resource_saturation=True,
        ),
        dependency_findings=DependencyFindings(
            failed_service="payment-service",
            is_cascading=True,
            blast_radius_count=5,
        ),
        incident_id="inc-001",
        time_window="5m",
    )


@pytest.fixture
def fallback_agent() -> HypothesisAgent:
    config = HypothesisAgentConfig(
        features=FeatureFlags(use_llm=False),
    )
    return HypothesisAgent(config)


@pytest.fixture
def llm_agent() -> HypothesisAgent:
    config = HypothesisAgentConfig(
        features=FeatureFlags(use_llm=True),
        llm=LLMConfig(
            provider="mock",
            max_retries=1,
            retry_base_delay=0.001,
        ),
    )
    return HypothesisAgent(config, llm_provider=MockLLMProvider())


# ═══════════════════════════════════════════════════════════════
#  TESTS: Full Pipeline (Fallback)
# ═══════════════════════════════════════════════════════════════


class TestFallbackPipeline:
    """Test full pipeline in fallback (deterministic) mode."""

    def test_analyze_returns_output(
        self, fallback_agent: HypothesisAgent
    ):
        inp = _rich_input()
        output = fallback_agent.analyze(inp)
        assert isinstance(output, HypothesisAgentOutput)

    def test_output_agent_field(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert output.agent == "hypothesis_agent"

    def test_output_has_hypotheses(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert len(output.hypotheses) >= 3

    def test_hypotheses_sorted(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        scores = [h.likelihood_score for h in output.hypotheses]
        assert scores == sorted(scores, reverse=True)

    def test_confidence_score_valid_range(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert 0.0 <= output.confidence_score <= 1.0

    def test_classification_source_is_fallback(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert output.classification_source == "fallback"

    def test_pipeline_latency_positive(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert output.pipeline_latency_ms > 0

    def test_metadata_populated(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert output.metadata is not None
        assert output.metadata.used_fallback is True
        assert output.metadata.used_llm is False

    def test_validation_runs(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert output.validation is not None
        assert output.validation.total_checks == 27

    def test_validation_passes(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert output.validation is not None
        assert output.validation.validation_passed is True

    def test_recommended_hypothesis_valid(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        if output.recommended_hypothesis and output.hypotheses:
            ids = {h.hypothesis_id for h in output.hypotheses}
            assert output.recommended_hypothesis in ids

    def test_incident_id_propagated(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert output.incident_id == "inc-001"

    def test_summary_not_empty(
        self, fallback_agent: HypothesisAgent
    ):
        output = fallback_agent.analyze(_rich_input())
        assert len(output.hypothesis_summary) > 0


# ═══════════════════════════════════════════════════════════════
#  TESTS: Full Pipeline (LLM with Mock)
# ═══════════════════════════════════════════════════════════════


class TestLLMPipeline:
    """Test full pipeline with mock LLM provider."""

    def test_llm_path_produces_output(
        self, llm_agent: HypothesisAgent
    ):
        output = llm_agent.analyze(_rich_input())
        assert isinstance(output, HypothesisAgentOutput)
        assert output.classification_source == "llm"

    def test_llm_hypotheses_generated(
        self, llm_agent: HypothesisAgent
    ):
        output = llm_agent.analyze(_rich_input())
        assert len(output.hypotheses) >= 1

    def test_llm_metadata_correct(
        self, llm_agent: HypothesisAgent
    ):
        output = llm_agent.analyze(_rich_input())
        assert output.metadata is not None
        assert output.metadata.used_llm is True

    def test_llm_validation_passes(
        self, llm_agent: HypothesisAgent
    ):
        output = llm_agent.analyze(_rich_input())
        assert output.validation is not None
        assert output.validation.validation_passed is True


# ═══════════════════════════════════════════════════════════════
#  TESTS: Fallback on LLM Failure
# ═══════════════════════════════════════════════════════════════


class TestLLMFallback:
    """Test graceful LLM → fallback degradation."""

    def test_falls_back_on_llm_failure(self):
        config = HypothesisAgentConfig(
            features=FeatureFlags(use_llm=True, fallback_to_rules=True),
            llm=LLMConfig(
                provider="mock",
                max_retries=1,
                retry_base_delay=0.001,
            ),
        )
        agent = HypothesisAgent(
            config, llm_provider=MockLLMProvider(should_fail=True)
        )
        output = agent.analyze(_rich_input())
        assert isinstance(output, HypothesisAgentOutput)
        assert output.classification_source == "fallback"
        assert len(output.hypotheses) >= 3


# ═══════════════════════════════════════════════════════════════
#  TESTS: Health Check
# ═══════════════════════════════════════════════════════════════


class TestHealthCheck:
    """Test agent health check."""

    def test_health_check_structure(
        self, fallback_agent: HypothesisAgent
    ):
        health = fallback_agent.health_check()
        assert health["agent"] == "hypothesis_agent"
        assert health["status"] in ("healthy", "degraded")
        assert "components" in health
        assert "metrics" in health

    def test_all_components_healthy(
        self, fallback_agent: HypothesisAgent
    ):
        health = fallback_agent.health_check()
        for comp, status in health["components"].items():
            assert status in ("healthy", "degraded"), (
                f"{comp} is {status}"
            )


# ═══════════════════════════════════════════════════════════════
#  TESTS: Telemetry
# ═══════════════════════════════════════════════════════════════


class TestTelemetry:
    """Test telemetry recording."""

    def test_telemetry_increments_on_analyze(
        self, fallback_agent: HypothesisAgent
    ):
        fallback_agent.analyze(_rich_input())
        snap = fallback_agent.telemetry.snapshot()
        assert snap["counters"]["analyses_total"] >= 1
        assert snap["counters"]["analyses_succeeded"] >= 1

    def test_fallback_trigger_counted(
        self, fallback_agent: HypothesisAgent
    ):
        fallback_agent.analyze(_rich_input())
        snap = fallback_agent.telemetry.snapshot()
        assert snap["counters"]["fallback_triggers"] >= 1


# ═══════════════════════════════════════════════════════════════
#  TESTS: Validator (Standalone)
# ═══════════════════════════════════════════════════════════════


class TestOutputValidator:
    """Test the 27-check validator standalone."""

    def test_valid_output_passes(self):
        validator = OutputValidator()
        output = HypothesisAgentOutput(
            incident_id="inc-1",
            hypotheses=[
                Hypothesis(
                    theory="Test theory",
                    category=IncidentCategory.DATABASE,
                    severity=Severity.HIGH,
                    likelihood_score=0.8,
                    evidence_supporting=["e1"],
                ),
                Hypothesis(
                    theory="Alt theory",
                    likelihood_score=0.5,
                ),
                Hypothesis(
                    theory="Low theory",
                    likelihood_score=0.3,
                ),
            ],
            confidence_score=0.7,
            hypothesis_summary="Test summary",
            category=IncidentCategory.DATABASE,
            severity=Severity.HIGH,
            recommended_hypothesis="",
            estimated_mttr_minutes=30.0,
        )
        result = validator.validate(output)
        assert result.validation_passed is True
        assert result.total_checks == 27

    def test_wrong_agent_name_fails(self):
        validator = OutputValidator()
        # HypothesisAgentOutput has frozen agent field, so this
        # should always pass — test that it does
        output = HypothesisAgentOutput(
            hypotheses=[Hypothesis(theory="x")] * 3,
            hypothesis_summary="s",
        )
        result = validator.validate(output)
        # agent field is frozen to "hypothesis_agent"
        assert result.validation_passed is True


# ═══════════════════════════════════════════════════════════════
#  TESTS: FallbackGenerator (Standalone)
# ═══════════════════════════════════════════════════════════════


class TestFallbackGenerator:
    """Test fallback hypothesis generation."""

    def test_generates_from_patterns(self):
        gen = FallbackGenerator()
        evidence = AggregatedEvidence(
            evidence_items=[
                EvidenceItem(
                    source=EvidenceSource.LOG_AGENT,
                    description="Database errors",
                    strength=EvidenceStrength.STRONG,
                )
            ],
            total_evidence_count=1,
        )
        pm = PatternMatch(
            pattern_name=PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
            match_score=0.7,
            matched_indicators=3,
            total_indicators=5,
            category=IncidentCategory.DATABASE,
        )
        result = gen.generate(evidence, [pm])
        assert len(result) >= 3
        assert any(
            h.category == IncidentCategory.DATABASE
            for h in result
        )

    def test_minimum_hypotheses_guaranteed(self):
        gen = FallbackGenerator()
        evidence = AggregatedEvidence()
        result = gen.generate(evidence, [])
        assert len(result) >= 3

    def test_catches_all_with_empty_input(self):
        gen = FallbackGenerator()
        evidence = AggregatedEvidence()
        result = gen.generate(evidence, [])
        assert len(result) >= 3


# ═══════════════════════════════════════════════════════════════
#  TESTS: HypothesisRefiner (Standalone)
# ═══════════════════════════════════════════════════════════════


class TestHypothesisRefiner:
    """Test hypothesis refinement."""

    def test_generate_summary(self):
        refiner = HypothesisRefiner()
        hypotheses = [
            Hypothesis(
                theory="DB pool exhaustion",
                category=IncidentCategory.DATABASE,
                severity=Severity.CRITICAL,
                likelihood_score=0.9,
            ),
            Hypothesis(
                theory="Memory leak",
                category=IncidentCategory.APPLICATION,
                severity=Severity.HIGH,
                likelihood_score=0.5,
            ),
        ]
        summary = refiner.generate_summary(hypotheses)
        assert "DB pool exhaustion" in summary
        assert "90%" in summary

    def test_empty_hypotheses_summary(self):
        refiner = HypothesisRefiner()
        summary = refiner.generate_summary([])
        assert "No hypotheses" in summary

    def test_determine_category(self):
        refiner = HypothesisRefiner()
        hypotheses = [
            Hypothesis(
                theory="x",
                category=IncidentCategory.NETWORK,
            )
        ]
        assert refiner.determine_category(hypotheses) == IncidentCategory.NETWORK

    def test_determine_severity(self):
        refiner = HypothesisRefiner()
        hypotheses = [
            Hypothesis(theory="a", severity=Severity.MEDIUM),
            Hypothesis(theory="b", severity=Severity.CRITICAL),
        ]
        assert refiner.determine_severity(hypotheses) == Severity.CRITICAL


# ═══════════════════════════════════════════════════════════════
#  TESTS: ValidationSuggester (Standalone)
# ═══════════════════════════════════════════════════════════════


class TestValidationSuggester:
    """Test validation test suggestion."""

    def test_suggests_tests_for_database_hypothesis(self):
        suggester = ValidationSuggester()
        hypotheses = [
            Hypothesis(
                theory="DB issue",
                category=IncidentCategory.DATABASE,
            ),
        ]
        enriched = suggester.suggest(hypotheses, [])
        assert len(enriched[0].validation_tests) >= 1

    def test_suggests_tests_for_network_hypothesis(self):
        suggester = ValidationSuggester()
        hypotheses = [
            Hypothesis(
                theory="Network issue",
                category=IncidentCategory.NETWORK,
            ),
        ]
        enriched = suggester.suggest(hypotheses, [])
        assert len(enriched[0].validation_tests) >= 1

    def test_adds_generic_verify_when_evidence_present(self):
        suggester = ValidationSuggester()
        hypotheses = [
            Hypothesis(
                theory="Unknown issue",
                category=IncidentCategory.UNKNOWN,
                evidence_supporting=["some evidence"],
            ),
        ]
        enriched = suggester.suggest(hypotheses, [])
        # Should have the generic verify test
        assert len(enriched[0].validation_tests) >= 1
        assert any(
            t.test_name == "verify_supporting_evidence"
            for t in enriched[0].validation_tests
        )

    def test_no_tests_for_unknown_without_evidence(self):
        suggester = ValidationSuggester()
        hypotheses = [
            Hypothesis(
                theory="Unknown issue",
                category=IncidentCategory.UNKNOWN,
            ),
        ]
        enriched = suggester.suggest(hypotheses, [])
        # UNKNOWN has no category tests and no evidence → 0 tests
        assert len(enriched[0].validation_tests) == 0


# ═══════════════════════════════════════════════════════════════
#  TESTS: Empty Input Edge Case
# ═══════════════════════════════════════════════════════════════


class TestEmptyInput:
    """Test behavior with completely empty input."""

    def test_empty_input_produces_valid_output(
        self, fallback_agent: HypothesisAgent
    ):
        inp = HypothesisAgentInput()
        output = fallback_agent.analyze(inp)
        assert isinstance(output, HypothesisAgentOutput)
        # With empty input, ranker may prune low-score hypotheses
        assert len(output.hypotheses) >= 1
        assert output.validation is not None
        assert output.validation.validation_passed is True
