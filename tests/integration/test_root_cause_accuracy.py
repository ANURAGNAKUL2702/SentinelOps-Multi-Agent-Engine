"""
tests/integration/test_root_cause_accuracy.py
Root-cause detection accuracy validation tests.

Verifies that the full pipeline (simulation → observability → 7 agents)
correctly identifies the injected root cause for each scenario with
sufficient confidence (≥85%).

Accuracy target: 100% (4/4 scenarios correct).
Confidence target: ≥0.85 for every scenario.
"""

from __future__ import annotations

import logging
import re

import pytest

from integration.config_manager import ConfigManager
from integration.pipeline import WarRoomPipeline


# ---------------------------------------------------------------------------
# Expected mappings: scenario → keyword that MUST appear in root-cause text
# ---------------------------------------------------------------------------

_EXPECTED: dict[str, dict] = {
    "memory_leak": {
        "keyword": "memory",
        "description": "Memory leak in payment-service",
    },
    "cpu_spike": {
        "keyword": "cpu",
        "description": "CPU saturation in fraud-service",
    },
    "network_latency": {
        "keyword": "network",
        "description": "Network partition / latency in api-gateway",
    },
    "database_timeout": {
        "keyword": "database",
        "description": "Database connection pool exhaustion",
    },
}

_MIN_CONFIDENCE = 0.85


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline() -> WarRoomPipeline:
    """Shared pipeline instance (config loaded once per module)."""
    logging.disable(logging.CRITICAL)
    try:
        config = ConfigManager.load("config.yaml")
        return WarRoomPipeline(config)
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def results(pipeline: WarRoomPipeline) -> dict:
    """Run all 4 scenarios once and cache the results."""
    logging.disable(logging.CRITICAL)
    try:
        out = {}
        for scenario in _EXPECTED:
            out[scenario] = pipeline.run_scenario(
                scenario_name=scenario, formats=[], save_to_db=False,
            )
        return out
    finally:
        logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# Per-scenario accuracy tests
# ---------------------------------------------------------------------------

class TestRootCauseAccuracy:
    """Each scenario must produce the correct root cause."""

    @pytest.mark.parametrize("scenario", list(_EXPECTED))
    def test_root_cause_matches_expected(
        self, scenario: str, results: dict,
    ) -> None:
        result = results[scenario]
        keyword = _EXPECTED[scenario]["keyword"]
        root_cause = str(result.root_cause).lower()

        assert keyword in root_cause, (
            f"[{scenario}] Expected '{keyword}' in root cause, "
            f"got: {result.root_cause!r}"
        )

    @pytest.mark.parametrize("scenario", list(_EXPECTED))
    def test_confidence_above_threshold(
        self, scenario: str, results: dict,
    ) -> None:
        result = results[scenario]
        assert result.confidence >= _MIN_CONFIDENCE, (
            f"[{scenario}] Confidence {result.confidence:.3f} "
            f"< {_MIN_CONFIDENCE} threshold"
        )


# ---------------------------------------------------------------------------
# Aggregate accuracy tests
# ---------------------------------------------------------------------------

class TestAggregateAccuracy:
    """System-level accuracy assertions."""

    def test_all_scenarios_correct(self, results: dict) -> None:
        """100% accuracy: every scenario must detect correct root cause."""
        failures = []
        for scenario, expected in _EXPECTED.items():
            root_cause = str(results[scenario].root_cause).lower()
            if expected["keyword"] not in root_cause:
                failures.append(
                    f"{scenario}: expected '{expected['keyword']}' "
                    f"in '{root_cause[:60]}'"
                )
        assert not failures, (
            f"Root cause accuracy failures:\n" + "\n".join(failures)
        )

    def test_average_confidence_above_threshold(
        self, results: dict,
    ) -> None:
        """Average confidence across all scenarios ≥ 85%."""
        confidences = [results[s].confidence for s in _EXPECTED]
        avg = sum(confidences) / len(confidences)
        assert avg >= _MIN_CONFIDENCE, (
            f"Average confidence {avg:.3f} < {_MIN_CONFIDENCE}"
        )

    def test_no_scenario_below_minimum_confidence(
        self, results: dict,
    ) -> None:
        """Every individual scenario must have confidence ≥ 85%."""
        for scenario in _EXPECTED:
            conf = results[scenario].confidence
            assert conf >= _MIN_CONFIDENCE, (
                f"[{scenario}] Confidence {conf:.3f} < {_MIN_CONFIDENCE}"
            )

    def test_pipeline_run_status_success(self, results: dict) -> None:
        """Every run must complete successfully."""
        for scenario, result in results.items():
            assert result.status == "SUCCESS", (
                f"[{scenario}] Pipeline status: {result.status}"
            )

    def test_distinct_root_causes(self, results: dict) -> None:
        """Each scenario should produce a distinct root cause verdict."""
        verdicts = [str(results[s].root_cause)[:40] for s in _EXPECTED]
        assert len(set(verdicts)) == len(verdicts), (
            f"Duplicate verdicts detected: {verdicts}"
        )


# ---------------------------------------------------------------------------
# Stability test (multiple runs)
# ---------------------------------------------------------------------------

class TestDetectionStability:
    """Root cause detection should be deterministic across runs."""

    def test_repeated_runs_consistent(
        self, pipeline: WarRoomPipeline,
    ) -> None:
        """Run memory_leak 3 times; verdict must be consistent."""
        logging.disable(logging.CRITICAL)
        try:
            verdicts = []
            for _ in range(3):
                r = pipeline.run_scenario(
                    scenario_name="memory_leak",
                    formats=[], save_to_db=False,
                )
                verdicts.append(
                    "memory" in str(r.root_cause).lower()
                )
            assert all(verdicts), (
                f"Inconsistent detection across 3 runs: {verdicts}"
            )
        finally:
            logging.disable(logging.NOTSET)
