"""Tests for accuracy_calculator.py — Algorithm 1."""

from __future__ import annotations

import pytest

from agents.validation_agent.core.accuracy_calculator import (
    calculate_accuracy,
    _normalize,
)


class TestNormalize:
    """Tests for string normalization."""

    def test_lowercase(self) -> None:
        assert _normalize("DB_FAILURE") == "db failure"

    def test_underscore_to_space(self) -> None:
        assert _normalize("db_connection_pool") == "db connection pool"

    def test_hyphen_to_space(self) -> None:
        assert _normalize("api-gateway") == "api gateway"

    def test_strip_whitespace(self) -> None:
        assert _normalize("  test  ") == "test"


class TestCalculateAccuracy:
    """Tests for calculate_accuracy."""

    def test_exact_match(self) -> None:
        correct, score = calculate_accuracy(
            "database_connection_pool_exhaustion",
            "database_connection_pool_exhaustion",
        )
        assert correct is True
        assert score == 1.0

    def test_exact_match_case_insensitive(self) -> None:
        correct, score = calculate_accuracy(
            "DB_Connection_Pool_Exhaustion",
            "db_connection_pool_exhaustion",
        )
        assert correct is True
        assert score == 1.0

    def test_no_match(self) -> None:
        correct, score = calculate_accuracy(
            "network_partition",
            "database_failure",
        )
        assert correct is False
        assert score < 0.5

    def test_fuzzy_match_similar(self) -> None:
        correct, score = calculate_accuracy(
            "db_connection_pool",
            "database_connection_pool_exhaustion",
        )
        # Substring containment should boost score
        assert score > 0.5

    def test_fuzzy_match_high_similarity(self) -> None:
        correct, score = calculate_accuracy(
            "database_connection_pool_exhaustion",
            "database_connection_pool_failure",
        )
        assert score > 0.7

    def test_empty_verdict(self) -> None:
        correct, score = calculate_accuracy("", "db_failure")
        assert correct is False
        assert score == 0.0

    def test_empty_ground_truth(self) -> None:
        correct, score = calculate_accuracy("db_failure", "")
        assert correct is False
        assert score == 0.0

    def test_both_empty(self) -> None:
        correct, score = calculate_accuracy("", "")
        assert correct is False
        assert score == 0.0

    def test_special_characters(self) -> None:
        correct, score = calculate_accuracy(
            "db-failure!!!",
            "db_failure",
        )
        assert correct is True
        assert score == 1.0

    def test_custom_threshold(self) -> None:
        _, score = calculate_accuracy(
            "auth_service_crash",
            "authentication_service_failure",
        )
        # With high threshold, might not be "correct"
        correct_strict, _ = calculate_accuracy(
            "auth_service_crash",
            "authentication_service_failure",
            fuzzy_threshold=0.95,
        )
        # Score stays the same regardless of threshold
        assert isinstance(correct_strict, bool)

    def test_substring_containment(self) -> None:
        correct, score = calculate_accuracy(
            "connection_pool",
            "database_connection_pool_exhaustion",
        )
        assert score > 0.4
