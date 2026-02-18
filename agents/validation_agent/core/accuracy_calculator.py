"""
File: core/accuracy_calculator.py
Purpose: Compare verdict root cause against ground truth.
Dependencies: difflib (standard library).
Performance: <1ms per call, pure function.

Algorithm 1: Exact match + fuzzy match (SequenceMatcher).
Returns (verdict_correct, accuracy_score).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Tuple


def _normalize(text: str) -> str:
    """Normalize a root cause string for comparison.

    Args:
        text: Raw root cause string.

    Returns:
        Lowercased, whitespace-collapsed, punctuation-stripped string.
    """
    text = text.strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def calculate_accuracy(
    verdict: str,
    ground_truth: str,
    fuzzy_threshold: float = 0.8,
) -> Tuple[bool, float]:
    """Compare verdict root cause against ground truth.

    Performs exact match first, then fuzzy match using
    SequenceMatcher. Returns accuracy score ∈ [0, 1].

    Args:
        verdict: Predicted root cause string.
        ground_truth: Actual root cause string.
        fuzzy_threshold: Minimum fuzzy score to consider correct.

    Returns:
        Tuple of (verdict_correct, accuracy_score).
    """
    if not verdict or not ground_truth:
        return False, 0.0

    norm_verdict = _normalize(verdict)
    norm_truth = _normalize(ground_truth)

    if not norm_verdict or not norm_truth:
        return False, 0.0

    # Exact match
    if norm_verdict == norm_truth:
        return True, 1.0

    # Check substring containment (partial credit)
    if norm_truth in norm_verdict or norm_verdict in norm_truth:
        longer = max(len(norm_verdict), len(norm_truth))
        shorter = min(len(norm_verdict), len(norm_truth))
        containment_score = shorter / longer if longer > 0 else 0.0
        # Boost containment score since it's a strong signal
        score = min(1.0, containment_score + 0.2)
        return score >= fuzzy_threshold, round(score, 4)

    # Fuzzy match via SequenceMatcher
    ratio = SequenceMatcher(None, norm_verdict, norm_truth).ratio()

    # Token-level overlap as secondary signal
    verdict_tokens = set(norm_verdict.split())
    truth_tokens = set(norm_truth.split())
    if verdict_tokens and truth_tokens:
        token_overlap = len(verdict_tokens & truth_tokens) / len(
            verdict_tokens | truth_tokens
        )
        # Weighted average: 60% sequence match, 40% token overlap
        score = 0.6 * ratio + 0.4 * token_overlap
    else:
        score = ratio

    score = round(min(1.0, max(0.0, score)), 4)
    is_correct = score >= fuzzy_threshold

    return is_correct, score
