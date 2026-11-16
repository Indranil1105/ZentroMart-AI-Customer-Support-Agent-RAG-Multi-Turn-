"""Evaluation metrics for ZentroMart support agent scoring.

Provides functions to measure retrieval quality, decision accuracy,
and response content completeness against the practice dataset's gold standards.
"""

from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Set, Union


def retrieval_precision(predicted_sections: List[str], gold_sections: List[str]) -> float:
    """Fraction of predicted sections that appear in the gold set.

    Args:
        predicted_sections: Section IDs the agent retrieved.
        gold_sections: Gold-standard section IDs from practice data.

    Returns:
        float: Precision score between 0.0 and 1.0.
    """
    if not predicted_sections:
        return 0.0
    pred = _normalize_sections(predicted_sections)
    gold = _normalize_sections(gold_sections)
    if not pred:
        return 0.0
    hits = pred & gold
    return len(hits) / len(pred)


def retrieval_recall(predicted_sections: List[str], gold_sections: List[str]) -> float:
    """Fraction of gold sections that were retrieved by the agent.

    Args:
        predicted_sections: Section IDs the agent retrieved.
        gold_sections: Gold-standard section IDs from practice data.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """
    if not gold_sections:
        return 1.0  # Nothing to recall
    pred = _normalize_sections(predicted_sections)
    gold = _normalize_sections(gold_sections)
    if not gold:
        return 1.0
    hits = pred & gold
    return len(hits) / len(gold)


def decision_accuracy(
    predicted: str, expected: str, acceptable: Optional[str] = None
) -> bool:
    """Check if the predicted decision matches expected or any acceptable alternative.

    Args:
        predicted: The agent's decision (answer/clarify/escalate/refuse).
        expected: The expected decision from practice data.
        acceptable: Optional ';'-separated list of acceptable decisions.

    Returns:
        bool: True if the predicted decision is acceptable.
    """
    pred = predicted.strip().lower() if predicted else ""
    exp = expected.strip().lower() if expected else ""

    if pred == exp:
        return True

    if acceptable:
        acceptable_set = {
            a.strip().lower()
            for a in acceptable.replace(";", " ; ").split(" ; ")
            if a.strip()
        }
        return pred in acceptable_set

    return False


def must_include_score(response: str, must_include: str) -> float:
    """Score how many required facts are present in the response.

    Facts are separated by ' ; '. Within each fact, alternatives are separated
    by ' / ' — any one alternative is sufficient. Matching is case-insensitive
    and accepts equivalent date formats.

    Args:
        response: The agent's response text.
        must_include: Required facts string from practice data.

    Returns:
        float: Fraction of required facts found (0.0 to 1.0).
    """
    if not must_include or not must_include.strip():
        return 1.0  # No requirements = perfect score

    response_lower = response.lower() if response else ""
    facts = [f.strip() for f in must_include.split(" ; ") if f.strip()]

    if not facts:
        return 1.0

    hits = 0
    for fact in facts:
        alternatives = [alt.strip() for alt in fact.split(" / ") if alt.strip()]
        if _any_alternative_found(response_lower, alternatives):
            hits += 1

    return hits / len(facts)


def must_not_include_score(response: str, must_not_include: str) -> bool:
    """Check that the response does NOT contain forbidden content.

    Args:
        response: The agent's response text.
        must_not_include: Forbidden phrases from practice data (';'-separated).

    Returns:
        bool: True if the response is clean (no forbidden content found).
    """
    if not must_not_include or not must_not_include.strip():
        return True

    response_lower = response.lower() if response else ""
    forbidden = [f.strip() for f in must_not_include.split(" ; ") if f.strip()]

    for phrase in forbidden:
        if phrase.lower() in response_lower:
            return False
    return True


def overall_score(
    retrieval_p: float,
    retrieval_r: float,
    decision_correct: bool,
    must_inc: float,
    must_not: bool,
) -> float:
    """Compute a weighted composite score.

    Weights:
        - Decision accuracy: 30%
        - Must-include coverage: 30%
        - Must-not-include compliance: 10%
        - Retrieval precision: 15%
        - Retrieval recall: 15%

    Args:
        retrieval_p: Retrieval precision (0-1).
        retrieval_r: Retrieval recall (0-1).
        decision_correct: Whether the decision was correct.
        must_inc: Must-include coverage score (0-1).
        must_not: Must-not-include compliance (True/False).

    Returns:
        float: Weighted score between 0.0 and 1.0.
    """
    return (
        0.30 * float(decision_correct)
        + 0.30 * must_inc
        + 0.10 * float(must_not)
        + 0.15 * retrieval_p
        + 0.15 * retrieval_r
    )


# ── Helper functions ──────────────────────────────────────────────────────────


def _normalize_sections(sections: List[str]) -> Set[str]:
    """Normalize section IDs: strip whitespace, uppercase, remove brackets."""
    result = set()
    for s in sections:
        if not s:
            continue
        cleaned = s.strip().strip("[]").strip()
        if cleaned:
            result.add(cleaned)
    return result


def _any_alternative_found(response_lower: str, alternatives: List[str]) -> bool:
    """Check if any alternative fact is present in the response."""
    for alt in alternatives:
        alt_lower = alt.lower().strip()
        if not alt_lower:
            continue

        # Direct substring match
        if alt_lower in response_lower:
            return True

        # Try date equivalence: "2026-10-27" == "27 October 2026" == "27 october"
        date_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", alt_lower)
        if date_match:
            y, m, d = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
            month_names = {
                1: "january", 2: "february", 3: "march", 4: "april",
                5: "may", 6: "june", 7: "july", 8: "august",
                9: "september", 10: "october", 11: "november", 12: "december",
            }
            month_name = month_names.get(m, "")
            # Check various date formats
            date_variants = [
                f"{d} {month_name} {y}",
                f"{d} {month_name}",
                f"{d}{_ordinal(d)} {month_name}",
                f"{d}/{m}/{y}",
                f"{d:02d}/{m:02d}/{y}",
            ]
            for variant in date_variants:
                if variant in response_lower:
                    return True

        # Try numeric equivalence: "49" matches "₹49", "Rs.49", "Rs 49"
        if re.match(r"^\d+$", alt_lower):
            # Check for the number with optional currency prefix
            pattern = r"(?:₹|rs\.?\s*|inr\s*)?" + re.escape(alt_lower) + r"\b"
            if re.search(pattern, response_lower):
                return True

    return False


def _ordinal(n: int) -> str:
    """Return ordinal suffix for a number."""
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
