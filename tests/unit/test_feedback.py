"""Unit tests for the programmatic feedback gate (a pure, deterministic function).

    uv run pytest
"""

import pytest

from core.learning.feedback import grade


@pytest.mark.unit
def test_grade_exact_kind_is_correct():
    assert grade("confirmed", "confirmed") == "correct"


@pytest.mark.unit
def test_grade_committed_wrong_kind_is_incorrect():
    assert grade("confirmed", "refuted") == "incorrect"


@pytest.mark.unit
def test_grade_hedged_verdict_is_partial():
    """Inconclusive when the truth was definite: partial credit, not wrong."""
    assert grade("inconclusive", "confirmed") == "partial"


@pytest.mark.unit
def test_grade_no_verdict_is_incorrect():
    assert grade(None, "confirmed") == "incorrect"
