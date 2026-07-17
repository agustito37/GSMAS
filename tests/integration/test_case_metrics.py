"""Integration tests for the per-case metrics reader (Neo4j-backed).

    uv run pytest -m integration
"""

import pytest

from core.graph.models import Case, Hypothesis, InputSignal, Investigation, Skill, Verdict
from core.graph.store import EdgeSpec
from experiments.metrics import case_metrics


@pytest.mark.integration
async def test_case_metrics_counts_effort_reuse_and_outcome(store):
    """A case with 2 hypotheses, 3 investigations, one skill applied on two of them,
    and a graded verdict: structural effort counts the nodes, reuse counts DISTINCT
    applied skills (1, not 2), and the outcome carries the verdict kind and feedback."""
    signal = InputSignal(raw_content="suspicious login for jdoe")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="determine if the login is malicious", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hyp1 = Hypothesis(description="credential theft", case_id=case.id)
    await store.create_node(hyp1, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    hyp2 = Hypothesis(description="legitimate travel", case_id=case.id)
    await store.create_node(hyp2, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    inv1 = Investigation(description="check auth logs", case_id=case.id)
    await store.create_node(inv1, "Investigation", edges=[EdgeSpec("TESTS", hyp1.id)])
    inv2 = Investigation(description="check mfa events", case_id=case.id)
    await store.create_node(inv2, "Investigation", edges=[EdgeSpec("TESTS", hyp1.id)])
    inv3 = Investigation(description="check travel", case_id=case.id)
    await store.create_node(inv3, "Investigation", edges=[EdgeSpec("TESTS", hyp2.id)])

    role_id = await store.ensure_role("default", "investigator")
    skill = Skill(role_id=role_id, summary="check MFA first", content="query mfa enrollment")
    skill_id = await store.create_skill(skill, case.id)
    await store.mark_skill_applied(inv1.id, skill_id)
    await store.mark_skill_applied(inv2.id, skill_id)  # same skill twice -> reuse stays 1

    verdict = Verdict(kind="resolved", content="malicious", case_id=case.id, feedback="correct")
    await store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])

    metrics = await case_metrics(store, case.id)

    assert metrics.hypotheses == 2
    assert metrics.investigations == 3
    assert metrics.reuse == 1  # distinct skills, not APPLIED edges
    assert metrics.verdict_kind == "resolved"
    assert metrics.feedback == "correct"


@pytest.mark.integration
async def test_case_metrics_open_case_has_no_verdict(store):
    """A case with no Verdict yet: the outcome fields are None (it never closed)."""
    signal = InputSignal(raw_content="benign login")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="check", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])

    metrics = await case_metrics(store, case.id)

    assert metrics.hypotheses == 0
    assert metrics.investigations == 0
    assert metrics.reuse == 0
    assert metrics.verdict_kind is None
    assert metrics.feedback is None
