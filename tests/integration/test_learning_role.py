"""Integration tests for the retrospective (the reflection that writes skills), driven
through the Investigator (a LearningRole) with a MockProvider.

    uv run pytest -m integration
"""

import json
from typing import Literal

import pytest

from core.agents.base import Agent
from core.graph.models import (
    Case,
    Evidence,
    Hypothesis,
    InputSignal,
    Investigation,
    Skill,
    Verdict,
)
from core.graph.store import EdgeSpec
from core.providers.base import LLMResponse
from core.tools.base import ToolRegistry
from domain.roles.investigator import Investigator
from tests.mocks.mock_provider import MockProvider


async def _closed_case(
    store, feedback: Literal["correct", "incorrect", "partial"] | None
) -> tuple[Case, Investigation]:
    """A minimal closed case (concluded by a Verdict carrying `feedback`), plus the
    Investigation used as the work unit that skills get APPLIED to."""
    signal = InputSignal(raw_content="suspicious login for jdoe")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="determine if the login is malicious", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hyp = Hypothesis(description="credential theft", case_id=case.id)
    await store.create_node(hyp, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    inv = Investigation(description="check jdoe auth logs", case_id=case.id)
    await store.create_node(inv, "Investigation", edges=[EdgeSpec("TESTS", hyp.id)])
    ev = Evidence(content="mfa enrolled from a new device", case_id=case.id)
    await store.create_node(ev, "Evidence", edges=[EdgeSpec("PRODUCES", inv.id)])
    verdict = Verdict(kind="resolved", content="done", case_id=case.id, feedback=feedback)
    await store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])
    return case, inv


def _retro(role):
    return next(r for r in role.all_reactions() if ("node_updated", "Verdict") in r.triggers)


async def _run_retrospection(store, distillation: str) -> Investigator:
    """Claim and run one retrospection, with the LLM returning `distillation`."""
    investigator = Investigator(store)
    reaction = _retro(investigator)
    work = await reaction.claim()
    assert work is not None
    agent = Agent(
        investigator,
        reaction.execute,
        work,
        provider=MockProvider([LLMResponse(content=distillation)]),
        tools=ToolRegistry([]),
    )
    await agent.run()
    return investigator


_CREATE = json.dumps({"changes": [{
    "action": "create",
    "when": "geo-anomalous login for a user",
    "goal": "rule out credential theft before escalating",
    "steps": ["query mfa enrollment events", "check helpdesk tickets", "cross-check prior logins"],
    "caveats": ["if the user reports travel, likely a false positive"],
    "rationale": "the winning path started from the mfa enrollment",
}]})


@pytest.mark.integration
async def test_retrospective_creates_and_corroborates_on_correct_feedback(store):
    """Correct feedback: the applied skill is corroborated (deterministic) and the LLM
    distills a new procedure, stored as when + numbered steps."""
    origin, _ = await _closed_case(store, None)  # where the applied skill was born
    role_id = await store.ensure_role("default", "investigator")
    applied = Skill(role_id=role_id, summary="old", content="old steps")
    applied_id = await store.create_skill(applied, origin.id)  # corroborations = 1
    case, inv = await _closed_case(store, "correct")
    await store.mark_skill_applied(inv.id, applied_id)

    await _run_retrospection(store, _CREATE)

    # the applied skill was corroborated by this case (origin + here = 2)
    support = await store.get_skill_support(applied_id)
    assert support["corroborations"] == 2 and support["refutations"] == 0
    # a new procedure was distilled: when -> summary, goal + numbered steps + caveats
    # rendered into content, and its provenance into rationale
    skills = {s.summary: s for s in await store.get_active_skills(role_id)}
    assert "geo-anomalous login for a user" in skills
    distilled = skills["geo-anomalous login for a user"]
    assert "Goal: rule out credential theft before escalating" in distilled.content
    assert "1. query mfa enrollment events" in distilled.content
    assert "Watch out:\n- if the user reports travel" in distilled.content
    assert distilled.rationale == "the winning path started from the mfa enrollment"


@pytest.mark.integration
async def test_retrospective_refutes_and_refines_below_threshold(store):
    """Incorrect feedback, still below the retire threshold: the applied skill is
    refuted (deterministic) AND refined by the LLM (a fix, a fresh chance): not
    retired."""
    role_id = await store.ensure_role("default", "investigator")
    applied = Skill(role_id=role_id, summary="old when", content="old steps")
    case, inv = await _closed_case(store, "incorrect")
    applied_id = await store.create_skill(applied, case.id)  # corroborations = 1
    await store.mark_skill_applied(inv.id, applied_id)

    refine = json.dumps({"changes": [{
        "action": "refine", "skill_id": applied_id,
        "when": "geo-anomalous login, but confirm travel first",
        "goal": "avoid false positives on legitimate travel",
        "steps": ["ask the user if they travelled", "only then query mfa"],
        "rationale": "the case was legitimate travel; the old procedure over-triggered",
    }]})
    await _run_retrospection(store, refine)

    support = await store.get_skill_support(applied_id)
    assert support["refutations"] == 1  # refuted (deterministic)
    active = {s.id: s for s in await store.get_active_skills(role_id)}
    assert applied_id in active  # NOT retired (total 2 < threshold)
    refined = active[applied_id]
    assert refined.summary.startswith("geo-anomalous login, but confirm")  # trigger tightened
    assert "Goal: avoid false positives on legitimate travel" in refined.content
    assert "1. ask the user if they travelled" in refined.content


@pytest.mark.integration
async def test_retrospective_retires_at_threshold_and_drops_refine(store):
    """Incorrect feedback that pushes the applied skill over the threshold: it is
    retired, and the LLM's refine of it is dropped (refine and retire are exclusive)."""
    origin, _ = await _closed_case(store, None)  # where the skill was born
    role_id = await store.ensure_role("default", "investigator")
    applied = Skill(role_id=role_id, summary="keep summary", content="keep content")
    applied_id = await store.create_skill(applied, origin.id)  # corroborations = 1
    prior, _ = await _closed_case(store, None)
    await store.add_refutation(applied_id, prior.id)  # refutations = 1
    case, inv = await _closed_case(store, "incorrect")
    await store.mark_skill_applied(inv.id, applied_id)

    refine = json.dumps({"changes": [{
        "action": "refine", "skill_id": applied_id,
        "when": "new", "steps": ["new"], "rationale": "x",
    }]})
    # this case's refutation -> refutations 2 > corroborations 1, total 3 -> retire
    await _run_retrospection(store, refine)

    active_ids = [s.id for s in await store.get_active_skills(role_id)]
    assert applied_id not in active_ids  # retired
    retired = await store.get_node(applied_id)
    assert retired is not None and retired.status == "retired"
    assert retired.summary == "keep summary"  # the refine was dropped


@pytest.mark.integration
async def test_retrospective_is_idempotent_per_case(store):
    """A case is retrospected once: the RETROSPECTED marker makes the second claim
    return nothing."""
    await _closed_case(store, "correct")
    investigator = Investigator(store)
    reaction = _retro(investigator)

    assert await reaction.claim() is not None
    assert await reaction.claim() is None


@pytest.mark.integration
async def test_retrospective_ignores_cases_without_feedback(store):
    """A closed case whose verdict has NO feedback is not claimed for retrospection."""
    await _closed_case(store, None)
    assert await _retro(Investigator(store)).claim() is None


@pytest.mark.integration
async def test_retrospective_partial_feedback_is_vitality_neutral(store):
    """Partial feedback (the system hedged): the applied skill's vitality does NOT move
    (no corroboration, no refutation), but the LLM distillation still runs."""
    origin, _ = await _closed_case(store, None)  # where the applied skill was born
    role_id = await store.ensure_role("default", "investigator")
    applied = Skill(role_id=role_id, summary="old", content="old steps")
    applied_id = await store.create_skill(applied, origin.id)  # corroborations = 1 (origin)
    case, inv = await _closed_case(store, "partial")
    await store.mark_skill_applied(inv.id, applied_id)

    await _run_retrospection(store, _CREATE)

    # vitality unchanged: only the origin corroboration, no new corroboration/refutation
    support = await store.get_skill_support(applied_id)
    assert support["corroborations"] == 1 and support["refutations"] == 0
    # the LLM distillation still ran (a new procedure was created)
    summaries = {s.summary for s in await store.get_active_skills(role_id)}
    assert "geo-anomalous login for a user" in summaries
