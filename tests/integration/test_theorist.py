"""Integration tests for the Theorist: reaction 1 (open the case) and reaction 2
(the generative motor: triage each Evidence -> suggest / refute / nothing).

    uv run pytest -m integration
"""

import json

import pytest

from core.agents.base import Agent
from core.graph.models import Case, Evidence, Hypothesis, InputSignal, Investigation
from core.graph.store import EdgeSpec
from core.providers.base import LLMResponse
from domain.roles.theorist import Theorist
from tests.mocks.mock_provider import MockProvider


async def _seed_case(store):
    """A case mid-investigation: one hypothesis, one investigation, one untriaged
    Evidence just produced (the trigger of the generative motor)."""
    signal = InputSignal(raw_content="suspicious login from a new country")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="determine if the login is malicious", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hypothesis = Hypothesis(description="credential theft", case_id=case.id)
    await store.create_node(hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    investigation = Investigation(description="check auth logs", case_id=case.id)
    await store.create_node(
        investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
    )
    evidence = Evidence(
        content="the account then accessed internal server X",
        rationale="found in the auth logs",
        case_id=case.id,
    )
    await store.create_node(evidence, "Evidence", edges=[EdgeSpec("PRODUCES", investigation.id)])
    return case, hypothesis, investigation, evidence


async def _run_triage(store, provider) -> None:
    """Simulate the orchestrator for reaction 2: claim the evidence, spawn an agent
    stamped with the engine, run it."""
    theorist = Theorist(store)
    work = await theorist._claim_evidence()
    assert work is not None
    await Agent(theorist, theorist.reactions()[1].execute, work, provider=provider).run()


@pytest.mark.integration
async def test_theorist_opens_case_and_derives_hypotheses(store):
    """Given a signal, the Theorist opens a Case (linked by OPENS) with the LLM's
    objective, and derives the hypotheses (linked by DERIVES), each carrying its
    rationale (the framework rule: every LLM emission persists its why)."""
    provider = MockProvider([LLMResponse(content=json.dumps({
        "objective": "determine if the login is malicious",
        "rationale": "the signal describes an anomalous login",
        "hypotheses": [
            {"description": "credential theft", "rationale": "geo mismatch"},
            {"description": "legitimate travel", "rationale": "user may be abroad"},
        ],
    }))])
    theorist = Theorist(store)

    signal = InputSignal(raw_content="suspicious login from a new country")
    await store.create_node(signal, "InputSignal")

    # simulate the orchestrator: claim the work, then run an engine-stamped agent
    work = await theorist._claim_signal()
    assert work is not None
    await Agent(theorist, theorist.reactions()[0].execute, work, provider=provider).run()

    cases = await store.query_nodes("Case", {})
    assert len(cases) == 1
    assert cases[0].objective == "determine if the login is malicious"
    assert cases[0].rationale == "the signal describes an anomalous login"

    # the Case was opened FROM the signal (OPENS edge)
    opened = await store.get_neighbors(signal.id, "OPENS", target_label="Case")
    assert [c.id for c in opened] == [cases[0].id]

    # the hypotheses were derived under the Case (DERIVES edges), each with its why
    hyps = await store.get_neighbors(cases[0].id, "DERIVES", target_label="Hypothesis")
    assert {h.description for h in hyps} == {"credential theft", "legitimate travel"}
    assert {h.rationale for h in hyps} == {"geo mismatch", "user may be abroad"}
    # each initial hypothesis is its own branch root
    assert all(h.root_id == h.id for h in hyps)


# ---- reaction 2: the generative motor ----


@pytest.mark.integration
async def test_triage_generates_a_suggested_hypothesis(store):
    """Evidence whose finding reveals something new: the Theorist derives a NEW
    hypothesis, born connected (DERIVES from the Case, SUGGESTS from the evidence),
    inheriting the parent's branch, claimable by the Planner; the evidence ends up
    triaged."""
    case, hypothesis, _, evidence = await _seed_case(store)
    provider = MockProvider([LLMResponse(content=json.dumps({
        "new_hypotheses": [
            {"description": "lateral movement", "rationale": "unexpected internal access"}
        ],
        "refuted": [],
    }))])

    await _run_triage(store, provider)

    hyps = await store.query_nodes("Hypothesis", {"case_id": case.id})
    assert len(hyps) == 2
    generated = next(h for h in hyps if h.id != hypothesis.id)
    assert generated.description == "lateral movement"
    assert generated.rationale == "unexpected internal access"
    assert generated.root_id == hypothesis.root_id  # inherits the parent's branch
    assert generated.claim_state == "pending"  # the Planner can claim and plan it

    suggested = await store.get_neighbors(evidence.id, "SUGGESTS", target_label="Hypothesis")
    assert [h.id for h in suggested] == [generated.id]
    derived = await store.get_neighbors(case.id, "DERIVES", target_label="Hypothesis")
    assert {h.id for h in derived} == {hypothesis.id, generated.id}

    refreshed = await store.get_node(evidence.id)
    assert refreshed is not None and refreshed.triaged is True


@pytest.mark.integration
async def test_triage_refutes_and_skips_pending_work(store):
    """Evidence that conclusively contradicts a hypothesis: the Theorist marks it
    refuted (recording the judgment) and its not-yet-claimed investigations are
    skipped, terminally (no one can claim them anymore)."""
    case, hypothesis, _, evidence = await _seed_case(store)
    pending = Investigation(description="interview the user", case_id=case.id)
    await store.create_node(pending, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)])
    provider = MockProvider([LLMResponse(content=json.dumps({
        "new_hypotheses": [],
        "refuted": [
            {"hypothesis_id": hypothesis.id, "rationale": "the login was confirmed legitimate"}
        ],
    }))])

    await _run_triage(store, provider)

    refuted = await store.get_node(hypothesis.id)
    assert refuted is not None
    assert refuted.status == "refuted"
    assert refuted.refutation_reason == "the login was confirmed legitimate"

    skipped = await store.get_node(pending.id)
    assert skipped is not None
    assert skipped.status == "skipped"
    assert "hypothesis refuted" in skipped.skip_reason
    assert skipped.claim_state == "done"  # not claimable: the line stopped consuming work


@pytest.mark.integration
async def test_triage_respects_the_branch_limit(store):
    """A full branch (4 hypotheses sharing the root) rejects further generation: the
    count-and-create is atomic, so the suggested hypothesis is NOT created - but the
    judgment still completes (evidence triaged)."""
    case, hypothesis, _, evidence = await _seed_case(store)
    for i in range(3):  # fill the branch: 4 hypotheses in total, same root
        sibling = Hypothesis(
            description=f"variant {i}", case_id=case.id, root_id=hypothesis.root_id
        )
        await store.create_node(sibling, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    provider = MockProvider([LLMResponse(content=json.dumps({
        "new_hypotheses": [{"description": "one too many", "rationale": "x"}],
        "refuted": [],
    }))])

    await _run_triage(store, provider)

    hyps = await store.query_nodes("Hypothesis", {"case_id": case.id})
    assert len(hyps) == 4  # the branch was full: nothing was created
    assert await store.get_neighbors(evidence.id, "SUGGESTS", target_label="Hypothesis") == []
    refreshed = await store.get_node(evidence.id)
    assert refreshed is not None and refreshed.triaged is True
