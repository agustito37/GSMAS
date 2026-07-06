"""Integration tests for the structural invariants (Neo4j docker instance).

For each invariant: one test on a fully valid subgraph (expects no violations) and
one test per violation it detects, on a minimal isolated subgraph. The store now
forbids invalid states by construction ("born connected"), so the violation tests
INJECT them with raw Cypher (the `raw` fixture): the checkers are the defense in
depth against writers that bypass the store. Run with:
    uv run pytest -m integration
"""

import pytest

from core.graph.invariants import (
    check_all,
    check_investigation_outcome,
    check_mandatory_validation,
    check_no_orphans,
    check_single_case,
    check_traceability,
)
from core.graph.models import (
    Case,
    Evidence,
    Hypothesis,
    InputSignal,
    Investigation,
    NodeBase,
    Verdict,
)
from core.graph.store import EdgeSpec


async def _inject(raw, node: NodeBase, label: str) -> None:
    """Create a node WITHOUT edges via raw Cypher: an invalid (orphan) state that
    the store API refuses to produce."""
    await raw(f"CREATE (n:{label} $props)", props=node.model_dump(mode="json"))


async def _open_case(store) -> Case:
    """A legal root: InputSignal (born bare, exempt) + Case born with OPENS."""
    signal = InputSignal(raw_content="signal")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="objective", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    return case


async def _build_valid_subgraph(store) -> str:
    """Create a fully valid case subgraph (born connected) and return its case_id.

    InputSignal -OPENS-> Case -DERIVES-> Hypothesis, tested by two Investigations
    that PRODUCE an investigation Evidence (which SUPPORTS the hypothesis) and a
    validation Evidence (which VALIDATES the first); the Case CONCLUDES a Verdict.
    Satisfies every invariant.
    """
    case = await _open_case(store)
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    await store.create_node(hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])

    investigation = Investigation(description="investigation", case_id=case.id)
    await store.create_node(
        investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
    )
    verification = Investigation(description="verification", case_id=case.id)
    await store.create_node(
        verification, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
    )

    evidence = Evidence(content="evidence", case_id=case.id)
    await store.create_node(
        evidence,
        "Evidence",
        edges=[
            EdgeSpec("PRODUCES", investigation.id),
            EdgeSpec("SUPPORTS", hypothesis.id, direction="out"),
        ],
    )
    validation = Evidence(content="validation", case_id=case.id)
    await store.create_node(
        validation,
        "Evidence",
        edges=[
            EdgeSpec("PRODUCES", verification.id),
            EdgeSpec("VALIDATES", evidence.id, direction="out"),
        ],
    )

    verdict = Verdict(kind="confirmed", content="verdict", case_id=case.id)
    await store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])

    return case.id


# ---------- traceability ----------


@pytest.mark.integration
async def test_check_traceability_passes_with_valid_subgraph(store):
    case_id = await _build_valid_subgraph(store)
    assert await check_traceability(store, case_id) == []


@pytest.mark.integration
async def test_check_traceability_verdict_not_linked_to_its_case(store, raw):
    case = await _open_case(store)
    # a Verdict with no CONCLUDES edge: injected raw (the API refuses orphans)
    verdict = Verdict(kind="confirmed", content="verdict", case_id=case.id)
    await _inject(raw, verdict, "Verdict")

    violations = await check_traceability(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "traceability"
    assert verdict.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_traceability_case_not_linked_to_its_input_signal(store, raw):
    # a Case with no InputSignal opening it: injected raw
    case = Case(objective="objective", case_id="")
    await _inject(raw, case, "Case")

    violations = await check_traceability(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "traceability"
    assert case.id in violations[0].node_ids


# ---------- no_orphans ----------


@pytest.mark.integration
async def test_check_no_orphans_passes_with_valid_subgraph(store):
    case_id = await _build_valid_subgraph(store)
    assert await check_no_orphans(store, case_id) == []


@pytest.mark.integration
async def test_check_no_orphans_hypothesis_not_derived_by_the_case(store, raw):
    case = await _open_case(store)
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    await _inject(raw, hypothesis, "Hypothesis")  # no DERIVES edge from the Case

    violations = await check_no_orphans(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "no_orphans"
    assert hypothesis.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_no_orphans_investigation_without_parent(store, raw):
    case = await _open_case(store)
    investigation = Investigation(description="investigation", case_id=case.id)
    await _inject(raw, investigation, "Investigation")  # no TESTS nor REQUIRES edge

    violations = await check_no_orphans(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "no_orphans"
    assert investigation.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_no_orphans_evidence_not_produced_by_an_investigation(store, raw):
    case = await _open_case(store)
    evidence = Evidence(content="evidence", case_id=case.id)
    await _inject(raw, evidence, "Evidence")  # no PRODUCES edge into the Evidence

    violations = await check_no_orphans(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "no_orphans"
    assert evidence.id in violations[0].node_ids


# ---------- mandatory_validation ----------


@pytest.mark.integration
async def test_check_mandatory_validation_passes_with_valid_subgraph(store):
    case_id = await _build_valid_subgraph(store)
    assert await check_mandatory_validation(store, case_id) == []


@pytest.mark.integration
async def test_check_mandatory_validation_evidence_without_verifier(store):
    """This invalid state IS producible through the API on purpose: the guard only
    requires birth edges, not the full ontology; whether every supporting Evidence
    got validated is the checker's job."""
    case = await _open_case(store)
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    await store.create_node(hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    evidence = Evidence(content="evidence", case_id=case.id)
    await store.create_node(
        evidence, "Evidence", edges=[EdgeSpec("SUPPORTS", hypothesis.id, direction="out")]
    )
    # the Evidence backs a hypothesis but no Verifier VALIDATES it

    violations = await check_mandatory_validation(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "mandatory_validation"
    assert evidence.id in violations[0].node_ids


# ---------- single_case ----------


@pytest.mark.integration
async def test_check_single_case_passes_with_valid_subgraph(store):
    case_id = await _build_valid_subgraph(store)
    assert await check_single_case(store, case_id) == []


@pytest.mark.integration
async def test_check_single_case_multiple_cases(store, raw):
    case = await _open_case(store)
    # a second Case node sharing the same case_id (root of the same subgraph)
    duplicate = Case(objective="duplicate", case_id=case.id)
    await _inject(raw, duplicate, "Case")

    violations = await check_single_case(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "single_case"


# ---------- investigation_outcome ----------


@pytest.mark.integration
async def test_check_investigation_outcome_passes_with_valid_subgraph(store):
    case_id = await _build_valid_subgraph(store)
    assert await check_investigation_outcome(store, case_id) == []


@pytest.mark.integration
async def test_check_investigation_outcome_validated_investigation_no_evidence_produced(store):
    case = await _open_case(store)
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    await store.create_node(hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    investigation = Investigation(description="investigation", case_id=case.id, status="validated")
    await store.create_node(
        investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
    )
    # validated, but no PRODUCES edge to an Evidence

    violations = await check_investigation_outcome(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "investigation_outcome"
    assert investigation.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_investigation_outcome_skipped_investigation_without_explicit_reason(store):
    case = await _open_case(store)
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    await store.create_node(hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    investigation = Investigation(description="investigation", case_id=case.id, status="skipped")
    await store.create_node(
        investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
    )
    # skipped, but skip_reason is None

    violations = await check_investigation_outcome(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "investigation_outcome"
    assert investigation.id in violations[0].node_ids


# ---------- check_all ----------


@pytest.mark.integration
async def test_check_all_passes_with_valid_subgraph(store):
    case_id = await _build_valid_subgraph(store)
    assert await check_all(store, case_id) == []
