"""Integration tests for the structural invariants (Neo4j docker instance).

For each invariant: one test on a fully valid subgraph (expects no violations) and
one test per violation it detects, on a minimal isolated subgraph. Run with:
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
    Verdict,
)


async def _build_valid_subgraph(store) -> str:
    """Create a fully valid case subgraph and return its case_id.

    InputSignal -OPENS-> Case -DERIVES-> Hypothesis, tested by two Investigations
    that PRODUCE an investigation Evidence (which SUPPORTS the hypothesis) and a
    validation Evidence (which VALIDATES the first); the Case CONCLUDES a Verdict.
    Satisfies every invariant.
    """
    signal = InputSignal(raw_content="signal")
    case = Case(objective="objective", case_id="")
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    investigation = Investigation(description="investigation", case_id=case.id)
    verification = Investigation(description="verification", case_id=case.id)
    evidence = Evidence(content="evidence", case_id=case.id)
    validation = Evidence(content="validation", case_id=case.id)
    verdict = Verdict(kind="confirmed", content="verdict", case_id=case.id)
    for node, label in [
        (signal, "InputSignal"),
        (case, "Case"),
        (hypothesis, "Hypothesis"),
        (investigation, "Investigation"),
        (verification, "Investigation"),
        (evidence, "Evidence"),
        (validation, "Evidence"),
        (verdict, "Verdict"),
    ]:
        await store.create_node(node, label=label)
    await store.create_edge(signal.id, case.id, "OPENS")
    await store.create_edge(case.id, hypothesis.id, "DERIVES")
    await store.create_edge(hypothesis.id, investigation.id, "TESTS")
    await store.create_edge(hypothesis.id, verification.id, "TESTS")
    await store.create_edge(investigation.id, evidence.id, "PRODUCES")
    await store.create_edge(verification.id, validation.id, "PRODUCES")
    await store.create_edge(evidence.id, hypothesis.id, "SUPPORTS")
    await store.create_edge(validation.id, evidence.id, "VALIDATES")
    await store.create_edge(case.id, verdict.id, "CONCLUDES")
    return case.id


# ---------- traceability ----------


@pytest.mark.integration
async def test_check_traceability_passes_with_valid_subgraph(store):
    case_id = await _build_valid_subgraph(store)
    assert await check_traceability(store, case_id) == []


@pytest.mark.integration
async def test_check_traceability_verdict_not_linked_to_its_case(store):
    signal = InputSignal(raw_content="signal")
    case = Case(objective="objective", case_id="")
    verdict = Verdict(kind="confirmed", content="verdict", case_id=case.id)
    await store.create_node(signal, label="InputSignal")
    await store.create_node(case, label="Case")
    await store.create_node(verdict, label="Verdict")
    await store.create_edge(signal.id, case.id, "OPENS")
    # no CONCLUDES edge from the Case to the Verdict

    violations = await check_traceability(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "traceability"
    assert verdict.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_traceability_case_not_linked_to_its_input_signal(store):
    case = Case(objective="objective", case_id="")
    await store.create_node(case, label="Case")
    # no InputSignal opening the Case

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
async def test_check_no_orphans_hypothesis_not_derived_by_the_case(store):
    case = Case(objective="objective", case_id="")
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    await store.create_node(case, label="Case")
    await store.create_node(hypothesis, label="Hypothesis")
    # no DERIVES edge from the Case

    violations = await check_no_orphans(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "no_orphans"
    assert hypothesis.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_no_orphans_investigation_without_parent(store):
    case = Case(objective="objective", case_id="")
    investigation = Investigation(description="investigation", case_id=case.id)
    await store.create_node(case, label="Case")
    await store.create_node(investigation, label="Investigation")
    # no TESTS nor REQUIRES edge into the Investigation

    violations = await check_no_orphans(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "no_orphans"
    assert investigation.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_no_orphans_evidence_not_produced_by_an_investigation(store):
    case = Case(objective="objective", case_id="")
    evidence = Evidence(content="evidence", case_id=case.id)
    await store.create_node(case, label="Case")
    await store.create_node(evidence, label="Evidence")
    # no PRODUCES edge into the Evidence

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
    case = Case(objective="objective", case_id="")
    hypothesis = Hypothesis(description="hypothesis", case_id=case.id)
    evidence = Evidence(content="evidence", case_id=case.id)
    await store.create_node(case, label="Case")
    await store.create_node(hypothesis, label="Hypothesis")
    await store.create_node(evidence, label="Evidence")
    await store.create_edge(evidence.id, hypothesis.id, "SUPPORTS")
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
async def test_check_single_case_multiple_cases(store):
    case = Case(objective="objective", case_id="")
    await store.create_node(case, label="Case")
    # a second Case node sharing the same case_id (root of the same subgraph)
    duplicate = Case(objective="duplicate", case_id=case.id)
    await store.create_node(duplicate, label="Case")

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
    case = Case(objective="objective", case_id="")
    investigation = Investigation(description="investigation", case_id=case.id, status="validated")
    await store.create_node(case, label="Case")
    await store.create_node(investigation, label="Investigation")
    # validated, but no PRODUCES edge to an Evidence

    violations = await check_investigation_outcome(store, case.id)

    assert len(violations) == 1
    assert violations[0].invariant == "investigation_outcome"
    assert investigation.id in violations[0].node_ids


@pytest.mark.integration
async def test_check_investigation_outcome_skipped_investigation_without_explicit_reason(store):
    case = Case(objective="objective", case_id="")
    investigation = Investigation(description="investigation", case_id=case.id, status="skipped")
    await store.create_node(case, label="Case")
    await store.create_node(investigation, label="Investigation")
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
