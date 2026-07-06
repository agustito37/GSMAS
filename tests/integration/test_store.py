"""Integration test for the GraphStore (Neo4j docker instance).

Runs node + edge + typed traversal round-trips under the "born connected" rule:
case-scoped nodes are created WITH their birth edges, atomically. Run with:
    uv run pytest -m integration
"""

import pytest

from core.graph.models import Case, Evidence, Hypothesis, InputSignal, Investigation
from core.graph.store import EdgeSpec


async def _open_case(store) -> Case:
    """The minimal legal root: an InputSignal (born bare, exempt) and a Case born
    connected to it via OPENS."""
    signal = InputSignal(raw_content="an input signal")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    return case


async def _derive_hypothesis(store, case: Case) -> Hypothesis:
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    await store.create_node(hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    return hypothesis


async def _plan_investigation(store, hypothesis: Hypothesis) -> Investigation:
    investigation = Investigation(description="an investigation", case_id=hypothesis.case_id)
    await store.create_node(
        investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
    )
    return investigation


# ---------- atomic birth (create_node) ----------


@pytest.mark.integration
async def test_create_node_born_connected(store):
    """A case-scoped node is created WITH its birth edge in one operation: the node
    is retrievable and the edge already exists."""
    signal = InputSignal(raw_content="an input signal")
    await store.create_node(signal, "InputSignal")

    case = Case(objective="demo objective", case_id="")
    node_id = await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])

    assert node_id == case.id
    assert await store.get_node(node_id) is not None
    opened = await store.get_neighbors(signal.id, "OPENS", direction="out", target_label="Case")
    assert [n.id for n in opened] == [case.id]


@pytest.mark.integration
async def test_create_node_case_scoped_without_edges_raises(store):
    """The framework restriction: a case-scoped node cannot be born an orphan (the
    no-orphans invariant holds by construction)."""
    with pytest.raises(ValueError, match="born connected"):
        await store.create_node(Case(objective="demo objective", case_id=""), "Case")

    assert await store.query_nodes("Case", {}) == []


@pytest.mark.integration
async def test_create_node_with_missing_endpoint_raises_and_creates_nothing(store):
    """Atomicity: if a birth-edge endpoint does not exist, NOTHING is created (the
    statement matches endpoints before creating)."""
    case = await _open_case(store)
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)

    with pytest.raises(ValueError, match="endpoint"):
        await store.create_node(
            hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", "unknown-id")]
        )

    assert await store.query_nodes("Hypothesis", {}) == []


@pytest.mark.integration
async def test_create_node_with_multiple_birth_edges(store):
    """A node can be born with several edges at once, mixing directions: Evidence
    born produced by its Investigation (in) and supporting a Hypothesis (out)."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    investigation = await _plan_investigation(store, hypothesis)

    evidence = Evidence(content="a supporting evidence", case_id=case.id)
    await store.create_node(
        evidence,
        "Evidence",
        edges=[
            EdgeSpec("PRODUCES", investigation.id),  # in: investigation -> evidence
            EdgeSpec("SUPPORTS", hypothesis.id, direction="out"),  # out: evidence -> hypothesis
        ],
    )

    produced = await store.get_evidence_of_investigation(investigation.id)
    supporting = await store.get_supporting_evidence(hypothesis.id)
    assert [e.id for e in produced] == [evidence.id]
    assert [e.id for e in supporting] == [evidence.id]


@pytest.mark.integration
async def test_get_node_fails_with_unknown_node_id(store):
    """Getting a node with an unknown ID returns None."""
    assert await store.get_node("unknown") is None


@pytest.mark.integration
async def test_update_node(store):
    """Updating a node changes its properties."""
    case = await _open_case(store)
    await store.update_node(case.id, {"objective": "updated objective"})
    node = await store.get_node(case.id)
    assert node is not None
    assert node.objective == "updated objective"


@pytest.mark.integration
async def test_create_edge_links_existing_nodes_directionally(store):
    """create_edge is for linking two EXISTING nodes (e.g. a second InputSignal
    opening an existing Case). The edge is *directed*: observable from the source,
    not from the target."""
    case = await _open_case(store)
    second_signal = InputSignal(raw_content="another signal for the same case")
    await store.create_node(second_signal, "InputSignal")

    await store.create_edge(second_signal.id, case.id, "OPENS")

    forward = await store.get_neighbors(
        second_signal.id, "OPENS", direction="out", target_label="Case"
    )
    backward = await store.get_neighbors(
        case.id, "OPENS", direction="out", target_label="InputSignal"
    )
    assert [n.id for n in forward] == [case.id]
    assert backward == []


# ---------- generic queries ----------


@pytest.mark.integration
async def test_query_nodes(store):
    """Querying nodes by properties returns the matching nodes."""
    case = await _open_case(store)
    nodes = await store.query_nodes(label="Case", filters={"objective": "demo objective"})
    assert len(nodes) == 1
    assert nodes[0].id == case.id
    assert nodes[0].objective == "demo objective"


@pytest.mark.integration
async def test_query_nodes_fails_with_unknown_label(store):
    """Querying nodes by an unknown label returns an empty list."""
    assert await store.query_nodes(label="Unknown", filters={}) == []


@pytest.mark.integration
async def test_query_nodes_fails_with_unknown_property(store):
    """Querying nodes by an unknown property returns an empty list."""
    await _open_case(store)
    assert await store.query_nodes(label="Case", filters={"unknown": "demo objective"}) == []


@pytest.mark.integration
async def test_get_neighbors(store):
    """Getting neighbors of a node returns the connected nodes."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    neighbors = await store.get_neighbors(
        case.id, "DERIVES", direction="out", target_label="Hypothesis"
    )
    assert len(neighbors) == 1
    assert neighbors[0].id == hypothesis.id
    assert neighbors[0].description == "a candidate explanation"


@pytest.mark.integration
async def test_get_neighbors_fails_with_unknown_edge_type(store):
    """Getting neighbors of a node with an unknown edge type returns an empty list."""
    case = await _open_case(store)
    neighbors = await store.get_neighbors(
        case.id, "Unknown", direction="out", target_label="Hypothesis"
    )
    assert neighbors == []


@pytest.mark.integration
async def test_get_neighbors_fails_with_unknown_target_label(store):
    """Getting neighbors of a node with an unknown target label returns an empty list."""
    case = await _open_case(store)
    neighbors = await store.get_neighbors(
        case.id, "DERIVES", direction="out", target_label="Unknown"
    )
    assert neighbors == []


# ---------- layer 2: domain queries ----------


@pytest.mark.integration
async def test_get_refuting_evidence(store):
    """Refuting evidence is the Evidence that CONTRADICTS the hypothesis."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    evidence = Evidence(content="a refuting evidence", case_id=case.id)
    # per the ontology the edge goes Evidence -> Hypothesis (out from the new node)
    await store.create_node(
        evidence, "Evidence", edges=[EdgeSpec("CONTRADICTS", hypothesis.id, direction="out")]
    )

    refuting = await store.get_refuting_evidence(hypothesis.id)

    assert len(refuting) == 1
    assert refuting[0].id == evidence.id


@pytest.mark.integration
async def test_get_refuting_evidence_fails_with_unknown_hypothesis_id(store):
    """Getting refuting evidence for a hypothesis with an unknown ID returns an empty list."""
    assert await store.get_refuting_evidence("unknown") == []


@pytest.mark.integration
async def test_get_supporting_evidence(store):
    """Supporting evidence is the Evidence that SUPPORTS the hypothesis."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    evidence = Evidence(content="a supporting evidence", case_id=case.id)
    await store.create_node(
        evidence, "Evidence", edges=[EdgeSpec("SUPPORTS", hypothesis.id, direction="out")]
    )

    supporting = await store.get_supporting_evidence(hypothesis.id)

    assert len(supporting) == 1
    assert supporting[0].id == evidence.id


@pytest.mark.integration
async def test_get_supporting_evidence_fails_with_unknown_hypothesis_id(store):
    """Getting supporting evidence for a hypothesis with an unknown ID returns an empty list."""
    assert await store.get_supporting_evidence("unknown") == []


@pytest.mark.integration
async def test_get_investigations_of_hypothesis(store):
    """Getting investigations of a hypothesis returns the investigations."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    investigation = await _plan_investigation(store, hypothesis)
    investigations = await store.get_investigations_of_hypothesis(hypothesis.id)
    assert len(investigations) == 1
    assert investigations[0].id == investigation.id
    assert investigations[0].description == "an investigation"


@pytest.mark.integration
async def test_get_investigations_of_hypothesis_fails_with_unknown_hypothesis_id(store):
    """Getting investigations of a hypothesis with an unknown ID returns an empty list."""
    assert await store.get_investigations_of_hypothesis("unknown") == []


@pytest.mark.integration
async def test_get_evidence_of_investigation(store):
    """Getting evidence of an investigation returns the evidence."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    investigation = await _plan_investigation(store, hypothesis)
    evidence = Evidence(content="an evidence", case_id=case.id)
    await store.create_node(evidence, "Evidence", edges=[EdgeSpec("PRODUCES", investigation.id)])

    produced = await store.get_evidence_of_investigation(investigation.id)

    assert len(produced) == 1
    assert produced[0].id == evidence.id
    assert produced[0].content == "an evidence"


@pytest.mark.integration
async def test_get_evidence_of_investigation_fails_with_unknown_investigation_id(store):
    """Getting evidence of an investigation with an unknown ID returns an empty list."""
    assert await store.get_evidence_of_investigation("unknown") == []


@pytest.mark.integration
async def test_get_pending_investigations(store):
    """Getting pending investigations for a case returns the not-yet-claimed ones."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    investigation = await _plan_investigation(store, hypothesis)
    investigations = await store.get_pending_investigations(case.id)
    assert len(investigations) == 1
    assert investigations[0].id == investigation.id
    assert investigations[0].description == "an investigation"


@pytest.mark.integration
async def test_get_pending_investigations_fails_with_unknown_case_id(store):
    """Getting pending investigations for a case with an unknown ID returns an empty list."""
    assert await store.get_pending_investigations("unknown") == []


@pytest.mark.integration
async def test_get_active_hypotheses(store):
    """Getting active hypotheses for a case returns the hypotheses."""
    case = await _open_case(store)
    hypothesis = await _derive_hypothesis(store, case)
    hypotheses = await store.get_active_hypotheses(case.id)
    assert len(hypotheses) == 1
    assert hypotheses[0].id == hypothesis.id
    assert hypotheses[0].description == "a candidate explanation"


@pytest.mark.integration
async def test_get_active_hypotheses_fails_with_unknown_case_id(store):
    """Getting active hypotheses for a case with an unknown ID returns an empty list."""
    assert await store.get_active_hypotheses("unknown") == []


# ---------- claim lifecycle recovery ----------


@pytest.mark.integration
async def test_recover_claimed_returns_to_pending(store):
    """A node left 'claimed' (orphan from a dead process) is reset to 'pending', its
    holder cleared, and its attempts incremented."""
    sig = InputSignal(raw_content="x", claim_state="claimed", claimed_by_agent_id="dead")
    await store.create_node(sig, "InputSignal")

    recovered = await store.recover_claimed(max_attempts=3)

    assert recovered == 1
    node = await store.get_node(sig.id)
    assert node is not None
    assert node.claim_state == "pending"
    assert node.claimed_by_agent_id is None
    assert node.attempts == 1


@pytest.mark.integration
async def test_recover_claimed_marks_failed_after_max_attempts(store):
    """A claimed node that reaches max_attempts on recovery goes to 'failed', not back
    to 'pending', so a unit that keeps crashing the process stops looping."""
    sig = InputSignal(raw_content="x", claim_state="claimed", attempts=2)
    await store.create_node(sig, "InputSignal")

    await store.recover_claimed(max_attempts=3)  # 2 + 1 = 3 >= 3 -> failed

    node = await store.get_node(sig.id)
    assert node is not None
    assert node.claim_state == "failed"
    assert node.attempts == 3


@pytest.mark.integration
async def test_recover_claimed_ignores_non_claimed(store):
    """Recovery only touches 'claimed' nodes; a 'pending' one is left untouched."""
    sig = InputSignal(raw_content="p")  # default claim_state='pending'
    await store.create_node(sig, "InputSignal")

    recovered = await store.recover_claimed(max_attempts=3)

    assert recovered == 0
    node = await store.get_node(sig.id)
    assert node is not None
    assert node.claim_state == "pending"
    assert node.attempts == 0


# ---------- visualization ----------


@pytest.mark.integration
async def test_get_case_subgraph(store):
    """get_case_subgraph returns the Case, its InputSignals, and its case-scoped
    nodes. Neo4j does not guarantee collection order, so compare by id sets."""
    signal = InputSignal(raw_content="an input signal")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hypothesis = await _derive_hypothesis(store, case)
    investigation = await _plan_investigation(store, hypothesis)
    evidence = Evidence(content="an evidence", case_id=case.id)
    await store.create_node(evidence, "Evidence", edges=[EdgeSpec("PRODUCES", investigation.id)])

    subgraph = await store.get_case_subgraph(case.id)

    assert subgraph["case"]["id"] == case.id
    assert {s["id"] for s in subgraph["signals"]} == {signal.id}
    # InputSignal carries no case_id, so it is in "signals", not "nodes".
    assert {n["id"] for n in subgraph["nodes"]} == {
        case.id,
        hypothesis.id,
        investigation.id,
        evidence.id,
    }


@pytest.mark.integration
async def test_get_case_subgraph_with_unknown_case_id_returns_empty(store):
    """An unknown case_id yields an empty result rather than raising."""
    assert await store.get_case_subgraph("unknown") == {}
