"""Integration test for the GraphStore (Neo4j docker instance).

Runs a node + edge + typed traversal round-trip. Run with:
    uv run pytest -m integration
"""

import pytest

from core.graph.models import Case, Evidence, Hypothesis, InputSignal, Investigation


@pytest.mark.integration
async def test_get_create_node(store):
    """Creating a node returns its ID and can be retrieved by ID."""
    case = Case(objective="demo objective", case_id="")
    node_id = await store.create_node(case, label="Case")
    assert node_id is not None
    node = await store.get_node(node_id)
    assert node is not None


@pytest.mark.integration
async def test_get_node_fails_with_unknown_node_id(store):
    """Getting a node with an unknown ID returns None."""
    node = await store.get_node("unknown")
    assert node is None


@pytest.mark.integration
async def test_update_node(store):
    """Updating a node changes its properties."""
    case = Case(objective="demo objective", case_id="")
    node_id = await store.create_node(case, label="Case")
    assert node_id is not None
    await store.update_node(node_id, {"objective": "updated objective"})
    node = await store.get_node(node_id)
    assert node is not None
    assert node.objective == "updated objective"


@pytest.mark.integration
async def test_create_edge_is_directional(store):
    """create_edge creates a *directed* edge: it is observable following the edge
    outward from the source, but not from the target. Edges are connections, not
    entities, so the store has no get_edge — they are observed via get_neighbors."""
    case = Case(objective="demo objective", case_id="")
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    await store.create_node(case, label="Case")
    await store.create_node(hypothesis, label="Hypothesis")
    await store.create_edge(case.id, hypothesis.id, "DERIVES")

    forward = await store.get_neighbors(
        case.id, "DERIVES", direction="out", target_label="Hypothesis"
    )
    backward = await store.get_neighbors(
        hypothesis.id, "DERIVES", direction="out", target_label="Case"
    )

    assert [n.id for n in forward] == [hypothesis.id]
    assert backward == []


@pytest.mark.integration
async def test_query_nodes(store):
    """Querying nodes by properties returns the matching nodes."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    nodes = await store.query_nodes(label="Case", filters={"objective": "demo objective"})
    assert len(nodes) == 1
    assert nodes[0].id == case.id
    assert nodes[0].objective == "demo objective"


@pytest.mark.integration
async def test_query_nodes_fails_with_unknown_label(store):
    """Querying nodes by an unknown label returns an empty list."""
    nodes = await store.query_nodes(label="Unknown", filters={})
    assert len(nodes) == 0


@pytest.mark.integration
async def test_query_nodes_fails_with_unknown_property(store):
    """Querying nodes by an unknown property returns an empty list."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    nodes = await store.query_nodes(label="Case", filters={"unknown": "demo objective"})
    assert len(nodes) == 0


@pytest.mark.integration
async def test_get_neighbors(store):
    """Getting neighbors of a node returns the connected nodes."""
    case = Case(objective="demo objective", case_id="")
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    await store.create_node(case, label="Case")
    await store.create_node(hypothesis, label="Hypothesis")
    await store.create_edge(case.id, hypothesis.id, "DERIVES")
    neighbors = await store.get_neighbors(
        case.id, "DERIVES", direction="out", target_label="Hypothesis"
    )
    assert len(neighbors) == 1
    assert neighbors[0].id == hypothesis.id
    assert neighbors[0].description == "a candidate explanation"


@pytest.mark.integration
async def test_get_neighbors_fails_with_unknown_edge_type(store):
    """Getting neighbors of a node with an unknown edge type returns an empty list."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    neighbors = await store.get_neighbors(
        case.id, "Unknown", direction="out", target_label="Hypothesis"
    )
    assert len(neighbors) == 0


@pytest.mark.integration
async def test_get_neighbors_fails_with_unknown_target_label(store):
    """Getting neighbors of a node with an unknown target label returns an empty list."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    neighbors = await store.get_neighbors(
        case.id, "DERIVES", direction="out", target_label="Unknown"
    )
    assert len(neighbors) == 0


@pytest.mark.integration
async def test_get_refuting_evidence(store):
    """Refuting evidence is the Evidence that CONTRADICTS the hypothesis."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    await store.create_node(hypothesis, label="Hypothesis")
    evidence = Evidence(content="a refuting evidence", case_id=case.id)
    await store.create_node(evidence, label="Evidence")
    # Per the ontology the edge goes Evidence -> Hypothesis.
    await store.create_edge(evidence.id, hypothesis.id, "CONTRADICTS")

    refuting = await store.get_refuting_evidence(hypothesis.id)

    assert len(refuting) == 1
    assert refuting[0].id == evidence.id


@pytest.mark.integration
async def test_get_refuting_evidence_fails_with_unknown_hypothesis_id(store):
    """Getting refuting evidence for a hypothesis with an unknown ID returns an empty list."""
    evidence = await store.get_refuting_evidence("unknown")
    assert len(evidence) == 0


@pytest.mark.integration
async def test_get_supporting_evidence(store):
    """Supporting evidence is the Evidence that SUPPORTS the hypothesis."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    await store.create_node(hypothesis, label="Hypothesis")
    evidence = Evidence(content="a supporting evidence", case_id=case.id)
    await store.create_node(evidence, label="Evidence")
    # Per the ontology the edge goes Evidence -> Hypothesis.
    await store.create_edge(evidence.id, hypothesis.id, "SUPPORTS")

    supporting = await store.get_supporting_evidence(hypothesis.id)

    assert len(supporting) == 1
    assert supporting[0].id == evidence.id


@pytest.mark.integration
async def test_get_supporting_evidence_fails_with_unknown_hypothesis_id(store):
    """Getting supporting evidence for a hypothesis with an unknown ID returns an empty list."""
    evidence = await store.get_supporting_evidence("unknown")
    assert len(evidence) == 0


@pytest.mark.integration
async def test_get_investigations_of_hypothesis(store):
    """Getting investigations of a hypothesis returns the investigations."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    await store.create_node(hypothesis, label="Hypothesis")
    investigation = Investigation(description="an investigation", case_id=case.id)
    await store.create_node(investigation, label="Investigation")
    await store.create_edge(hypothesis.id, investigation.id, "TESTS")
    investigations = await store.get_investigations_of_hypothesis(hypothesis.id)
    assert len(investigations) == 1
    assert investigations[0].id == investigation.id
    assert investigations[0].description == "an investigation"


@pytest.mark.integration
async def test_get_investigations_of_hypothesis_fails_with_unknown_hypothesis_id(store):
    """Getting investigations of a hypothesis with an unknown ID returns an empty list."""
    investigations = await store.get_investigations_of_hypothesis("unknown")
    assert len(investigations) == 0


@pytest.mark.integration
async def test_get_evidence_of_investigation(store):
    """Getting evidence of an investigation returns the evidence."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    investigation = Investigation(description="an investigation", case_id=case.id)
    await store.create_node(investigation, label="Investigation")
    evidence = Evidence(content="an evidence", case_id=case.id)
    await store.create_node(evidence, label="Evidence")
    await store.create_edge(investigation.id, evidence.id, "PRODUCES")

    produced = await store.get_evidence_of_investigation(investigation.id)

    assert len(produced) == 1
    assert produced[0].id == evidence.id
    assert produced[0].content == "an evidence"


@pytest.mark.integration
async def test_get_evidence_of_investigation_fails_with_unknown_investigation_id(store):
    """Getting evidence of an investigation with an unknown ID returns an empty list."""
    evidence = await store.get_evidence_of_investigation("unknown")
    assert len(evidence) == 0


@pytest.mark.integration
async def test_get_pending_investigations(store):
    """Getting pending investigations for a case returns the investigations."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    investigation = Investigation(description="an investigation", case_id=case.id)
    await store.create_node(investigation, label="Investigation")
    await store.create_edge(case.id, investigation.id, "OPENS")
    investigations = await store.get_pending_investigations(case.id)
    assert len(investigations) == 1
    assert investigations[0].id == investigation.id
    assert investigations[0].description == "an investigation"


@pytest.mark.integration
async def test_get_pending_investigations_fails_with_unknown_case_id(store):
    """Getting pending investigations for a case with an unknown ID returns an empty list."""
    investigations = await store.get_pending_investigations("unknown")
    assert len(investigations) == 0


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


@pytest.mark.integration
async def test_get_active_hypotheses(store):
    """Getting active hypotheses for a case returns the hypotheses."""
    case = Case(objective="demo objective", case_id="")
    await store.create_node(case, label="Case")
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    await store.create_node(hypothesis, label="Hypothesis")
    await store.create_edge(case.id, hypothesis.id, "DERIVES")
    hypotheses = await store.get_active_hypotheses(case.id)
    assert len(hypotheses) == 1
    assert hypotheses[0].id == hypothesis.id
    assert hypotheses[0].description == "a candidate explanation"


@pytest.mark.integration
async def test_get_active_hypotheses_fails_with_unknown_case_id(store):
    """Getting active hypotheses for a case with an unknown ID returns an empty list."""
    hypotheses = await store.get_active_hypotheses("unknown")
    assert len(hypotheses) == 0


@pytest.mark.integration
async def test_get_case_subgraph(store):
    """get_case_subgraph returns the Case, its InputSignals, and its case-scoped
    nodes. Neo4j does not guarantee collection order, so compare by id sets."""
    case = Case(objective="demo objective", case_id="")
    signal = InputSignal(raw_content="an input signal")
    hypothesis = Hypothesis(description="a candidate explanation", case_id=case.id)
    investigation = Investigation(description="an investigation", case_id=case.id)
    evidence = Evidence(content="an evidence", case_id=case.id)
    for node, label in [
        (case, "Case"),
        (signal, "InputSignal"),
        (hypothesis, "Hypothesis"),
        (investigation, "Investigation"),
        (evidence, "Evidence"),
    ]:
        await store.create_node(node, label=label)
    await store.create_edge(signal.id, case.id, "OPENS")
    await store.create_edge(case.id, hypothesis.id, "DERIVES")
    await store.create_edge(hypothesis.id, investigation.id, "TESTS")
    await store.create_edge(investigation.id, evidence.id, "PRODUCES")

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
    subgraph = await store.get_case_subgraph("unknown")
    assert subgraph == {}
