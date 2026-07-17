"""End-to-end minimal flow: InputSignal -> Case -> Hypotheses -> Investigations ->
Evidence -> Verdict, driving the four REAL roles (all four agents on MockProviders) over
the runtime + Neo4j. Roles register engine-free; each one's provider is given at
registration and stamped on its agents at spawn.

    uv run pytest -m integration
"""

import asyncio
import json

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase

from core.providers.base import LLMResponse
from core.runtime.orchestrator import Orchestrator
from domain.roles.investigator import Investigator
from domain.roles.planner import Planner
from domain.roles.synthesizer import Synthesizer
from domain.roles.theorist import Theorist
from tests.mocks.mock_provider import MockProvider

_THEORIST_OUTPUT = json.dumps(
    {
        "objective": "determine whether the login is malicious",
        "rationale": "the signal describes an anomalous login",
        "hypotheses": [
            {"description": "credential theft", "rationale": "geo mismatch"},
            {"description": "legitimate travel", "rationale": "user may be abroad"},
        ],
    }
)

# the generative motor triages every Evidence: one call per evidence, judging nothing
_TRIAGE_NOTHING = json.dumps({"new_hypotheses": [], "refuted": []})

# the Investigator runs with no tool catalog here: an honest empty-handed finding,
# once per Investigation (2 hypotheses -> 2 investigations)
_NEUTRAL_FINDING = json.dumps(
    {
        "content": "no telemetry available for this step",
        "rationale": "no tools were available; nothing to examine",
        "stance": "neutral",
    }
)

# the Synthesizer weighs the (neutral) evidence and reasons the verdict; both
# investigations returned nothing, so 'unresolved' is the grounded call
_VERDICT_OUTPUT = json.dumps(
    {
        "kind": "unresolved",
        "content": "no telemetry was available to settle the case",
        "rationale": "both investigations returned neutral findings",
        "dispositions": [],  # an unresolved case confirms nothing
    }
)

# the Planner reasons the plan for each hypothesis; one step per hypothesis keeps the
# structural assertion (one investigation per hypothesis) intact
_PLAN_OUTPUT = json.dumps(
    {
        "steps": [
            {"description": "check jdoe's auth logs", "rationale": "test the hypothesis directly"}
        ]
    }
)


@pytest_asyncio.fixture
async def orchestrator(neo4j_container):
    uri = neo4j_container.get_connection_url()
    auth = (neo4j_container.username, neo4j_container.password)
    async with AsyncGraphDatabase.driver(uri, auth=auth) as admin, admin.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    orch = Orchestrator(uri, *auth)
    yield orch
    await orch.aclose()


async def _wait_until(store, label: str, count: int, timeout: float = 20.0):
    """Poll until at least `count` nodes of `label` exist (or time out). Returns as
    soon as the condition holds, so it is not a fixed sleep."""
    for _ in range(int(timeout / 0.1)):
        nodes = await store.query_nodes(label, {})
        if len(nodes) >= count:
            return nodes
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out waiting for {count} {label} node(s)")


@pytest.mark.integration
async def test_input_signal_reaches_a_verdict(orchestrator):
    """A submitted signal travels the whole chain and the case closes with a Verdict,
    with every structural link in place (2 hypotheses -> 2 investigations -> 2
    evidence -> 1 verdict)."""
    store = orchestrator.store
    # 3 responses: 1 opens the case, then one triage per evidence (2)
    orchestrator.register(
        Theorist(store),
        provider=MockProvider(
            [
                LLMResponse(content=_THEORIST_OUTPUT),
                LLMResponse(content=_TRIAGE_NOTHING),
                LLMResponse(content=_TRIAGE_NOTHING),
            ]
        ),
    )
    orchestrator.register(
        Planner(store),
        provider=MockProvider(
            [LLMResponse(content=_PLAN_OUTPUT), LLMResponse(content=_PLAN_OUTPUT)]
        ),
    )
    orchestrator.register(
        Investigator(store),
        provider=MockProvider(
            [
                LLMResponse(content=_NEUTRAL_FINDING),
                LLMResponse(content=_NEUTRAL_FINDING),
            ]
        ),
    )
    orchestrator.register(
        Synthesizer(store),
        provider=MockProvider([LLMResponse(content=_VERDICT_OUTPUT)]),
    )

    await orchestrator.start()
    await orchestrator.submit_signal("suspicious login from a new country")

    verdicts = await _wait_until(store, "Verdict", 1)

    # the full subgraph took shape
    cases = await store.query_nodes("Case", {})
    assert len(cases) == 1
    case = cases[0]

    hyps = await store.get_neighbors(case.id, "DERIVES", target_label="Hypothesis")
    assert len(hyps) == 2  # from the MockProvider output

    # every hypothesis was planned into an investigation, every investigation produced evidence
    for h in hyps:
        invs = await store.get_neighbors(h.id, "TESTS", target_label="Investigation")
        assert len(invs) == 1
        evs = await store.get_neighbors(invs[0].id, "PRODUCES", target_label="Evidence")
        assert len(evs) == 1

    # the case concluded with exactly one verdict
    assert len(verdicts) == 1
    concluded = await store.get_neighbors(case.id, "CONCLUDES", target_label="Verdict")
    assert [v.id for v in concluded] == [verdicts[0].id]
    assert getattr(verdicts[0], "case_id", None) == case.id

    # convergence guard: closure only fires once every Evidence was triaged by the
    # Theorist (the hypothesis space is stable)
    for evidence in await store.query_nodes("Evidence", {"case_id": case.id}):
        assert evidence.triaged is True
