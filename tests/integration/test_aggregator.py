"""Integration test for the LLM Aggregator: it claims a quiescent case, reads the
surviving hypotheses and their evidence, crowns the winner, and closes the case.

    uv run pytest -m integration
"""

import json

import pytest

from core.agents.base import Agent
from core.graph.models import Case, Evidence, Hypothesis, InputSignal, Investigation
from core.graph.store import EdgeSpec
from core.providers.base import LLMResponse
from core.tools.base import ToolRegistry
from domain.roles.aggregator import Aggregator
from tests.mocks.mock_provider import MockProvider


@pytest.mark.integration
async def test_aggregator_crowns_the_winner_and_closes(store):
    """A quiescent case (a hypothesis tested, its supporting evidence produced and
    triaged): the Aggregator claims it, its prompt carries the surviving hypothesis and
    its evidence, it names the winner (which gets confirmed), writes the verdict, and
    closes the case."""
    signal = InputSignal(raw_content="suspicious login for jdoe")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="determine if the login is malicious", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hyp = Hypothesis(description="credential theft", case_id=case.id)
    await store.create_node(hyp, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    inv = Investigation(description="check auth logs", case_id=case.id)
    await store.create_node(inv, "Investigation", edges=[EdgeSpec("TESTS", hyp.id)])
    ev = Evidence(
        content="login from Belarus at 03:14; the user normally logs in from Argentina",
        case_id=case.id,
        triaged=True,  # quiescence requires every Evidence triaged
    )
    await store.create_node(
        ev, "Evidence",
        edges=[EdgeSpec("PRODUCES", inv.id), EdgeSpec("SUPPORTS", hyp.id, direction="out")],
    )

    verdict_out = json.dumps({
        "winner_id": hyp.id,
        "content": "credential theft confirmed: anomalous geo-login for jdoe",
        "rationale": "the Belarus login contradicts the user's normal Argentina location",
    })
    aggregator = Aggregator(store)
    reaction = aggregator.reactions()[0]
    work = await reaction.claim()  # claim_quiescent_case -> the quiescent case
    assert work is not None and work.id == case.id
    agent = Agent(
        aggregator, reaction.execute, work,
        provider=MockProvider([LLMResponse(content=verdict_out)]), tools=ToolRegistry([]),
    )
    await agent.run()

    # the surviving hypothesis and its evidence reached the prompt
    user_msg = next(m for m in agent.messages if m.get("role") == "user")
    assert "Belarus" in user_msg["content"]
    # the verdict was written: a winner was named, so the case resolved
    verdicts = await store.get_neighbors(case.id, "CONCLUDES", target_label="Verdict")
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.kind == "resolved"
    assert "credential theft" in verdict.content
    assert verdict.rationale != ""
    # the winner was crowned (confirmed), so the structure matches the verdict
    winner = await store.get_node(hyp.id)
    assert winner is not None and winner.status == "confirmed"
    closed = await store.get_node(case.id)
    assert closed is not None and closed.status == "closed"
