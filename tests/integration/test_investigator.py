"""Integration tests for the Investigator: the generic domain role that executes
Investigations with the common tool catalog (Neo4j + MockProvider, no tokens).

    uv run pytest -m integration
"""

import json

import pytest

from core.agents.base import Agent
from core.graph.models import Case, Hypothesis, InputSignal, Investigation
from core.graph.store import EdgeSpec
from core.providers.base import LLMResponse, ToolCall
from core.tools.base import ToolRegistry
from domain.roles.investigator import Investigator
from domain.tools.log_query import LogQueryTool
from tests.mocks.mock_provider import MockProvider


async def _seed_investigation(store):
    signal = InputSignal(raw_content="suspicious login for user jdoe")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="determine if the login is malicious", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hypothesis = Hypothesis(description="credential theft", case_id=case.id)
    await store.create_node(hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    investigation = Investigation(description="check jdoe's auth logs", case_id=case.id)
    await store.create_node(
        investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
    )
    return case, hypothesis, investigation


@pytest.mark.integration
async def test_investigator_searches_logs_and_produces_judged_evidence(store):
    """The Investigator claims the Investigation, its agent drives the tool loop (a
    log_query call against the real telemetry file), and produces Evidence born with
    PRODUCES plus the SUPPORTS/CONTRADICTS edge of its own judgment (stance),
    carrying the finding and its rationale."""
    case, hypothesis, investigation = await _seed_investigation(store)
    provider = MockProvider([
        LLMResponse(content="", tool_calls=[
            ToolCall(id="c1", name="log_query", arguments={"query": "jdoe"})
        ]),
        LLMResponse(content=json.dumps({
            "content": "auth logs show a login for jdoe from Belarus at 03:14 UTC",
            "rationale": "the telemetry contains the anomalous login entry",
            "stance": "supports",
        })),
    ])
    tools = ToolRegistry([LogQueryTool("data/telemetry.jsonl")])
    investigator = Investigator(store)

    work = await investigator.reactions()[0].claim()
    assert work is not None and work.id == investigation.id
    agent = Agent(
        investigator, investigator.reactions()[0].execute, work, provider=provider, tools=tools
    )
    await agent.run()

    # the tool result actually entered the STM (the loop ran against real telemetry)
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "jdoe" in tool_messages[0]["content"]

    # the Evidence was born connected: PRODUCES from the investigation and the
    # stance edge (SUPPORTS) toward the hypothesis under test
    produced = await store.get_evidence_of_investigation(investigation.id)
    assert len(produced) == 1
    evidence = produced[0]
    assert "belarus" in evidence.content.lower()
    assert evidence.rationale != ""
    supporting = await store.get_supporting_evidence(hypothesis.id)
    assert [e.id for e in supporting] == [evidence.id]


@pytest.mark.integration
async def test_investigator_neutral_finding_creates_no_stance_edge(store):
    """A neutral finding produces Evidence linked only by PRODUCES: no SUPPORTS nor
    CONTRADICTS edge is fabricated."""
    case, hypothesis, investigation = await _seed_investigation(store)
    provider = MockProvider([
        LLMResponse(content=json.dumps({
            "content": "no relevant entries found",
            "rationale": "the logs show nothing about this hypothesis",
            "stance": "neutral",
        })),
    ])
    investigator = Investigator(store)

    work = await investigator.reactions()[0].claim()
    assert work is not None
    agent = Agent(
        investigator,
        investigator.reactions()[0].execute,
        work,
        provider=provider,
        tools=ToolRegistry([]),
    )
    await agent.run()

    produced = await store.get_evidence_of_investigation(investigation.id)
    assert len(produced) == 1
    assert await store.get_supporting_evidence(hypothesis.id) == []
    assert await store.get_refuting_evidence(hypothesis.id) == []
