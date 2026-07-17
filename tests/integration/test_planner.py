"""Integration tests for the LLM Planner: it claims a Hypothesis and reasons a plan of
targeted investigations that TEST it.

    uv run pytest -m integration
"""

import json

import pytest

from core.agents.base import Agent
from core.graph.models import Case, Hypothesis, InputSignal
from core.graph.store import EdgeSpec
from core.providers.base import LLMResponse
from core.tools.base import ToolRegistry
from domain.roles.planner import Planner
from tests.mocks.mock_provider import MockProvider


async def _seed_hypothesis(store):
    signal = InputSignal(raw_content="suspicious login for jdoe")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="determine if the login is malicious", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hyp = Hypothesis(description="credential theft", case_id=case.id)
    await store.create_node(hyp, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    return case, hyp


@pytest.mark.integration
async def test_planner_reasons_multiple_targeted_investigations(store):
    """The Planner claims the Hypothesis, its prompt carries the objective + hypothesis,
    and it materializes the reasoned steps as Investigations that TEST it."""
    case, hyp = await _seed_hypothesis(store)
    plan_out = json.dumps({"steps": [
        {"description": "query auth logs for jdoe", "rationale": "confirm the anomalous login"},
        {"description": "check MFA enrollment events", "rationale": "theft often enrolls a device"},
    ]})
    planner = Planner(store)
    reaction = planner.reactions()[0]
    work = await reaction.claim()
    assert work is not None and work.id == hyp.id
    agent = Agent(
        planner, reaction.execute, work,
        provider=MockProvider([LLMResponse(content=plan_out)]), tools=ToolRegistry([]),
    )
    await agent.run()

    user_msg = next(m for m in agent.messages if m.get("role") == "user")
    assert "credential theft" in user_msg["content"]
    invs = await store.get_neighbors(hyp.id, "TESTS", target_label="Investigation")
    assert len(invs) == 2
    assert "query auth logs for jdoe" in {i.description for i in invs}
    assert all(i.rationale != "" for i in invs)


@pytest.mark.integration
async def test_planner_empty_plan_still_creates_one_investigation(store):
    """If the LLM returns no steps, the Planner still creates one investigation, so the
    hypothesis is never left un-tested (which would deadlock quiescence)."""
    case, hyp = await _seed_hypothesis(store)
    planner = Planner(store)
    reaction = planner.reactions()[0]
    work = await reaction.claim()
    assert work is not None
    agent = Agent(
        planner, reaction.execute, work,
        provider=MockProvider([LLMResponse(content=json.dumps({"steps": []}))]),
        tools=ToolRegistry([]),
    )
    await agent.run()

    invs = await store.get_neighbors(hyp.id, "TESTS", target_label="Investigation")
    assert len(invs) == 1
