"""Unit tests for the base Role/Reaction/Agent runtime.

Run with:
    uv run pytest
"""
from typing import cast

import pytest

from core.agents.base import Agent, DeterministicRole, LLMRole, Reaction
from core.graph.models import NodeBase
from core.graph.store import GraphStore
from tests.mocks.mock_provider import MockProvider

# The store is unused by these tests (roles only keep the reference); a real
# GraphStore would open a Neo4j driver, so we pass a stand-in.
_STORE = cast(GraphStore, object())

class _FakeLLMRole(LLMRole):
    """One reaction whose execute records that it ran on the agent it was handed."""

    async def _claim(self) -> NodeBase | None:
        return None

    async def _run(self, agent: Agent) -> None:
        agent.messages.append({"executed_by": "llm"})

    def reactions(self) -> list[Reaction]:
        return [Reaction({("node_created", "InputSignal")}, self._claim, self._run)]

class _FakeDeterministicRole(DeterministicRole):
    async def _claim(self) -> NodeBase | None:
        return None

    async def _run(self, agent: Agent) -> None:
        agent.messages.append({"executed_by": "deterministic"})

    def reactions(self) -> list[Reaction]:
        return [Reaction({("node_updated", "Investigation")}, self._claim, self._run)]

@pytest.mark.unit
def test_llm_role_keeps_store_and_provider():
    """An LLMRole keeps its store and provider and has no per-work state."""
    provider = MockProvider([])
    role = _FakeLLMRole(_STORE, provider)

    assert role.store is _STORE
    assert role.provider is provider

@pytest.mark.unit
def test_deterministic_role_keeps_store_and_has_no_provider():
    """A DeterministicRole keeps its store and has no provider."""
    role = _FakeDeterministicRole(_STORE)

    assert role.store is _STORE
    assert not hasattr(role, "provider")

@pytest.mark.unit
def test_role_declares_its_reactions():
    """A role exposes its behaviors as Reactions binding triggers, claim and execute."""
    role = _FakeDeterministicRole(_STORE)

    reactions = role.reactions()

    assert len(reactions) == 1
    assert reactions[0].triggers == {("node_updated", "Investigation")}

@pytest.mark.unit
def test_agent_composes_role_and_starts_with_empty_stm():
    """An Agent composes the role (a reference, not a copy) and carries the work and
    an empty STM."""
    role = _FakeDeterministicRole(_STORE)
    work = cast(NodeBase, object())

    agent = Agent(role, role.reactions()[0].execute, work)

    assert agent.role is role  # composition: the same instance, not a clone
    assert agent.work is work
    assert agent.messages == []

@pytest.mark.unit
async def test_agent_run_delegates_to_reaction_execute():
    """Agent.run() delegates to the reaction's execute, passing the agent itself, so
    the logic operates on this agent's state (work + STM)."""
    role = _FakeLLMRole(_STORE, MockProvider([]))
    reaction = role.reactions()[0]
    agent = Agent(role, reaction.execute, work=cast(NodeBase, object()))

    await agent.run()

    assert agent.messages == [{"executed_by": "llm"}]

@pytest.mark.unit
async def test_role_on_failure_is_noop_by_default():
    """The base on_failure hook is a no-op: it must not raise."""
    role = _FakeDeterministicRole(_STORE)

    await role.on_failure(cast(NodeBase, object()))  # must not raise
