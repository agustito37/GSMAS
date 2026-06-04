"""Unit tests for the base agents.

Run with:
    uv run pytest
"""

from typing import cast

import pytest

from core.agents.base import DeterministicAgent, LLMAgent
from core.events.bus import Event, EventBus, Handler
from core.graph.models import Role
from core.graph.store import GraphStore
from tests.mocks.mock_provider import MockProvider


def _role() -> Role:
    return Role(name="theorist", kind="domain", agent_type="llm")


# The store is unused by these tests (the agents only keep the reference); a real
# GraphStore would open a Neo4j driver, so we pass a stand-in.
_STORE = cast(GraphStore, object())


class _FakeLLMAgent(LLMAgent):
    def subscriptions(self) -> dict[tuple[str, str | None], Handler]:
        return {}


class _FakeDeterministicAgent(DeterministicAgent):
    def subscriptions(self) -> dict[tuple[str, str | None], Handler]:
        return {}


@pytest.mark.unit
def test_llm_agent_init():
    """An LLMAgent keeps its id/role/bus/provider and starts with an empty STM."""
    role = _role()
    bus = EventBus()
    provider = MockProvider([])

    agent = _FakeLLMAgent("a1", role, _STORE, bus, provider)

    assert agent.id == "a1"
    assert agent.role is role
    assert agent.bus is bus
    assert agent.provider is provider
    assert agent._messages == []


@pytest.mark.unit
def test_deterministic_agent_init():
    """A DeterministicAgent keeps its id/role/bus and has no provider."""
    role = _role()
    bus = EventBus()

    agent = _FakeDeterministicAgent("a2", role, _STORE, bus)

    assert agent.id == "a2"
    assert agent.role is role
    assert agent.bus is bus
    assert not hasattr(agent, "provider")


@pytest.mark.unit
async def test_start_registers_subscriptions_on_the_bus():
    """start() wires each subscription to the bus, so a matching event reaches the
    right method and a non-matching node_type is ignored."""
    bus = EventBus()
    seen: list[Event] = []

    class _Agent(DeterministicAgent):
        def subscriptions(self) -> dict[tuple[str, str | None], Handler]:
            return {("node_created", "Case"): self.on_case}

        async def on_case(self, event: Event) -> None:
            seen.append(event)

    agent = _Agent("a3", _role(), _STORE, bus)
    agent.start()

    bus.publish(Event(type="node_created", node_id="c1", node_type="Case"))
    bus.publish(Event(type="node_created", node_id="e1", node_type="Evidence"))  # ignored
    await bus.aclose()

    assert len(seen) == 1  # only the Case event matched the subscription
    assert seen[0].node_id == "c1"
