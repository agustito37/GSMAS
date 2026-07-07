"""Unit tests for the base Role/Reaction/Agent runtime and the tool loop.

Run with:
    uv run pytest
"""

import json
from typing import cast

import pytest

from core.agents.base import Agent, Reaction, Role
from core.agents.tool_loop import MAX_TOOL_ITERATIONS, run_tool_loop
from core.graph.models import NodeBase
from core.graph.store import GraphStore
from core.providers.base import LLMResponse, ToolCall
from core.tools.base import Tool, ToolRegistry
from tests.mocks.mock_provider import MockProvider

# The store is unused by these tests (roles only keep the reference); a real
# GraphStore would open a Neo4j driver, so we pass a stand-in.
_STORE = cast(GraphStore, object())


class _LLMBackedRole(Role):
    """A role whose judgment is LLM-backed: it declares its own provider dependency
    (there is no LLMRole class; the framework does not classify roles by substrate)."""

    def __init__(self, store, provider) -> None:
        super().__init__(store)
        self.provider = provider

    async def _claim(self) -> NodeBase | None:
        return None

    async def _run(self, agent: Agent) -> None:
        agent.messages.append({"executed_by": "llm"})

    def reactions(self) -> list[Reaction]:
        return [Reaction({("node_created", "InputSignal")}, self._claim, self._run)]


class _RuleBackedRole(Role):
    """A role whose judgment is pure rules: store only, no provider."""

    async def _claim(self) -> NodeBase | None:
        return None

    async def _run(self, agent: Agent) -> None:
        agent.messages.append({"executed_by": "rules"})

    def reactions(self) -> list[Reaction]:
        return [Reaction({("node_updated", "Investigation")}, self._claim, self._run)]


@pytest.mark.unit
def test_roles_declare_their_own_dependencies():
    """One Role base: an LLM-backed role carries the provider it declared; a
    rule-backed one carries only the store. No substrate hierarchy."""
    provider = MockProvider([])
    llm_backed = _LLMBackedRole(_STORE, provider)
    rule_backed = _RuleBackedRole(_STORE)

    assert llm_backed.store is _STORE
    assert llm_backed.provider is provider
    assert rule_backed.store is _STORE
    assert not hasattr(rule_backed, "provider")


@pytest.mark.unit
def test_role_declares_its_reactions():
    """A role exposes its behaviors as Reactions binding triggers, claim and execute."""
    role = _RuleBackedRole(_STORE)

    reactions = role.reactions()

    assert len(reactions) == 1
    assert reactions[0].triggers == {("node_updated", "Investigation")}


@pytest.mark.unit
def test_agent_composes_role_and_starts_with_empty_stm():
    """An Agent composes the role (a reference, not a copy) and carries the work and
    an empty STM."""
    role = _RuleBackedRole(_STORE)
    work = cast(NodeBase, object())

    agent = Agent(role, role.reactions()[0].execute, work)

    assert agent.role is role  # composition: the same instance, not a clone
    assert agent.work is work
    assert agent.messages == []


@pytest.mark.unit
async def test_agent_run_delegates_to_reaction_execute():
    """Agent.run() delegates to the reaction's execute, passing the agent itself, so
    the logic operates on this agent's state (work + STM)."""
    role = _LLMBackedRole(_STORE, MockProvider([]))
    reaction = role.reactions()[0]
    agent = Agent(role, reaction.execute, work=cast(NodeBase, object()))

    await agent.run()

    assert agent.messages == [{"executed_by": "llm"}]


@pytest.mark.unit
async def test_role_on_failure_is_noop_by_default():
    """The base on_failure hook is a no-op: it must not raise."""
    role = _RuleBackedRole(_STORE)

    await role.on_failure(cast(NodeBase, object()))  # must not raise


# ---- tools: registry ----


class _EchoTool(Tool):
    name = "echo"
    description = "echoes the given text back"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, text: str) -> str:
        return f"echo: {text}"


@pytest.mark.unit
async def test_registry_runs_tools_and_returns_errors_as_text():
    """The registry runs a tool by name; unknown tools and tool crashes come back as
    TEXT (the model is the error handler: it reads and adapts)."""
    registry = ToolRegistry([_EchoTool()])

    assert await registry.run("echo", {"text": "hola"}) == "echo: hola"
    assert "unknown tool" in await registry.run("nope", {})
    assert "error running" in await registry.run("echo", {"wrong_arg": 1})


@pytest.mark.unit
def test_registry_specs_expose_the_catalog():
    """specs() renders the catalog in function-calling format for the provider."""
    registry = ToolRegistry([_EchoTool()])

    specs = registry.specs()

    assert len(specs) == 1
    assert specs[0]["function"]["name"] == "echo"


# ---- the tool loop ----


def _tool_call_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content="", tool_calls=[ToolCall(id="call-1", name=name, arguments=arguments)]
    )


@pytest.mark.unit
async def test_tool_loop_runs_tools_then_returns_final_answer():
    """The loop feeds tool results back to the model (via the agent's STM) until it
    answers without tool calls."""
    provider = MockProvider(
        [
            _tool_call_response("echo", {"text": "primera"}),
            LLMResponse(content=json.dumps({"done": True})),
        ]
    )
    role = _LLMBackedRole(_STORE, provider)
    agent = Agent(role, role.reactions()[0].execute, work=cast(NodeBase, object()))
    registry = ToolRegistry([_EchoTool()])

    response = await run_tool_loop(provider, registry, agent, [{"role": "user", "content": "x"}])

    assert response.content == json.dumps({"done": True})
    # the STM recorded the whole exchange: user, assistant tool call, tool result
    roles_in_stm = [message["role"] for message in agent.messages]
    assert roles_in_stm == ["user", "assistant", "tool"]
    assert agent.messages[2]["content"] == "echo: primera"


@pytest.mark.unit
async def test_tool_loop_is_bounded():
    """A model that never stops calling tools hits MAX_TOOL_ITERATIONS and raises:
    the runaway loop becomes a normal agent failure (fail/retry bounds the spend)."""
    provider = MockProvider(
        [_tool_call_response("echo", {"text": "otra vez"}) for _ in range(MAX_TOOL_ITERATIONS)]
    )
    role = _LLMBackedRole(_STORE, provider)
    agent = Agent(role, role.reactions()[0].execute, work=cast(NodeBase, object()))
    registry = ToolRegistry([_EchoTool()])

    with pytest.raises(RuntimeError, match="MAX_TOOL_ITERATIONS"):
        await run_tool_loop(provider, registry, agent, [{"role": "user", "content": "x"}])
