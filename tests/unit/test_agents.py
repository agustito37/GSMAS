"""Unit tests for the base Role/Reaction/Agent runtime, the tool loop, and the
agent's LLM capability (run_llm).

Run with:
    uv run pytest
"""

import json
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from core.agents.base import Agent
from core.agents.tool_loop import MAX_TOOL_ITERATIONS, run_tool_loop
from core.graph.models import NodeBase
from core.graph.store import GraphStore
from core.providers.base import LLMResponse, ToolCall
from core.roles.base import Executor, Reaction, Role
from core.tools.base import Tool, ToolRegistry
from tests.mocks.mock_provider import MockProvider

# The store is unused by these tests (roles only keep the reference); a real
# GraphStore would open a Neo4j driver, so we pass a stand-in.
_STORE = cast(GraphStore, object())


class _SomeRole(Role):
    """Minimal role: the substrate of each judgment is the reaction's business;
    the role declares no engine (providers ride with the agents)."""

    async def _claim(self) -> NodeBase | None:
        return None

    async def _run(self, agent: Executor) -> None:
        agent.messages.append({"executed_by": "the reaction"})

    def reactions(self) -> list[Reaction]:
        return [Reaction({("node_created", "InputSignal")}, self._claim, self._run)]


@pytest.mark.unit
def test_role_declares_its_reactions():
    """A role exposes its behaviors as Reactions binding triggers, claim and execute."""
    role = _SomeRole(_STORE)

    reactions = role.reactions()

    assert len(reactions) == 1
    assert reactions[0].triggers == {("node_created", "InputSignal")}


@pytest.mark.unit
def test_agent_carries_the_engine_not_the_role():
    """The provider and the catalog access are per-execution state stamped on the
    AGENT at spawn; the role carries only the store. Two agents of the same role
    can run on different engines."""
    role = _SomeRole(_STORE)
    provider_a, provider_b = MockProvider([]), MockProvider([])
    work = cast(NodeBase, object())

    agent_a = Agent(role, role.reactions()[0].execute, work, provider=provider_a)
    agent_b = Agent(role, role.reactions()[0].execute, work, provider=provider_b)

    assert agent_a.provider is provider_a
    assert agent_b.provider is provider_b
    assert not hasattr(role, "provider")


@pytest.mark.unit
def test_agent_composes_role_and_starts_with_empty_stm():
    """An Agent composes the role (a reference, not a copy) and carries the work and
    an empty STM. Without an engine stamped, provider and tools are None (fine for
    deterministic reactions: they never use them)."""
    role = _SomeRole(_STORE)
    work = cast(NodeBase, object())

    agent = Agent(role, role.reactions()[0].execute, work)

    assert agent.role is role  # composition: the same instance, not a clone
    assert agent.work is work
    assert agent.messages == []
    assert agent.provider is None
    assert agent.tools is None


@pytest.mark.unit
async def test_agent_run_delegates_to_reaction_execute():
    """Agent.run() delegates to the reaction's execute, passing the agent itself, so
    the logic operates on this agent's state (work + STM)."""
    role = _SomeRole(_STORE)
    reaction = role.reactions()[0]
    agent = Agent(role, reaction.execute, work=cast(NodeBase, object()))

    await agent.run()

    assert agent.messages == [{"executed_by": "the reaction"}]


@pytest.mark.unit
async def test_role_on_failure_is_noop_by_default():
    """The base on_failure hook is a no-op: it must not raise."""
    role = _SomeRole(_STORE)

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


# ---- the tool loop (internal machinery of run_llm) ----


def _tool_call_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content="", tool_calls=[ToolCall(id="call-1", name=name, arguments=arguments)]
    )


def _agent(provider=None, tools=None) -> Agent:
    role = _SomeRole(_STORE)
    return Agent(
        role,
        role.reactions()[0].execute,
        work=cast(NodeBase, object()),
        provider=provider,
        tools=tools,
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
    agent = _agent(provider)
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
    agent = _agent(provider)
    registry = ToolRegistry([_EchoTool()])

    with pytest.raises(RuntimeError, match="MAX_TOOL_ITERATIONS"):
        await run_tool_loop(provider, registry, agent, [{"role": "user", "content": "x"}])


# ---- run_llm: the agent's LLM capability ----


class _Verdict(BaseModel):
    ok: bool


@pytest.mark.unit
async def test_run_llm_returns_the_parsed_answer_and_records_stm():
    """run_llm assembles the exchange on the agent's engine and returns the answer
    already parsed into the schema; the whole exchange lands in the agent's STM."""
    provider = MockProvider([LLMResponse(content=json.dumps({"ok": True}))])
    agent = _agent(provider)

    verdict = await agent.run_llm(system="you decide", user="the facts", schema=_Verdict)

    assert verdict.ok is True
    assert [m["role"] for m in agent.messages] == ["system", "user"]
    assert agent.messages[0]["content"] == "you decide"


@pytest.mark.unit
async def test_run_llm_with_tools_drives_the_tool_loop():
    """With a tool catalog, run_llm runs the full tool protocol before parsing the
    final answer."""
    provider = MockProvider(
        [
            _tool_call_response("echo", {"text": "dato"}),
            LLMResponse(content=json.dumps({"ok": False})),
        ]
    )
    registry = ToolRegistry([_EchoTool()])
    agent = _agent(provider, tools=registry)

    verdict = await agent.run_llm(system="s", user="u", schema=_Verdict, tools=agent.tools)

    assert verdict.ok is False
    assert "tool" in [m["role"] for m in agent.messages]


@pytest.mark.unit
async def test_run_llm_without_provider_is_a_wiring_error():
    """An agent spawned without an engine cannot run LLM judgments: loud failure
    pointing at the registration, not a silent skip."""
    agent = _agent(provider=None)

    with pytest.raises(RuntimeError, match="register"):
        await agent.run_llm(system="s", user="u", schema=_Verdict)


@pytest.mark.unit
async def test_run_llm_malformed_answer_is_a_normal_failure():
    """A final answer that does not match the schema raises: the worker fails the
    agent and the retry budget takes over (no silent bad judgment)."""
    provider = MockProvider([LLMResponse(content="not json at all")])
    agent = _agent(provider)

    with pytest.raises(ValidationError):
        await agent.run_llm(system="s", user="u", schema=_Verdict)
