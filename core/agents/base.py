import json
from typing import TypeVar

from pydantic import BaseModel

from core.graph.models import Claimable, NodeBase
from core.providers.base import LLMProvider, LLMResponse
from core.roles.base import ExecuteFn, Executor, Role
from core.tools.base import ToolRegistry

MAX_ATTEMPTS = 3  # retries of one unit of work before it is marked 'failed' and dropped
MAX_TOOL_ITERATIONS = 12  # LLM turns per agent run; overflow raises, so fail/retry bounds spend

T = TypeVar("T", bound=BaseModel)


class Agent(Executor):
    """Ephemeral execution unit. COMPOSES the registered role (a reference) plus the
    per-execution state: the work (a graph node), its own STM, the reaction's
    execute, the ENGINE backing this execution (provider + catalog access, stamped
    by the runtime at spawn), and the running cost counters of this episode.
    IMPLEMENTS the roles layer's Executor contract explicitly (classic dependency
    inversion: the declaration layer owns the interface, the execution layer
    subscribes to it; the import direction stays agents -> roles). Drives its claim
    lifecycle: complete() on success, fail() on error (bounded by MAX_ATTEMPTS)."""

    def __init__(
        self,
        role: Role,
        execute: ExecuteFn,
        work: NodeBase,
        provider: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.role = role  # composition: shared deps and on_failure
        self.work = work  # the graph node this agent processes; the role knows its concrete type
        self.messages: list[dict] = []  # STM, isolated per agent
        self.provider = provider  # the engine backing THIS execution, not the role's
        self.tools = tools  # access to the common tool catalog
        self.tokens_in = 0  # cost of this episode, summed over every LLM call in the loop
        self.tokens_out = 0
        self.llm_calls = 0
        self._execute = execute  # the reaction's execute, bound to this work

    async def run(self) -> None:
        await self._execute(self)  # delegate to the reaction's logic

    async def run_llm(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        tools: ToolRegistry | None = None,
    ) -> T:
        """Run one LLM exchange on this agent's engine and return the answer parsed
        into `schema`. The role supplies the judgment (system prompt, user context,
        schema, and tools IF its judgment uses them); the agent supplies the engine
        (its provider) and records the whole exchange in its own STM. Runs through
        the tool loop even without tools (it degenerates to a single completion).
        A malformed final answer raises (pydantic ValidationError): the worker fails
        the agent and the retry budget takes over."""
        if self.provider is None:
            raise RuntimeError(
                "agent has no provider: register the role with one (register(role, provider=...))"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = await run_tool_loop(self.provider, tools, self, messages, response_schema=schema)
        return schema.model_validate_json(response.content)

    async def record_cost(self, elapsed_ms: float) -> None:
        """Persist what THIS episode spent onto its work node (bookkeeping for
        evaluation; the framework never reads it back). Called by the worker once the
        episode ends, on success AND failure - the tokens were spent either way; with
        retries the store accumulates. Uses store.record_cost, which does not emit an
        event (a domain mutation here would spuriously wake reactions)."""
        await self.role.store.record_cost(
            self.work.id, self.tokens_in, self.tokens_out, self.llm_calls, elapsed_ms
        )

    async def complete(self) -> None:
        if isinstance(self.work, Claimable):
            await self.role.store.complete(self.work.id)  # claimed -> done

    async def fail(self) -> None:
        # framework guard FIRST (increment + pending-or-failed), then the role hook
        if isinstance(self.work, Claimable):
            await self.role.store.fail(self.work.id, MAX_ATTEMPTS)
        await self.role.on_failure(self.work)


async def run_tool_loop(
    provider: LLMProvider,
    tools: ToolRegistry | None,
    agent: Agent,
    messages: list[dict],
    response_schema: type | None = None,
) -> LLMResponse:
    """Drive the loop: the model invokes tools until it emits its final answer.
    Lives here (not in a separate module) because it mutates the Agent: agent.messages
    is the STM, and every provider call's usage accumulates onto the agent's cost
    counters (so the worker can persist the episode's spend, even on a loop that
    raises). Tool failures return to the model as text (it adapts). The loop cannot run
    away: on the final turn the tools are withheld, so a model that keeps calling them is
    forced to commit to its answer from what it already gathered. That degrades a runaway
    loop to a grounded (possibly empty) answer instead of a hard failure, which matters
    because a stranded unit of work with no result blocks the case's quiescence and the
    verdict never comes."""
    agent.messages = list(messages)
    specs = tools.specs() if tools else None
    for i in range(MAX_TOOL_ITERATIONS):
        # withhold the tools on the last allowed turn: the model MUST answer now, so the
        # loop always terminates with a result rather than raising and stranding the work
        last_turn = i == MAX_TOOL_ITERATIONS - 1
        response = await provider.complete(
            agent.messages,
            tools=None if last_turn else specs,
            response_schema=response_schema,
        )
        agent.tokens_in += response.usage.get("input_tokens", 0)
        agent.tokens_out += response.usage.get("output_tokens", 0)
        agent.llm_calls += 1
        if not response.tool_calls:
            return response
        agent.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in response.tool_calls
                ],
            }
        )
        for call in response.tool_calls:
            result = (
                await tools.run(call.name, call.arguments)
                if tools
                else f"error: no tools available (requested '{call.name}')"
            )
            agent.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return response  # the final turn withheld tools, so it returned a tool-free answer
