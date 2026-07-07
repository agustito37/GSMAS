from typing import TypeVar

from pydantic import BaseModel

from core.agents.tool_loop import run_tool_loop
from core.graph.models import Claimable, NodeBase
from core.providers.base import LLMProvider
from core.roles.base import ExecuteFn, Executor, Role
from core.tools.base import ToolRegistry

MAX_ATTEMPTS = 3  # retries of one unit of work before it is marked 'failed' and dropped

T = TypeVar("T", bound=BaseModel)


class Agent(Executor):
    """Ephemeral execution unit. COMPOSES the registered role (a reference) plus the
    per-execution state: the work (a graph node), its own STM, the reaction's
    execute, and the ENGINE backing this execution (provider + catalog access,
    stamped by the runtime at spawn). IMPLEMENTS the roles layer's Executor
    contract explicitly (classic dependency inversion: the declaration layer owns
    the interface, the execution layer subscribes to it; the import direction stays
    agents -> roles). Drives its claim lifecycle: complete() on success, fail() on
    error (bounded by MAX_ATTEMPTS)."""

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

    async def complete(self) -> None:
        if isinstance(self.work, Claimable):
            await self.role.store.complete(self.work.id)  # claimed -> done

    async def fail(self) -> None:
        # framework guard FIRST (increment + pending-or-failed), then the role hook
        if isinstance(self.work, Claimable):
            await self.role.store.fail(self.work.id, MAX_ATTEMPTS)
        await self.role.on_failure(self.work)
