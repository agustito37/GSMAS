from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

from core.graph.models import NodeBase
from core.graph.store import GraphStore
from core.tools.base import ToolRegistry

T = TypeVar("T", bound=BaseModel)


class Executor(Protocol):
    """The minimal contract a role's execute receives from whatever runs it: the
    claimed work, the episode's STM, catalog access, and the LLM capability.
    Roles declare against THIS, never against the concrete executor: the roles
    layer is pure declaration, and the agents layer satisfies the contract
    structurally (no inheritance, checked by the type checker)."""

    work: NodeBase
    messages: list[dict]
    tools: ToolRegistry | None

    async def run_llm(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        tools: ToolRegistry | None = None,
    ) -> T: ...


ClaimFn = Callable[[], Awaitable[NodeBase | None]]
ExecuteFn = Callable[[Executor], Awaitable[None]]


@dataclass(frozen=True)
class Reaction:
    """One reactive behavior of a role: which events wake it (triggers), what work
    it then claims from the graph (claim), and how an executor runs that work
    (execute). Keeping the three together makes each line of work self-contained,
    so a role can have several independent ones without coupling them through a
    single dispatch."""

    triggers: set[tuple[str, str | None]]
    claim: ClaimFn
    execute: ExecuteFn


class Role(ABC):
    """A role: the ABSTRACT responsibility (its reactions) plus the store and its
    own working state. The framework does NOT classify roles by how they reason:
    each judgment (a reaction's execute) decides its own substrate - LLM, rules,
    or a mix. The role defines the judgment (prompt, context, output schema, what
    the output means for the graph); the ENGINE that runs it is per-execution
    state carried by the executor, behind the Executor contract. This module
    depends on no execution machinery: roles are pure declaration. Register ONE
    instance; it carries NO per-work state (executors do), so many concurrent
    agents share it safely. The graph's Role.agent_type remains as descriptive
    metadata; it is not a class hierarchy."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    @abstractmethod
    def reactions(self) -> list[Reaction]:
        """The role's reactive behaviors, one per independent line of work. Each
        Reaction binds the events that wake it, the claim that pulls its pending unit
        from the graph (atomic SET ... WHERE so a unit is never taken twice), and the
        execute an executor runs on that unit. Mono-purpose roles return one Reaction."""

    # on_failure is an optional hook with a default no-op; not @abstractmethod on purpose.
    async def on_failure(self, work: NodeBase) -> None:  # noqa: B027
        """Optional hook called AFTER the framework's retry guard ran (store.fail
        already incremented attempts and moved the node to pending or failed). For
        custom cleanup/logging only; default no-op."""
