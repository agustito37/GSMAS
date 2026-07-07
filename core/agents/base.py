from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from core.graph.models import Claimable, NodeBase
from core.graph.store import GraphStore

MAX_ATTEMPTS = 3  # retries of one unit of work before it is marked 'failed' and dropped

ClaimFn = Callable[[], Awaitable[NodeBase | None]]
ExecuteFn = Callable[["Agent"], Awaitable[None]]


@dataclass(frozen=True)
class Reaction:
    """One reactive behavior of a role: which events wake it (triggers), what work
    it then claims from the graph (claim), and how an agent runs that work (execute).
    Keeping the three together makes each line of work self-contained, so a role can
    have several independent ones without coupling them through a single dispatch."""

    triggers: set[tuple[str, str | None]]
    claim: ClaimFn
    execute: ExecuteFn


class Role(ABC):
    """A role: the ABSTRACT responsibility (its reactions) plus its shared
    dependencies. The framework does NOT classify roles by how they reason: each
    judgment (a reaction's execute) decides its own substrate - LLM, rules, or a
    mix - and each concrete role declares whatever dependencies its judgments use
    (store always; a provider and/or the tool catalog only if needed) in its own
    __init__. Register ONE instance; it carries NO per-work state (agents do), so
    many concurrent agents share it safely. The graph's Role.agent_type remains as
    descriptive metadata; it is not a class hierarchy."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    @abstractmethod
    def reactions(self) -> list[Reaction]:
        """The role's reactive behaviors, one per independent line of work. Each
        Reaction binds the events that wake it, the claim that pulls its pending unit
        from the graph (atomic SET ... WHERE so a unit is never taken twice), and the
        execute an agent runs on that unit. Mono-purpose roles return one Reaction."""

    # on_failure is an optional hook with a default no-op; not @abstractmethod on purpose.
    async def on_failure(self, work: NodeBase) -> None:  # noqa: B027
        """Optional hook called AFTER the framework's retry guard ran (store.fail
        already incremented attempts and moved the node to pending or failed). For
        custom cleanup/logging only; default no-op."""


class Agent:
    """Ephemeral execution unit. COMPOSES the registered role (a reference) plus the
    per-execution state: the work (a graph node), its own STM, and the reaction's
    execute. Drives its claim lifecycle: complete() on success, fail() on error
    (bounded by MAX_ATTEMPTS)."""

    def __init__(self, role: Role, execute: ExecuteFn, work: NodeBase) -> None:
        self.role = role  # composition: shared deps and on_failure
        self.work = work  # the graph node this agent processes; the role knows its concrete type
        self.messages: list[dict] = []  # STM, isolated per agent
        self._execute = execute  # the reaction's execute, bound to this work

    async def run(self) -> None:
        await self._execute(self)  # delegate to the reaction's logic

    async def complete(self) -> None:
        if isinstance(self.work, Claimable):
            await self.role.store.complete(self.work.id)  # claimed -> done

    async def fail(self) -> None:
        # framework guard FIRST (increment + pending-or-failed), then the role hook
        if isinstance(self.work, Claimable):
            await self.role.store.fail(self.work.id, MAX_ATTEMPTS)
        await self.role.on_failure(self.work)
