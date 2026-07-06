from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from core.graph.models import Claimable, NodeBase
from core.graph.store import GraphStore
from core.providers.base import LLMProvider

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
    """A role. Register ONE instance. It declares its reactions and carries the
    shared dependencies (store, provider) but NO per-work state, agents carry that,
    so a single role instance is used by many concurrent agents safely."""

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

class LLMRole(Role):
    """A role backed by an LLM (the domain roles + the LLM system roles). The model
    is a parameter of the role, not of the architecture (provider-agnostic)."""

    def __init__(self, store: GraphStore, provider: LLMProvider) -> None:
        super().__init__(store)
        self.provider = provider

class DeterministicRole(Role):
    """A role with no LLM: applies structural rules over the graph (e.g. recovery).
    Inherits Role.__init__ unchanged."""

class Agent:
    """Ephemeral execution unit. COMPOSES the registered role (a reference) plus the
    per-execution state: the work (a graph node), its own STM, and the reaction's
    execute. Drives its claim lifecycle: complete() on success, fail() on error
    (bounded by MAX_ATTEMPTS)."""

    def __init__(self, role: Role, execute: ExecuteFn, work: NodeBase) -> None:
        self.role = role  # composition: shared store/provider, and on_failure
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
