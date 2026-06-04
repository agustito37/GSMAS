from abc import ABC, abstractmethod

from core.events.bus import EventBus, Handler
from core.graph.models import Role
from core.graph.store import GraphStore
from core.providers.base import LLMProvider


class Agent(ABC):
    """Base agent: reacts to graph events relevant to its role.

    Agents are ephemeral and interchangeable within a role. They never message one
    another, they read the graph (pull) and mutate it; the bus only points them
    at what changed. All coordination lives in the graph (stigmergy).
    """

    def __init__(self, agent_id: str, role: Role, store: GraphStore, bus: EventBus) -> None:
        self.id = agent_id
        self.role = role
        self.store = store
        self.bus = bus

    @abstractmethod
    def subscriptions(self) -> dict[tuple[str, str | None], Handler]:
        """Map each (event_type, node_type) to the method that handles it;
        node_type None means 'any'. One method per responsibility, no monolithic
        handler. E.g. a Verifier returns {("node_created", "Evidence"): self.on_evidence}."""
        ...

    def start(self) -> None:
        """Register this agent's handlers on the bus, per its subscriptions."""
        for (event_type, node_type), handler in self.subscriptions().items():
            self.bus.subscribe(event_type, handler, node_type=node_type)


class LLMAgent(Agent):
    """An agent backed by an LLM. The model is a parameter of the agent, not of
    the architecture (provider-agnostic): two agents of the same role may run on
    different providers and still share the role's LTM."""

    def __init__(
        self,
        agent_id: str,
        role: Role,
        store: GraphStore,
        bus: EventBus,
        provider: LLMProvider,
    ) -> None:
        super().__init__(agent_id, role, store, bus)
        self.provider = provider
        # STM, the working context (message list) for THIS agent's single
        # unit of work. Per-instance is safe under the ephemeral model: one agent
        # handles one unit of work and is discarded; it never serves two at once.
        # In memory, private, never persisted (what must survive goes to the graph).
        self._messages: list[dict] = []


class DeterministicAgent(Agent):
    """An agent with no LLM: applies structural rules over the graph (e.g. the
    Monitor). Used wherever the task is structural rather than linguistic."""
