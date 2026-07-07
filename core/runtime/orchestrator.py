import asyncio
import logging

from core.agents.base import MAX_ATTEMPTS, Agent, Reaction, Role
from core.events.bus import Event, EventBus
from core.graph.models import InputSignal
from core.graph.store import GraphStore
from core.runtime.swarm import Swarm

logger = logging.getLogger("haive.orchestrator")

class Orchestrator:
    """System runtime. Builds the generic pieces (bus, store, swarm), registers roles,
    and wires each reaction to the bus: claim that reaction's pending work and enqueue
    it on the swarm. Entry point for input."""

    def __init__(
        self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, max_agents: int = 8
    ) -> None:
        self._bus = EventBus()
        self._store = GraphStore(
            neo4j_uri, neo4j_user, neo4j_password, on_mutation=self._on_mutation
        )
        self._swarm = Swarm(max_agents)
        self._roles: list[Role] = []

    def _on_mutation(
        self, event_type: str, node_id: str | None, node_type: str | None, payload: dict
    ) -> None:
        self._bus.publish(
            Event(type=event_type, node_id=node_id, node_type=node_type, payload=payload)
        )

    @property
    def store(self) -> GraphStore:
        return self._store  # domain builds roles with this, then registers them

    @property
    def bus(self):
        return self._bus  # read-only observers (e.g. the dashboard) subscribe here

    # ---- registration / wiring ----
    def register(self, role: Role) -> None:
        """Register a role instance and wire each of its reactions to the bus. On a
        matching event, drain that reaction's pending work and enqueue an agent per
        unit. Domain builds the concrete roles (store/provider) and registers them."""
        self._roles.append(role)
        for reaction in role.reactions():
            for event_type, node_type in reaction.triggers:
                self._bus.subscribe(
                    event_type, self._handler_for(role, reaction), node_type=node_type
                )

    def _handler_for(self, role: Role, reaction: Reaction):
        async def on_event(_event: Event) -> None:
            await self._drain(role, reaction)
        return on_event

    async def _drain(self, role: Role, reaction: Reaction) -> None:
        while (work := await reaction.claim()) is not None:
            await self._swarm.submit(Agent(role, reaction.execute, work))

    async def start(self) -> None:
        """Spawn the worker pool, recover any 'claimed' orphans left by a previous
        process (startup sweep), then do the initial drain across every reaction."""
        self._swarm.start()
        recovered = await self._store.recover_claimed(MAX_ATTEMPTS)
        logger.info(
            "started: %d roles registered, %d orphaned unit(s) recovered",
            len(self._roles),
            recovered,
        )
        for role in self._roles:
            for reaction in role.reactions():
                await self._drain(role, reaction)

    # ---- lifecycle ----
    async def submit_signal(self, raw_content: str) -> str:
        """Materialize an InputSignal and return its id. The ONLY intake step;
        opening the Case is the Theorist's decision (a registered role)."""
        signal = InputSignal(raw_content=raw_content)
        await self._store.create_node(signal, "InputSignal")
        return signal.id

    async def wait_for_closure(self, case_id: str) -> None:
        """Block until a Verdict exists for the case (event-driven).
        Subscribes BEFORE checking the graph, so a Verdict that appears between the
        check and the wait is still caught; if it already exists, returns at once.
        Unsubscribes on exit (no leaked handler)."""
        closed = asyncio.Event()

        async def on_verdict(event: Event) -> None:
            verdict = await self._store.get_node(event.node_id) if event.node_id else None
            if verdict is not None and getattr(verdict, "case_id", None) == case_id:
                closed.set()

        self._bus.subscribe("node_created", on_verdict, node_type="Verdict")
        try:
            existing = await self._store.query_nodes("Verdict", {"case_id": case_id})
            if not existing:
                await closed.wait()
        finally:
            self._bus.unsubscribe("node_created", on_verdict, node_type="Verdict")

    async def aclose(self) -> None:
        await self._swarm.aclose()
        await self._bus.aclose()
        await self._store.close()
