import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Event(BaseModel):
    type: str  # e.g. "node_created", "node_updated", "investigation_closed"
    node_id: str | None = None  # a minimal pointer: "look at node X"
    node_type: str | None = None  # e.g. "Evidence", "Hypothesis"
    payload: dict = {}


# A handler takes an Event and returns either an awaitable (async def) or nothing
# (a sync function or lambda). Both are supported.
Handler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """In-memory async event bus (asyncio, no external deps).

    Agents subscribe to graph event types relevant to their role; the GraphStore
    publishes after each mutation. Coordination stays in the graph, an event is
    only a pointer ("look at node X"), not a message between agents.
    """

    def __init__(self) -> None:
        # key: (event_type, node_type) — node_type None means "any node type"
        self._handlers: dict[tuple[str, str | None], list[Handler]] = defaultdict(list)
        self._tasks: set[asyncio.Task] = set()

    def subscribe(self, event_type: str, handler: Handler, node_type: str | None = None) -> None:
        """Register a sync or async handler. Narrow it to a node_type, or leave it
        None to receive every node_type for this event (e.g. the Monitor)."""
        self._handlers[(event_type, node_type)].append(handler)

    def publish(self, event: Event) -> None:
        """Fire-and-forget: schedule every subscribed handler as its own task.

        Sync on purpose — it only schedules, it does not await. The publisher
        (a GraphStore mutation) returns immediately and never blocks on, or
        deadlocks with, subscribers. All matching handlers run concurrently.
        """
        # handlers narrowed to this node_type, plus those subscribed to "any"
        # (node_type=None). Guard against double-delivery when the event itself
        # has no node_type (e.g. edge_created).
        handlers = list(self._handlers.get((event.type, event.node_type), ()))
        if event.node_type is not None:
            handlers += self._handlers.get((event.type, None), ())
        for handler in handlers:
            task = asyncio.create_task(self._safe_run(handler, event))
            self._tasks.add(task)  # keep a strong ref so the GC can't cancel it
            task.add_done_callback(self._tasks.discard)

    async def _safe_run(self, handler: Handler, event: Event) -> None:
        # Supports sync and async handlers: a sync one already ran when called and
        # returned None; an async one returned a coroutine we still have to await.
        # A failing handler must not take down the publisher or the other handlers.
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("event handler failed (type=%s)", event.type)

    async def aclose(self) -> None:
        """Wait for in-flight handlers to finish (orderly shutdown / tests)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
