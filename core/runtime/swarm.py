import asyncio
import logging
import time
from collections.abc import Callable

from core.agents.base import Agent

logger = logging.getLogger("haive.swarm")

# (phase, role, action, work_id, work_type) -> None. Observes agent lifecycle
# transitions for live tracing; agents are ephemeral (not graph nodes), so this is the
# only window in. `action` is the reaction being run (e.g. investigate, retrospect).
OnAgentEvent = Callable[[str, str, str, str, str], None]


class Swarm:
    """The swarm's capacity, as a producer/consumer queue: a FIFO work queue with
    a FIXED pool of N workers. Roles enqueue agents (submit); each free worker pulls
    the next agent and runs it, then pulls the next. All workers busy -> agents wait
    in the queue until one frees. The cap is the number of workers; a worker blocks
    on queue.get() when idle.
    """

    def __init__(self, max_agents: int, on_agent_event: OnAgentEvent | None = None) -> None:
        self._max = max_agents
        self._queue: asyncio.Queue[Agent] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._on_agent_event = on_agent_event

    def start(self) -> None:
        """Spawn the worker pool. Call once inside a running event loop (creating
        tasks needs one)."""
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self._max)]

    async def submit(self, agent: Agent) -> None:
        """Enqueue an agent. Returns at once (unbounded queue); a free worker picks
        it up. If all workers are busy it simply waits its turn in the queue."""
        await self._queue.put(agent)

    async def _worker(self) -> None:
        while True:
            agent = await self._queue.get()  # block until there is work (no polling)
            self._trace(agent, "started")
            start = time.monotonic()
            try:
                await agent.run()
            except Exception:
                # the failure is handled (fail() bounds retries) but must be SEEN:
                # without this line a broken role dies silently, attempt after attempt
                logger.exception(
                    "agent failed (role=%s, work=%s)",
                    type(agent.role).__name__,
                    agent.work.id,
                )
                await agent.fail()
                self._trace(agent, "failed")
            else:
                await agent.complete()
                self._trace(agent, "finished")
            finally:
                # cost is recorded regardless of outcome: the tokens were spent even
                # if the episode raised (bookkeeping for evaluation, no side effects)
                await agent.record_cost((time.monotonic() - start) * 1000)

    def _trace(self, agent: Agent, phase: str) -> None:
        """Announce an agent lifecycle transition (started/finished/failed). Mutates
        nothing; it is the live window into who works on what, since agents are
        ephemeral and never reified as graph nodes. The action is the reaction being run
        (its execute's name), so a retrospective reads as such, not as generic Case work."""
        if self._on_agent_event is None:
            return
        work = agent.work
        action = getattr(agent._execute, "__name__", "").lstrip("_")
        self._on_agent_event(
            phase, agent.role.name, action, getattr(work, "id", ""), type(work).__name__
        )

    async def aclose(self) -> None:
        for worker in self._workers:
            worker.cancel()  # CancelledError is BaseException -> not caught above
