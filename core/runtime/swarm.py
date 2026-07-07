import asyncio
import logging

from core.agents.base import Agent

logger = logging.getLogger("haive.swarm")

class Swarm:
    """The swarm's capacity, as a producer/consumer queue: a FIFO work queue with
    a FIXED pool of N workers. Roles enqueue agents (submit); each free worker pulls
    the next agent and runs it, then pulls the next. All workers busy -> agents wait
    in the queue until one frees. The cap is the number of workers; a worker blocks
    on queue.get() when idle.
    """

    def __init__(self, max_agents: int) -> None:
        self._max = max_agents
        self._queue: asyncio.Queue[Agent] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []

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
            else:
                await agent.complete()

    async def aclose(self) -> None:
        for worker in self._workers:
            worker.cancel()  # CancelledError is BaseException -> not caught above
