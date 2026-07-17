"""End-to-end smoke test of the runtime wiring.

Exercises the whole chain a single InputSignal travels through:

    submit_signal -> store.create_node -> on_mutation callback -> bus.publish
        -> orchestrator drain -> swarm worker -> Agent.run -> Role.execute

It proves the pieces are wired together (no direct store<->bus coupling, fire-and-
forget handlers, the swarm queue + worker delivering the Agent to the role) with no
LLM or domain logic. Runs against a throwaway Neo4j (testcontainers).

    uv run pytest -m integration
"""

import asyncio
from typing import cast

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase

from core.graph.models import Case, InputSignal, NodeBase, Verdict
from core.graph.store import EdgeSpec
from core.roles.base import Executor, Reaction, Role
from core.runtime.orchestrator import Orchestrator


async def _open_case(store) -> Case:
    """A legal case root: InputSignal (born bare) + Case born connected via OPENS."""
    signal = InputSignal(raw_content="a signal")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="an objective", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    return case


class _DummyRole(Role):
    """Minimal role for the smoke test: one reaction that reacts to InputSignal
    creation, claims each signal via the store's atomic claim, and on execute records
    what it received and signals completion. No LLM, no domain logic."""

    def __init__(self, store) -> None:
        super().__init__(store)
        self.done = asyncio.Event()  # the test awaits this
        self.seen_content: str | None = None

    def reactions(self) -> list[Reaction]:
        trigger = ("node_created", "InputSignal")
        return [Reaction({trigger}, self._claim_signal, self._handle_signal)]

    async def _claim_signal(self) -> NodeBase | None:
        # the real claim: atomic, in the graph (pending -> claimed), survives restarts
        return await self.store.claim("InputSignal", {})

    async def _handle_signal(self, agent: Executor) -> None:
        signal = cast(InputSignal, agent.work)  # the claim only ever returns InputSignals
        self.seen_content = signal.raw_content  # proof the chain delivered the node
        self.done.set()  # wake the waiting test


@pytest_asyncio.fixture
async def orchestrator(neo4j_container):
    """An Orchestrator on a freshly-emptied throwaway Neo4j (its GraphStore wires
    on_mutation -> bus). The graph is emptied in setup so the test starts clean."""
    uri = neo4j_container.get_connection_url()
    auth = (neo4j_container.username, neo4j_container.password)
    async with AsyncGraphDatabase.driver(uri, auth=auth) as admin, admin.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    orch = Orchestrator(uri, *auth)
    yield orch
    await orch.aclose()


@pytest.mark.integration
async def test_input_signal_reaches_a_registered_role_end_to_end(orchestrator):
    """A submitted InputSignal travels the full runtime chain and reaches the role's
    execute, carrying the right node."""
    dummy = _DummyRole(orchestrator.store)
    orchestrator.register(dummy)
    await orchestrator.start()

    raw = "raw log: suspicious login from 10.0.0.5"
    await orchestrator.submit_signal(raw)

    # The bus handler is fire-and-forget: the role runs AFTER submit_signal returns.
    # Wait for completion instead of asserting immediately (otherwise it is flaky).
    await asyncio.wait_for(dummy.done.wait(), timeout=10)

    assert dummy.seen_content == raw


@pytest.mark.integration
async def test_wait_for_closure_returns_if_verdict_already_exists(orchestrator):
    """The race fix: if the Verdict already exists when wait_for_closure is called (it
    could have appeared before subscribing), the check-then-wait finds it and returns
    at once instead of hanging."""
    case = await _open_case(orchestrator.store)
    verdict = Verdict(case_id=case.id, kind="resolved", content="done")
    await orchestrator.store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])

    # must return promptly; without check-then-wait this would hang (event already passed)
    await asyncio.wait_for(orchestrator.wait_for_closure(case.id), timeout=5)


@pytest.mark.integration
async def test_wait_for_closure_completes_when_verdict_appears_after(orchestrator):
    """If the Verdict appears after wait_for_closure is called, the subscription (or
    the check) catches it and the wait completes instead of hanging."""
    case = await _open_case(orchestrator.store)
    waiter = asyncio.create_task(orchestrator.wait_for_closure(case.id))

    verdict = Verdict(case_id=case.id, kind="unresolved", content="done")
    await orchestrator.store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])

    await asyncio.wait_for(waiter, timeout=5)  # must complete


@pytest.mark.integration
async def test_submit_signal_scopes_to_a_workspace(orchestrator):
    """A signal is submitted into a workspace: the InputSignal carries the
    workspace_id and the Workspace node is ensured (the scope every skill inherits)."""
    await orchestrator.start()
    signal_id = await orchestrator.submit_signal("a signal", workspace_id="exp1")

    signal = await orchestrator.store.get_node(signal_id)
    assert isinstance(signal, InputSignal)
    assert signal.workspace_id == "exp1"
    workspaces = await orchestrator.store.query_nodes("Workspace", {})
    assert "exp1" in [w.id for w in workspaces]
