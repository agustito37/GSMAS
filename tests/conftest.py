"""Shared test fixtures.

Integration tests run against a throwaway Neo4j started via testcontainers
(Docker). They are excluded by default (see `addopts` in pyproject.toml); run them
explicitly:

    uv run pytest                 # unit only (fast, no Docker)
    uv run pytest -m integration  # integration (starts a Neo4j container)
"""

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from core.graph.store import GraphStore

_NEO4J_IMAGE = "neo4j:5"


@pytest.fixture(scope="session")
def neo4j_container():
    """Start one throwaway Neo4j for the whole test session."""
    with Neo4jContainer(_NEO4J_IMAGE) as container:
        yield container


@pytest_asyncio.fixture
async def store(neo4j_container):
    """A GraphStore on a freshly-emptied throwaway Neo4j.

    The graph is emptied in setup (not teardown) so each test starts clean even if
    a previous test crashed before its teardown ran. The wipe uses a dedicated
    admin driver, kept separate from the store under test.
    """
    uri = neo4j_container.get_connection_url()
    auth = (neo4j_container.username, neo4j_container.password)

    async with AsyncGraphDatabase.driver(uri, auth=auth) as admin, admin.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    graph_store = GraphStore(uri, *auth)
    yield graph_store
    await graph_store.close()


@pytest_asyncio.fixture
async def raw(neo4j_container):
    """Run raw Cypher against the test Neo4j. For building INVALID states on purpose
    (orphans, unlinked verdicts): the store forbids them by construction ("born
    connected"), but the invariant checkers must still detect them if they appear by
    another path (manual edits, bugs, writers that bypass the store)."""
    uri = neo4j_container.get_connection_url()
    auth = (neo4j_container.username, neo4j_container.password)
    driver = AsyncGraphDatabase.driver(uri, auth=auth)

    async def run(query, **params):
        async with driver.session() as session:
            await session.run(query, **params)

    yield run
    await driver.close()
