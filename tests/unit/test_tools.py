"""Unit tests for read-only graph tools.

Run with:
    uv run pytest
"""

import json
from typing import cast

import pytest

from core.graph.models import Evidence, NodeBase
from core.graph.store import GraphStore
from core.tools.graph_read import GraphReadTool


class _FakeStore:
    """A stand-in store: get_node returns the seeded node for its id, else None."""

    def __init__(self, node: NodeBase | None) -> None:
        self._node = node

    async def get_node(self, node_id: str) -> NodeBase | None:
        if self._node is not None and node_id == self._node.id:
            return self._node
        return None


@pytest.mark.unit
async def test_graph_read_tool_returns_node_content():
    """The tool fetches a node by id and returns its fields as JSON text (it goes back
    into the LLM's context)."""
    evidence = Evidence(content="jdoe logged in from Belarus", case_id="c1")
    tool = GraphReadTool(cast(GraphStore, _FakeStore(evidence)))

    result = await tool.run(evidence.id)

    assert "jdoe logged in from Belarus" in result
    assert json.loads(result)["id"] == evidence.id


@pytest.mark.unit
async def test_graph_read_tool_reports_missing_node_as_text():
    """An unknown id comes back as text (the model is the error handler), not an
    exception."""
    tool = GraphReadTool(cast(GraphStore, _FakeStore(None)))

    assert "no node with id" in await tool.run("nope")


@pytest.mark.unit
def test_graph_read_tool_spec_is_function_calling_format():
    """spec() renders the read tool in the provider's function-calling format."""
    tool = GraphReadTool(cast(GraphStore, _FakeStore(None)))

    assert tool.spec()["function"]["name"] == "get_node"
