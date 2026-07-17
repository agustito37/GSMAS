import json

from core.graph.store import GraphStore
from core.tools.base import Tool


class GraphReadTool(Tool):
    """Read-only zoom into the graph: fetch the full content of a single node by its
    id. Pairs with a case map (which lists node ids): the agent scans the map, then
    pulls the detail only for the nodes it needs. Never mutates."""

    name = "get_node"
    description = (
        "Fetch the full content (all fields) of one graph node by its id. Read-only. "
        "Use it to zoom into a node listed in the case map (an evidence's text, a "
        "hypothesis's rationale, etc.) instead of loading everything up front."
    )
    parameters = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "the id of the node to fetch"}
        },
        "required": ["node_id"],
    }

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    async def run(self, node_id: str) -> str:
        node = await self._store.get_node(node_id)
        if node is None:
            return f"no node with id '{node_id}'"
        return json.dumps(node.model_dump(mode="json"))
