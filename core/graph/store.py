from neo4j import AsyncGraphDatabase
from core.graph.models import (
  NodeBase, to_model,
  Investigation, Evidence, Hypothesis,
)

class GraphStore:
  def __init__(self, uri: str, user: str, password: str):
    self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

  async def close(self):
    await self._driver.close()

  # ---------- internal helper ----------

  def _to_model(self, neo4j_node) -> NodeBase:
    # a neo4j node carries both its properties and its labels
    props = dict(neo4j_node)
    label = list(neo4j_node.labels)[0]
    return to_model(label, props)

  # ---------- layer 1: generic primitives ----------

  async def create_node(self, node: NodeBase, label: str) -> str:
    props = node.model_dump(mode="json")  # mode="json" serializes datetime to string
    query = f"CREATE (n:{label} $props) RETURN n.id AS id"
    async with self._driver.session() as session:
      result = await session.run(query, props=props)
      record = await result.single()
      return record["id"]

  async def get_node(self, node_id: str) -> NodeBase | None:
    query = "MATCH (n {id: $id}) RETURN n"
    async with self._driver.session() as session:
      result = await session.run(query, id=node_id)
      record = await result.single()
      return self._to_model(record["n"]) if record else None

  async def update_node(self, node_id: str, changes: dict) -> None:
    query = "MATCH (n {id: $id}) SET n += $changes"
    async with self._driver.session() as session:
      await session.run(query, id=node_id, changes=changes)

  async def create_edge(self, from_id: str, to_id: str, edge_type: str) -> None:
    query = (
      f"MATCH (a {{id: $from_id}}), (b {{id: $to_id}}) "
      f"CREATE (a)-[:{edge_type}]->(b)"
    )
    async with self._driver.session() as session:
      await session.run(query, from_id=from_id, to_id=to_id)

  async def query_nodes(self, label: str, filters: dict) -> list[NodeBase]:
    where = " AND ".join(f"n.{k} = ${k}" for k in filters)
    query = f"MATCH (n:{label})"
    if where:
      query += f" WHERE {where}"
    query += " RETURN n"
    async with self._driver.session() as session:
      result = await session.run(query, **filters)
      return [self._to_model(record["n"]) async for record in result]

  async def get_neighbors(
    self,
    node_id: str,
    edge_type: str,
    direction: str = "out", # "out" or "in"
    target_label: str | None = None,
  ) -> list[NodeBase]:
    # follow one edge_type from node_id in the given direction
    if direction == "out":
      pattern = f"(n {{id: $id}})-[:{edge_type}]->(m{{label}})"
    else:
      pattern = f"(n {{id: $id}})<-[:{edge_type}]-(m{{label}})"
    pattern = pattern.replace("{label}", f":{target_label}" if target_label else "")
    query = f"MATCH {pattern} RETURN m"
    async with self._driver.session() as session:
      result = await session.run(query, id=node_id)
      return [self._to_model(record["m"]) async for record in result]

  # ---------- layer 2: domain queries ----------

  async def get_refuting_evidence(self, hypothesis_id: str) -> list[Evidence]:
    return await self.get_neighbors(
      hypothesis_id, "CONTRADICTS", direction="in", target_label="Evidence"
    )

  async def get_supporting_evidence(self, hypothesis_id: str) -> list[Evidence]:
    return await self.get_neighbors(
      hypothesis_id, "SUPPORTS", direction="in", target_label="Evidence"
    )

  async def get_investigations_of_hypothesis(self, hypothesis_id: str) -> list[Investigation]:
    return await self.get_neighbors(
      hypothesis_id, "TESTS", direction="out", target_label="Investigation"
    )

  async def get_evidence_of_investigation(self, investigation_id: str) -> list[Evidence]:
    return await self.get_neighbors(
      investigation_id, "PRODUCES", direction="out", target_label="Evidence"
    )

  async def get_pending_investigations(self, case_id: str) -> list[Investigation]:
    # carries extra logic (state filter), not just following an edge
    return await self.query_nodes(
      "Investigation", {"case_id": case_id, "status": "pending_dispatch"}
    )

  async def get_active_hypotheses(self, case_id: str) -> list[Hypothesis]:
    return await self.query_nodes(
      "Hypothesis", {"case_id": case_id, "status": "active"}
    )

  # ---------- visualization (returns plain dicts, not models) ----------

  async def get_case_subgraph(self, case_id: str) -> dict:
    query = """
    MATCH (c:Case {id: $case_id})
    OPTIONAL MATCH (s:InputSignal)-[:OPENS]->(c)
    OPTIONAL MATCH (n {case_id: $case_id})
    RETURN c AS case,
      collect(DISTINCT s) AS signals,
      collect(DISTINCT n) AS nodes
    """
    async with self._driver.session() as session:
      result = await session.run(query, case_id=case_id)
      record = await result.single()
      return {
        "case": dict(record["case"]),
        "signals": [dict(s) for s in record["signals"]],
        "nodes": [dict(n) for n in record["nodes"]],
      }