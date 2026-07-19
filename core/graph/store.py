import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, LiteralString, cast

from neo4j import AsyncGraphDatabase

from core.graph.models import (
    Case,
    CaseNode,
    Evidence,
    Hypothesis,
    Investigation,
    NodeBase,
    Skill,
    to_model,
)

logger = logging.getLogger("haive.store")

# A generic mutation callback: (event_type, node_id, node_type, payload). Using
# primitives keeps the store fully decoupled from core.events, the orchestrator
# wires this to bus.publish.
OnMutation = Callable[[str, str | None, str | None, dict], None]

@dataclass(frozen=True)
class EdgeSpec:
    """A birth edge for create_node: connects the NEW node to an EXISTING node.
    direction 'in'  -> (other)-[:edge_type]->(new)   e.g. Case -DERIVES-> new Hypothesis
    direction 'out' -> (new)-[:edge_type]->(other)   e.g. new Evidence -SUPPORTS-> Hypothesis
    """

    edge_type: str
    other_id: str
    direction: Literal["in", "out"] = "in"


class GraphStore:
    def __init__(self, uri: str, user: str, password: str, on_mutation: OnMutation | None = None):
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._on_mutation = on_mutation
        self._claim_lock = asyncio.Lock()  # serializes claims (see claim())

    async def close(self):
        await self._driver.close()

    # ---------- internal helper ----------

    def _emit(
        self,
        event_type: str,
        node_id: str | None = None,
        node_type: str | None = None,
        **payload,
    ) -> None:
        # Every graph mutation notifies. The store knows nothing about the bus; it
        # just calls the callback the orchestrator wired to bus.publish. This keeps
        # all coordination flowing through the medium (stigmergy).
        if self._on_mutation is not None:
            self._on_mutation(event_type, node_id, node_type, payload)

    def _to_model(self, neo4j_node) -> NodeBase:
        # a neo4j node carries both its properties and its labels
        props = dict(neo4j_node)
        label = list(neo4j_node.labels)[0]
        return to_model(label, props)

    # ---------- layer 1: generic primitives ----------

    async def ensure_workspace(self, workspace_id: str) -> str:
        """Find-or-create the Workspace container (keyed by its id). Idempotent.
        Infrastructure setup, not a case mutation, so it does NOT emit."""
        query = "MERGE (w:Workspace {id: $id}) RETURN w.id AS id"
        async with self._driver.session() as session:
            result = await session.run(query, id=workspace_id)
            record = await result.single()
        assert record is not None  # MERGE ... RETURN always yields exactly one row
        return record["id"]

    async def ensure_role(self, workspace_id: str, name: str, kind: str = "domain") -> str:
        """Find-or-create the Role node for (workspace, name) and its LTM (the role's
        long-term memory container - skills hang off the LTM, not the Role directly, so
        the memory can later hold more than skills). Born connected: Workspace HAS_ROLE
        Role HAS_LTM LTM. Scoped to the workspace (id = '{workspace}:{name}'), stable
        across restarts. Idempotent (MERGE). Returns the role_id. Does NOT emit."""
        role_id = f"{workspace_id}:{name}"
        ltm_id = f"{role_id}:ltm"
        query = """
            MERGE (w:Workspace {id: $ws})
            MERGE (r:Role {id: $role_id})
              ON CREATE SET r.name = $name, r.kind = $kind, r.workspace_id = $ws
            MERGE (w)-[:HAS_ROLE]->(r)
            MERGE (m:LTM {id: $ltm_id})
              ON CREATE SET m.role_id = $role_id
            MERGE (r)-[:HAS_LTM]->(m)
            RETURN r.id AS id
        """
        async with self._driver.session() as session:
            result = await session.run(
                query, ws=workspace_id, role_id=role_id, ltm_id=ltm_id, name=name, kind=kind
            )
            record = await result.single()
        assert record is not None  # MERGE ... RETURN always yields exactly one row
        return record["id"]

    async def create_node(
        self, node: NodeBase, label: str, edges: Sequence[EdgeSpec] = ()
    ) -> str:
        """Create a node AND its birth edges in ONE atomic Cypher statement, then
        emit (node_created first, then one edge_created per edge). Events therefore
        always announce a CONSISTENT state: the node exists with its edges, or not
        at all.

        Case-scoped nodes (CaseNode) MUST be born connected: passing no edges raises
        ValueError, which makes the no-orphans invariant hold by construction (the
        checker stays as defense in depth). InputSignal and the persistent nodes are
        exempt (they are not CaseNode). Also raises if an edge endpoint is missing
        (the whole statement matches endpoints first: nothing is created)."""
        if isinstance(node, CaseNode) and not edges:
            raise ValueError(
                f"{label} is case-scoped: it must be born connected (pass its birth edges)"
            )
        props = node.model_dump(mode="json")  # mode="json" serializes datetime to string
        # labels/edge_types come from the ontology (controlled), safe to interpolate
        match_parts = [f"(o{i} {{id: $oid{i}}})" for i in range(len(edges))]
        create_parts = [
            f"CREATE (o{i})-[:{e.edge_type}]->(n)"
            if e.direction == "in"
            else f"CREATE (n)-[:{e.edge_type}]->(o{i})"
            for i, e in enumerate(edges)
        ]
        query = ""
        if match_parts:
            query += "MATCH " + ", ".join(match_parts) + " "
        query += f"CREATE (n:{label} $props) " + " ".join(create_parts) + " RETURN n.id AS id"
        params: dict[str, Any] = {f"oid{i}": e.other_id for i, e in enumerate(edges)}
        async with self._driver.session() as session:
            result = await session.run(cast(LiteralString, query), props=props, **params)
            record = await result.single()
        if record is None:  # an endpoint did not exist: MATCH found nothing, nothing created
            raise ValueError(f"create_node({label}): an edge endpoint does not exist")
        self._emit("node_created", node_id=node.id, node_type=label)
        for e in edges:
            from_id, to_id = (e.other_id, node.id) if e.direction == "in" else (node.id, e.other_id)
            self._emit("edge_created", from_id=from_id, to_id=to_id, edge_type=e.edge_type)
        return node.id  # the id is generated by the model, not by Neo4j

    async def get_node(self, node_id: str) -> NodeBase | None:
        query = "MATCH (n {id: $id}) RETURN n"
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            return self._to_model(record["n"]) if record else None

    async def update_node(self, node_id: str, changes: dict) -> None:
        query = "MATCH (n {id: $id}) SET n += $changes RETURN labels(n)[0] AS label"
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id, changes=changes)
            record = await result.single()
        if record is not None:  # no row => the node did not exist; nothing mutated
            self._emit("node_updated", node_id=node_id, node_type=record["label"], changes=changes)

    async def create_edge(self, from_id: str, to_id: str, edge_type: str) -> None:
        # `edge_type` is an ontology type (controlled), so interpolating it is safe.
        query = cast(
            LiteralString,
            f"MATCH (a {{id: $from_id}}), (b {{id: $to_id}}) CREATE (a)-[:{edge_type}]->(b)",
        )
        async with self._driver.session() as session:
            await session.run(query, from_id=from_id, to_id=to_id)
        self._emit("edge_created", from_id=from_id, to_id=to_id, edge_type=edge_type)

    async def query_nodes(self, label: str, filters: dict) -> list[NodeBase]:
        where = " AND ".join(f"n.{k} = ${k}" for k in filters)
        query = f"MATCH (n:{label})"
        if where:
            query += f" WHERE {where}"
        query += " RETURN n"
        async with self._driver.session() as session:
            result = await session.run(cast(LiteralString, query), **filters)
            return [self._to_model(record["n"]) async for record in result]

    async def get_neighbors(
        self,
        node_id: str,
        edge_type: str,
        direction: str = "out",  # "out" or "in"
        target_label: str | None = None,
    ) -> list[NodeBase]:
        # follow one edge_type from node_id in the given direction
        target = f":{target_label}" if target_label else ""
        if direction == "out":
            pattern = f"(n {{id: $id}})-[:{edge_type}]->(m{target})"
        else:
            pattern = f"(n {{id: $id}})<-[:{edge_type}]-(m{target})"
        query = cast(LiteralString, f"MATCH {pattern} RETURN m")
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id)
            return [self._to_model(record["m"]) async for record in result]

    # ---------- layer 2: domain queries ----------
    # These return concrete model types. The cast is safe because each query is
    # restricted to a single label/edge that yields exactly that type.

    async def get_refuting_evidence(self, hypothesis_id: str) -> list[Evidence]:
        return cast(
            list[Evidence],
            await self.get_neighbors(
                hypothesis_id, "CONTRADICTS", direction="in", target_label="Evidence"
            ),
        )

    async def get_supporting_evidence(self, hypothesis_id: str) -> list[Evidence]:
        return cast(
            list[Evidence],
            await self.get_neighbors(
                hypothesis_id, "SUPPORTS", direction="in", target_label="Evidence"
            ),
        )

    async def get_investigations_of_hypothesis(self, hypothesis_id: str) -> list[Investigation]:
        return cast(
            list[Investigation],
            await self.get_neighbors(
                hypothesis_id, "TESTS", direction="out", target_label="Investigation"
            ),
        )

    async def get_evidence_of_investigation(self, investigation_id: str) -> list[Evidence]:
        return cast(
            list[Evidence],
            await self.get_neighbors(
                investigation_id, "PRODUCES", direction="out", target_label="Evidence"
            ),
        )

    async def claim(self, label: str, filters: dict) -> NodeBase | None:
        """Find a `label` node with claim_state='pending' matching `filters`, mark it
        'claimed', and return it (or None if none is free). Serialized by _claim_lock,
        so claims happen one at a time and two concurrent drains never take the same
        node (independent of Neo4j's locking). The only place the claim is written."""
        where = " AND ".join(f"n.{k} = ${k}" for k in filters)
        query = f"MATCH (n:{label} {{claim_state: 'pending'}})"
        if where:
            query += f" WHERE {where}"
        query += " WITH n LIMIT 1 SET n.claim_state = 'claimed' RETURN n"
        async with self._claim_lock, self._driver.session() as session:
            result = await session.run(cast(LiteralString, query), **filters)
            record = await result.single()
        return self._to_model(record["n"]) if record else None

    async def complete(self, node_id: str) -> None:
        """Mark a claimed node finished (claim_state -> 'done'): success, never
        reclaimed."""
        query = "MATCH (n {id: $id}) SET n.claim_state = 'done'"
        async with self._driver.session() as session:
            await session.run(query, id=node_id)

    async def fail(self, node_id: str, max_attempts: int) -> str:
        """Record a failed attempt: increment `attempts`; back to 'pending' (retry)
        or, at max_attempts, 'failed' (terminal, never reclaimed). Emits node_updated
        ONLY when the node goes terminal: a retry is internal bookkeeping, a terminal
        failure is an outcome other roles react to (e.g. the Aggregator closing a
        case whose last line failed). Returns the new state."""
        query = """
            MATCH (n {id: $id})
            WITH n, coalesce(n.attempts, 0) + 1 AS attempts
            SET n.attempts = attempts,
                n.claimed_by_agent_id = null,
                n.claim_state = CASE WHEN attempts >= $max THEN 'failed' ELSE 'pending' END
            RETURN n.claim_state AS state, labels(n)[0] AS label
        """
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id, max=max_attempts)
            record = await result.single()
        state = record["state"] if record else "failed"
        if record and state == "failed":
            self._emit(
                "node_updated",
                node_id=node_id,
                node_type=record["label"],
                changes={"claim_state": "failed"},
            )
            logger.warning("work unit %s exhausted its attempts -> failed", node_id[:8])
        return state

    async def record_cost(
        self, node_id: str, tokens_in: int, tokens_out: int, llm_calls: int, elapsed_ms: float
    ) -> None:
        """Accumulate one episode's cost onto its work node. Atomic increment (no
        read-modify-write race; retries add up). Does NOT emit: this is evaluation
        bookkeeping, not a domain mutation - emitting node_updated would spuriously
        wake reactions (e.g. the Aggregator on node_updated/Investigation)."""
        query = """
            MATCH (n {id: $id})
            SET n.tokens_in  = coalesce(n.tokens_in, 0)  + $tokens_in,
                n.tokens_out = coalesce(n.tokens_out, 0) + $tokens_out,
                n.llm_calls  = coalesce(n.llm_calls, 0)  + $llm_calls,
                n.elapsed_ms = coalesce(n.elapsed_ms, 0) + $elapsed_ms
        """
        async with self._driver.session() as session:
            await session.run(
                query,
                id=node_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                llm_calls=llm_calls,
                elapsed_ms=elapsed_ms,
            )

    async def recover_claimed(self, max_attempts: int) -> int:
        """Reset every node stuck in claim_state='claimed' back to 'pending' (or
        'failed' if it has exhausted its attempts), clearing the holder. Call once at
        startup: with no agents alive yet, any 'claimed' node is an orphan. Returns
        how many were touched."""
        query = """
            MATCH (n {claim_state: 'claimed'})
            WITH n, coalesce(n.attempts, 0) + 1 AS attempts
            SET n.attempts = attempts,
                n.claimed_by_agent_id = null,
                n.claim_state = CASE WHEN attempts >= $max THEN 'failed' ELSE 'pending' END
            RETURN count(n) AS recovered
        """
        async with self._driver.session() as session:
            result = await session.run(query, max=max_attempts)
            record = await result.single()
        return record["recovered"] if record else 0

    async def claim_quiescent_case(self) -> Case | None:
        query = """
            MATCH (c:Case {claim_state: 'pending'})
            WHERE NOT (c)-[:CONCLUDES]->(:Verdict)
              AND EXISTS { MATCH (h:Hypothesis {case_id: c.id}) }
              AND NOT EXISTS {
                MATCH (h:Hypothesis {case_id: c.id})
                WHERE coalesce(h.status, 'active') <> 'refuted'
                  AND NOT (h)-[:TESTS]->(:Investigation)
              }
              AND NOT EXISTS {
                MATCH (i:Investigation {case_id: c.id})
                WHERE NOT (i)-[:PRODUCES]->(:Evidence)
                  AND coalesce(i.status, '') <> 'skipped'
                  AND i.claim_state <> 'failed'
              }
              AND NOT EXISTS {
                MATCH (ev:Evidence {case_id: c.id})
                WHERE coalesce(ev.triaged, false) = false
              }
            WITH c LIMIT 1
            SET c.claim_state = 'claimed'
            RETURN c
        """
        async with self._claim_lock, self._driver.session() as session:
            result = await session.run(query)
            record = await result.single()
        return cast(Case, self._to_model(record["c"])) if record else None

    async def claim_case_for_retrospection(self, role_name: str) -> Case | None:
        """Claim a closed case (a Verdict carrying human feedback) this role has NOT
        retrospected yet, mark it RETROSPECTED from the workspace-scoped role node
        (ensured here), and return it. Serialized by the claim lock so each (role,
        case) is retrospected exactly once; survives restarts (the marker is a graph
        edge, not in-memory state)."""
        query = """
            MATCH (c:Case)-[:CONCLUDES]->(v:Verdict)
            WHERE v.feedback IS NOT NULL
              AND NOT EXISTS {
                MATCH (r:Role {id: c.workspace_id + ':' + $name})-[:RETROSPECTED]->(c)
              }
            WITH c LIMIT 1
            MERGE (r:Role {id: c.workspace_id + ':' + $name})
              ON CREATE SET r.name = $name, r.kind = 'domain', r.workspace_id = c.workspace_id
            MERGE (m:LTM {id: c.workspace_id + ':' + $name + ':ltm'})
              ON CREATE SET m.role_id = c.workspace_id + ':' + $name
            MERGE (r)-[:HAS_LTM]->(m)
            MERGE (r)-[:RETROSPECTED]->(c)
            RETURN c
        """
        async with self._claim_lock, self._driver.session() as session:
            result = await session.run(query, name=role_name)
            record = await result.single()
        return cast(Case, self._to_model(record["c"])) if record else None

    async def skip(self, node_id: str, reason: str) -> bool:
        """Terminally skip a work unit that is still PENDING, atomically: sets the
        domain outcome (status='skipped' + skip_reason) and claim_state='done' so no
        one claims it. Returns False if it was not pending anymore (already claimed,
        done or failed): an in-flight or finished unit is not silently discarded.
        EMITS node_updated: a skip is a domain outcome others react to (it can be
        the mutation that leaves a case quiescent)."""
        query = """
            MATCH (n {id: $id})
            WHERE n.claim_state = 'pending'
            SET n.status = 'skipped', n.skip_reason = $reason, n.claim_state = 'done'
            RETURN labels(n)[0] AS label
        """
        async with self._claim_lock, self._driver.session() as session:
            result = await session.run(query, id=node_id, reason=reason)
            record = await result.single()
        if record is not None:
            self._emit(
                "node_updated",
                node_id=node_id,
                node_type=record["label"],
                changes={"status": "skipped", "skip_reason": reason},
            )
            logger.info("skipped %s: %s", node_id[:8], reason)
        return record is not None

    async def create_suggested_hypothesis(
        self, hypothesis: Hypothesis, evidence_id: str, max_per_branch: int, max_per_case: int
    ) -> bool:
        """Create a GENERATED hypothesis, born connected (DERIVES from its Case and
        SUGGESTS from the evidence that inspired it), ONLY if both caps have room: its
        branch (max_per_branch) AND its case total (max_per_case). Both counts and the
        create happen in one statement under the claim lock, so concurrent generations
        cannot exceed either cap (no check-then-act race). Counts every hypothesis
        INCLUDING refuted ones (exploring a dead line still spends budget). Returns False
        if either cap was hit (nothing created)."""
        props = hypothesis.model_dump(mode="json")
        query = """
            MATCH (c:Case {id: $case_id}), (e:Evidence {id: $evidence_id})
            WITH c, e,
                 COUNT { (b:Hypothesis {root_id: $root_id}) } AS branch,
                 COUNT { (t:Hypothesis {case_id: $case_id}) } AS total
            WHERE branch < $max_branch AND total < $max_case
            CREATE (h:Hypothesis $props)
            CREATE (c)-[:DERIVES]->(h)
            CREATE (e)-[:SUGGESTS]->(h)
            RETURN h.id AS id
        """
        async with self._claim_lock, self._driver.session() as session:
            result = await session.run(
                query,
                case_id=hypothesis.case_id,
                evidence_id=evidence_id,
                root_id=hypothesis.root_id,
                max_branch=max_per_branch,
                max_case=max_per_case,
                props=props,
            )
            record = await result.single()
        if record is None:
            return False  # branch full (or endpoints missing): nothing created
        self._emit("node_created", node_id=hypothesis.id, node_type="Hypothesis")
        self._emit(
            "edge_created", from_id=hypothesis.case_id, to_id=hypothesis.id, edge_type="DERIVES"
        )
        self._emit("edge_created", from_id=evidence_id, to_id=hypothesis.id, edge_type="SUGGESTS")
        return True

    async def get_full_graph(self) -> dict:
        """Every node and edge as plain dicts (the dashboard's connect snapshot).
        Visualization only: models are not needed, and unknown/legacy properties
        must survive as-is."""
        async with self._driver.session() as session:
            result = await session.run("MATCH (n) RETURN n")
            nodes = [
                {"label": list(record["n"].labels)[0], "props": dict(record["n"])}
                async for record in result
            ]
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (a)-[r]->(b) RETURN a.id AS source, type(r) AS type, b.id AS target"
            )
            edges = [dict(record) async for record in result]
        return {"nodes": nodes, "edges": edges}

    async def get_workspace_graph(self, workspace_id: str) -> dict:
        """Every node and edge scoped to one workspace, for the dashboard's per-workspace
        view. A node belongs to the workspace if it carries workspace_id (InputSignal,
        Case, Role) or its case does (the case-scoped nodes, via case_id). Same shape as
        get_full_graph."""
        nodes_query = """
            MATCH (n)
            WHERE n.workspace_id = $ws
               OR (n:Workspace AND n.id = $ws)
               OR EXISTS { MATCH (c:Case {id: n.case_id, workspace_id: $ws}) }
               OR (n.role_id IS NOT NULL AND n.role_id STARTS WITH $prefix)
            RETURN labels(n)[0] AS label, properties(n) AS props
        """
        edges_query = """
            MATCH (a)-[r]->(b)
            WHERE (a.workspace_id = $ws OR (a:Workspace AND a.id = $ws)
                   OR EXISTS { MATCH (ca:Case {id: a.case_id, workspace_id: $ws}) }
                   OR (a.role_id IS NOT NULL AND a.role_id STARTS WITH $prefix))
              AND (b.workspace_id = $ws OR (b:Workspace AND b.id = $ws)
                   OR EXISTS { MATCH (cb:Case {id: b.case_id, workspace_id: $ws}) }
                   OR (b.role_id IS NOT NULL AND b.role_id STARTS WITH $prefix))
            RETURN a.id AS source, type(r) AS type, b.id AS target
        """
        prefix = f"{workspace_id}:"
        async with self._driver.session() as session:
            result = await session.run(nodes_query, ws=workspace_id, prefix=prefix)
            nodes = [{"label": r["label"], "props": dict(r["props"])} async for r in result]
            result = await session.run(edges_query, ws=workspace_id, prefix=prefix)
            edges = [dict(r) async for r in result]
        return {"nodes": nodes, "edges": edges}

    async def delete_workspace(self, workspace_id: str) -> int:
        """Delete a workspace and everything scoped to it: its signals and cases (with the
        case-scoped nodes), its Role/Skill/learning nodes (role_id-prefixed), and the
        Workspace node itself. Returns how many nodes were removed. The graph is the
        source of truth, so this is a genuine wipe of that partition."""
        query = """
            MATCH (n)
            WHERE n.workspace_id = $ws
               OR (n:Workspace AND n.id = $ws)
               OR EXISTS { MATCH (c:Case {id: n.case_id, workspace_id: $ws}) }
               OR (n.role_id IS NOT NULL AND n.role_id STARTS WITH $prefix)
            WITH n, n.id AS nid
            DETACH DELETE n
            RETURN count(nid) AS deleted
        """
        async with self._driver.session() as session:
            result = await session.run(query, ws=workspace_id, prefix=f"{workspace_id}:")
            record = await result.single()
        return record["deleted"] if record else 0

    async def get_case_map(self, case_id: str) -> dict:
        """A compact skeleton of a case for the retrospective to scan: per node its
        type, id, a short label, status and cost (summed token counters), plus the
        reasoning edges. Deliberately NOT the full content: the reader zooms into the
        nodes that matter with get_node, keeping its context small. Plain dicts."""
        nodes_query = """
            MATCH (c:Case {id: $case_id})
            OPTIONAL MATCH (sig:InputSignal)-[:OPENS]->(c)
            OPTIONAL MATCH (n {case_id: $case_id})
            WITH collect(DISTINCT n) + collect(DISTINCT sig) AS ns
            UNWIND ns AS x
            WITH DISTINCT x
            RETURN labels(x)[0] AS type,
                   x.id AS id,
                   coalesce(x.description, x.content, x.objective, x.raw_content, '') AS label,
                   coalesce(x.status, x.claim_state, '') AS status,
                   coalesce(x.tokens_in, 0) + coalesce(x.tokens_out, 0) AS tokens
        """
        edges_query = """
            MATCH (a)-[r]->(b)
            WHERE (a.case_id = $case_id AND b.case_id = $case_id)
               OR (a:InputSignal AND b:Case AND b.id = $case_id)
            RETURN a.id AS source, type(r) AS type, b.id AS target
        """
        async with self._driver.session() as session:
            result = await session.run(nodes_query, case_id=case_id)
            nodes = [{**dict(rec), "label": (rec["label"] or "")[:80]} async for rec in result]
            result = await session.run(edges_query, case_id=case_id)
            edges = [dict(rec) async for rec in result]
        return {"nodes": nodes, "edges": edges}

    async def create_skill(self, skill: Skill, case_id: str) -> str:
        """Create a Skill born connected: HAS_SKILL from its role's LTM (the role's
        long-term memory, created by ensure_role) and CORROBORATED_BY to the Case that
        first taught it. The (workspace, role) scope is carried by skill.role_id =
        '{workspace}:{name}'; the LTM is '{role_id}:ltm'."""
        return await self.create_node(
            skill,
            "Skill",
            edges=[
                EdgeSpec("HAS_SKILL", f"{skill.role_id}:ltm", direction="in"),
                EdgeSpec("CORROBORATED_BY", case_id, direction="out"),
            ],
        )

    async def add_corroboration(self, skill_id: str, case_id: str) -> None:
        """Link a skill to a case that corroborated it (applied + correct outcome)."""
        await self.create_edge(skill_id, case_id, "CORROBORATED_BY")

    async def add_refutation(self, skill_id: str, case_id: str) -> None:
        """Link a skill to a case that refuted it (applied + wrong outcome, blamed)."""
        await self.create_edge(skill_id, case_id, "REFUTED_BY")

    async def mark_skill_applied(self, node_id: str, skill_id: str) -> None:
        """Record that a work unit used a skill (an APPLIED edge, work -> skill).
        Idempotent (MERGE), so a retried judgment does not duplicate it. Does NOT
        emit: learning bookkeeping, not a domain mutation."""
        query = "MATCH (a {id: $a}), (s:Skill {id: $s}) MERGE (a)-[:APPLIED]->(s)"
        async with self._driver.session() as session:
            await session.run(query, a=node_id, s=skill_id)

    async def retire_skill(self, skill_id: str) -> None:
        """Take a skill out of active retrieval (kept in the graph for audit). The
        decision to call this belongs to the retrospective policy, not the store."""
        await self.update_node(skill_id, {"status": "retired"})

    async def get_active_skills(self, role_id: str) -> list[Skill]:
        """The active skills of a role (role_id = '{workspace}:{name}'), for injection
        into that role's judgments in that workspace."""
        return cast(
            list[Skill],
            await self.query_nodes("Skill", {"role_id": role_id, "status": "active"}),
        )

    async def get_skill_support(self, skill_id: str) -> dict:
        """The skill's visible track record: how many cases corroborate vs refute it.
        The retirement rule (retrospective policy) reads this; it is not stored as a
        counter, it is counted from the graph edges on demand."""
        query = """
            MATCH (s:Skill {id: $id})
            RETURN COUNT { (s)-[:CORROBORATED_BY]->(:Case) } AS corroborations,
                   COUNT { (s)-[:REFUTED_BY]->(:Case) } AS refutations
        """
        async with self._driver.session() as session:
            result = await session.run(query, id=skill_id)
            record = await result.single()
        if record is None:
            return {"corroborations": 0, "refutations": 0}
        return dict(record)

    async def get_case_applied_skills(self, case_id: str, role_id: str) -> list[str]:
        """The ids of this role's skills that were APPLIED in this case (APPLIED edges
        from its work nodes). The deterministic credit-assignment input: these are the
        skills the case's human feedback corroborates or refutes."""
        query = """
            MATCH (n {case_id: $case_id})-[:APPLIED]->(s:Skill {role_id: $role_id})
            RETURN DISTINCT s.id AS id
        """
        async with self._driver.session() as session:
            result = await session.run(query, case_id=case_id, role_id=role_id)
            return [record["id"] async for record in result]

    async def get_case_reuse(self, case_id: str) -> int:
        """How many distinct skills were APPLIED across this case (any role): the reuse
        signal that attributes an effort drop to learning rather than to luck. A
        learning read (moves with the learning module when the store is split)."""
        query = """
            MATCH (n {case_id: $case_id})-[:APPLIED]->(s:Skill)
            RETURN count(DISTINCT s) AS reuse
        """
        async with self._driver.session() as session:
            result = await session.run(query, case_id=case_id)
            record = await result.single()
        return record["reuse"] if record else 0

    async def get_case_cost(self, case_id: str) -> dict:
        """Total spend of a case subgraph, for evaluation: token/call/time sums plus
        node count, over every case-scoped node AND the InputSignal that opened it
        (the open-case episode's cost lands on the signal, one OPENS hop out). Plain
        dict (like get_full_graph): it feeds metrics/dashboards, not role reasoning."""
        query = """
            MATCH (c:Case {id: $case_id})
            OPTIONAL MATCH (s:InputSignal)-[:OPENS]->(c)
            OPTIONAL MATCH (n {case_id: $case_id})
            WITH collect(DISTINCT n) + collect(DISTINCT s) AS nodes
            UNWIND nodes AS x
            RETURN count(x) AS node_count,
                   sum(coalesce(x.tokens_in, 0))  AS tokens_in,
                   sum(coalesce(x.tokens_out, 0)) AS tokens_out,
                   sum(coalesce(x.llm_calls, 0))  AS llm_calls,
                   sum(coalesce(x.elapsed_ms, 0)) AS elapsed_ms
        """
        async with self._driver.session() as session:
            result = await session.run(query, case_id=case_id)
            record = await result.single()
        if record is None:
            return {
                "node_count": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "llm_calls": 0,
                "elapsed_ms": 0.0,
            }
        return dict(record)

