from dataclasses import dataclass

from core.graph.store import GraphStore


_CASE_RELS = "OPENS|DERIVES|TESTS|REQUIRES|PRODUCES|SUPPORTS|CONTRADICTS|VALIDATES|CONCLUDES"


@dataclass(frozen=True)
class InvariantViolation:
    invariant: str                    # name of the violated invariant
    detail: str                       # readable description
    node_ids: tuple[str, ...] = ()    # nodes involved

    def __str__(self) -> str:
        nodes = f" [{', '.join(self.node_ids)}]" if self.node_ids else ""
        return f"{self.invariant}: {self.detail}{nodes}"

async def check_traceability(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """Every Verdict is reachable from the Case root (CONCLUDES) and the Case from
    its InputSignal (OPENS). Guarantees that any conclusion can be reconstructed
    step by step until the signal that opened the case."""
    violations: list[InvariantViolation] = []

    orphan_verdicts = await store.run_read(
        "MATCH (v:Verdict {case_id: $cid}) "
        "WHERE NOT (:Case {id: $cid})-[:CONCLUDES]->(v) "
        "RETURN v.id AS id",
        cid=case_id,
    )
    violations += [
        InvariantViolation(
            "traceability", "Verdict not connected to the Case root by CONCLUDES", (r["id"],)
        )
        for r in orphan_verdicts
    ]

    rootless_case = await store.run_read(
        "MATCH (c:Case {id: $cid}) "
        "WHERE NOT (:InputSignal)-[:OPENS]->(c) "
        "RETURN c.id AS id",
        cid=case_id,
    )
    violations += [
        InvariantViolation(
            "traceability", "Case without InputSignal that opens it (OPENS)", (r["id"],)
        )
        for r in rootless_case
    ]
    return violations

async def check_no_orphans(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """Every node of the case's subgraph (same case_id, except the Case itself) stays
    connected to the root Case through some typed domain chain. InputSignal is
    excluded because it carries no case_id (it links via OPENS, not an orphan)."""
    rows = await store.run_read(
        f"MATCH (c:Case {{id: $cid}}), (n {{case_id: $cid}}) "
        f"WHERE n.id <> c.id "
        f"  AND NOT (n)-[:{_CASE_RELS}*1..]-(c) "
        f"RETURN n.id AS id, labels(n)[0] AS label",
        cid=case_id,
    )
    return [
        InvariantViolation(
            "no_orphans", f"{r['label']} node not connected to the root Case", (r["id"],)
        )
        for r in rows
    ]

async def check_mandatory_validation(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """No Evidence that backs a hypothesis (SUPPORTS/CONTRADICTS) may remain
    without validation from a Verifier (an incoming VALIDATES edge). Validation
    evidence itself —the one with an outgoing VALIDATES— is excluded: it is not
    recursively validated.

    Interpretation: the Verdict does not link Evidence by edge (§3.1: its
    substantive content lives in attributes), so the invariant is operated over
    the evidence that backs hypotheses. It applies in full to cases with a
    `confirmed`/`refuted` Verdict; under `inconclusive` there may be unvalidated
    evidence due to quiescence, so the caller may skip it in that case."""
    rows = await store.run_read(
        "MATCH (e:Evidence {case_id: $cid}) "
        "WHERE (e)-[:SUPPORTS|CONTRADICTS]->(:Hypothesis) "
        "  AND NOT (e)-[:VALIDATES]->(:Evidence) "
        "  AND NOT (:Evidence)-[:VALIDATES]->(e) "
        "RETURN e.id AS id",
        cid=case_id,
    )
    return [
        InvariantViolation(
            "mandatory_validation",
            "Evidence backing a hypothesis without validation from a Verifier",
            (r["id"],),
        )
        for r in rows
    ]

async def check_single_case(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """Exactly one root Case per subgraph, and no REQUIRES edge crosses into an
    Investigation of another case."""
    violations: list[InvariantViolation] = []

    counted = await store.run_read(
        "MATCH (c:Case {case_id: $cid}) RETURN count(c) AS n", cid=case_id
    )
    n = counted[0]["n"] if counted else 0
    if n != 1:
        violations.append(InvariantViolation(
            "single_case", f"Subgraph has {n} root Case nodes (must be exactly 1)"
        ))

    cross = await store.run_read(
        "MATCH (a:Investigation {case_id: $cid})-[:REQUIRES]->(b:Investigation) "
        "WHERE b.case_id <> $cid "
        "RETURN a.id AS a_id, b.id AS b_id",
        cid=case_id,
    )
    violations += [
        InvariantViolation(
            "single_case",
            "REQUIRES crossing into an Investigation of another case",
            (r["a_id"], r["b_id"]),
        )
        for r in cross
    ]
    return violations

async def check_investigation_outcome(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """An Investigation in a terminal execution state (validated/rejected) produced
    its Evidence (PRODUCES); a skipped one carries an explicit reason
    (skip_reason)."""
    violations: list[InvariantViolation] = []

    no_evidence = await store.run_read(
        "MATCH (i:Investigation {case_id: $cid}) "
        "WHERE i.status IN ['validated', 'rejected'] "
        "  AND NOT (i)-[:PRODUCES]->(:Evidence) "
        "RETURN i.id AS id",
        cid=case_id,
    )
    violations += [
        InvariantViolation(
            "investigation_outcome",
            "validated/rejected Investigation with no Evidence produced",
            (r["id"],),
        )
        for r in no_evidence
    ]

    skipped_no_reason = await store.run_read(
        "MATCH (i:Investigation {case_id: $cid}) "
        "WHERE i.status = 'skipped' AND (i.skip_reason IS NULL OR i.skip_reason = '') "
        "RETURN i.id AS id",
        cid=case_id,
    )
    violations += [
        InvariantViolation(
            "investigation_outcome",
            "skipped Investigation without an explicit reason (skip_reason)",
            (r["id"],),
        )
        for r in skipped_no_reason
    ]
    return violations

_ALL_CHECKS = (
    check_traceability,
    check_no_orphans,
    check_mandatory_validation,
    check_single_case,
    check_investigation_outcome,
)

async def check_all(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """Run every invariant over the case subgraph.

    Returns the aggregated list of violations; an empty list means the subgraph
    satisfies all structural invariants. Intended for case closure (before
    archiving) and as a sanity check in integration tests."""
    violations: list[InvariantViolation] = []
    for check in _ALL_CHECKS:
        violations += await check(store, case_id)
    return violations