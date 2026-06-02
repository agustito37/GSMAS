"""Structural graph invariants (§3.2 of the design).

Each invariant is a multi-node property checked by traversing the case subgraph.
The checks are expressed entirely over the GraphStore's domain API (query_nodes +
get_neighbors), never over raw Cypher: they reference ontology types ("Verdict",
"CONCLUDES"), not the backend.

They run at case closure and as a sanity check in tests. Each function returns the
list of violations found; an empty list means the invariant holds.
"""

from dataclasses import dataclass
from typing import cast

from core.graph.models import Investigation
from core.graph.store import GraphStore


@dataclass(frozen=True)
class InvariantViolation:
    invariant: str  # name of the violated invariant
    detail: str  # readable description
    node_ids: tuple[str, ...] = ()  # nodes involved

    def __str__(self) -> str:
        nodes = f" [{', '.join(self.node_ids)}]" if self.node_ids else ""
        return f"{self.invariant}: {self.detail}{nodes}"


async def check_traceability(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """Every Verdict is reachable from the Case root (CONCLUDES) and the Case from
    its InputSignal (OPENS). Guarantees that any conclusion can be reconstructed
    step by step back to the signal that opened the case."""
    violations: list[InvariantViolation] = []

    for verdict in await store.query_nodes("Verdict", {"case_id": case_id}):
        concluded_by = await store.get_neighbors(
            verdict.id, "CONCLUDES", direction="in", target_label="Case"
        )
        if not concluded_by:
            violations.append(
                InvariantViolation(
                    "traceability",
                    "Verdict not connected to the Case root by CONCLUDES",
                    (verdict.id,),
                )
            )

    opened_by = await store.get_neighbors(
        case_id, "OPENS", direction="in", target_label="InputSignal"
    )
    if not opened_by:
        violations.append(
            InvariantViolation(
                "traceability",
                "Case without an InputSignal that opens it (OPENS)",
                (case_id,),
            )
        )
    return violations


async def check_no_orphans(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """Every case node stays connected to the Case root through its expected
    upward edge. Checked per node type (equivalent to root-reachability by
    transitivity): a Hypothesis is DERIVEd by the Case, an Investigation is TESTed
    by a Hypothesis or REQUIREd by another Investigation, and an Evidence is
    PRODUCEd by an Investigation. (Verdict<->Case is covered by check_traceability;
    InputSignal carries no case_id and links via OPENS.)"""
    violations: list[InvariantViolation] = []

    for hypothesis in await store.query_nodes("Hypothesis", {"case_id": case_id}):
        if not await store.get_neighbors(
            hypothesis.id, "DERIVES", direction="in", target_label="Case"
        ):
            violations.append(
                InvariantViolation(
                    "no_orphans", "Hypothesis not derived by the Case", (hypothesis.id,)
                )
            )

    for investigation in await store.query_nodes("Investigation", {"case_id": case_id}):
        tested_by = await store.get_neighbors(
            investigation.id, "TESTS", direction="in", target_label="Hypothesis"
        )
        required_by = await store.get_neighbors(
            investigation.id, "REQUIRES", direction="in", target_label="Investigation"
        )
        if not tested_by and not required_by:
            violations.append(
                InvariantViolation(
                    "no_orphans",
                    "Investigation neither tested by a Hypothesis nor "
                    "required by another Investigation",
                    (investigation.id,),
                )
            )

    for evidence in await store.query_nodes("Evidence", {"case_id": case_id}):
        if not await store.get_neighbors(
            evidence.id, "PRODUCES", direction="in", target_label="Investigation"
        ):
            violations.append(
                InvariantViolation(
                    "no_orphans", "Evidence not produced by an Investigation", (evidence.id,)
                )
            )

    return violations


async def check_mandatory_validation(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """No Evidence that backs a hypothesis (SUPPORTS/CONTRADICTS) may remain
    without validation from a Verifier (an incoming VALIDATES edge). Validation
    evidence itself —the one with an outgoing VALIDATES— is excluded: it is not
    recursively validated.

    Interpretation: the Verdict does not link Evidence by edge (§3.1: its
    substantive content lives in attributes), so the invariant is operated over
    the evidence that backs hypotheses. It applies in full to a `confirmed`/
    `refuted` Verdict; under `inconclusive` there may be unvalidated evidence due
    to quiescence, so the caller may skip it in that case."""
    violations: list[InvariantViolation] = []

    for evidence in await store.query_nodes("Evidence", {"case_id": case_id}):
        backs_hypothesis = await store.get_neighbors(
            evidence.id, "SUPPORTS", direction="out", target_label="Hypothesis"
        ) or await store.get_neighbors(
            evidence.id, "CONTRADICTS", direction="out", target_label="Hypothesis"
        )
        if not backs_hypothesis:
            continue
        is_validation_evidence = await store.get_neighbors(
            evidence.id, "VALIDATES", direction="out", target_label="Evidence"
        )
        if is_validation_evidence:
            continue
        validated_by = await store.get_neighbors(
            evidence.id, "VALIDATES", direction="in", target_label="Evidence"
        )
        if not validated_by:
            violations.append(
                InvariantViolation(
                    "mandatory_validation",
                    "Evidence backing a hypothesis without validation from a Verifier",
                    (evidence.id,),
                )
            )
    return violations


async def check_single_case(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """Exactly one root Case per subgraph, and no REQUIRES edge crosses into an
    Investigation of another case."""
    violations: list[InvariantViolation] = []

    cases = await store.query_nodes("Case", {"case_id": case_id})
    if len(cases) != 1:
        violations.append(
            InvariantViolation(
                "single_case", f"Subgraph has {len(cases)} root Case nodes (must be exactly 1)"
            )
        )

    for investigation in await store.query_nodes("Investigation", {"case_id": case_id}):
        required = cast(
            list[Investigation],
            await store.get_neighbors(
                investigation.id, "REQUIRES", direction="out", target_label="Investigation"
            ),
        )
        for other in required:
            if other.case_id != case_id:
                violations.append(
                    InvariantViolation(
                        "single_case",
                        "REQUIRES crossing into an Investigation of another case",
                        (investigation.id, other.id),
                    )
                )
    return violations


async def check_investigation_outcome(store: GraphStore, case_id: str) -> list[InvariantViolation]:
    """An Investigation in a terminal execution state (validated/rejected) produced
    its Evidence (PRODUCES); a skipped one carries an explicit reason
    (skip_reason)."""
    violations: list[InvariantViolation] = []

    investigations = cast(
        list[Investigation],
        await store.query_nodes("Investigation", {"case_id": case_id}),
    )
    for investigation in investigations:
        if investigation.status in ("validated", "rejected"):
            produced = await store.get_neighbors(
                investigation.id, "PRODUCES", direction="out", target_label="Evidence"
            )
            if not produced:
                violations.append(
                    InvariantViolation(
                        "investigation_outcome",
                        "validated/rejected Investigation with no Evidence produced",
                        (investigation.id,),
                    )
                )
        elif investigation.status == "skipped" and not (investigation.skip_reason or "").strip():
            violations.append(
                InvariantViolation(
                    "investigation_outcome",
                    "skipped Investigation without an explicit reason (skip_reason)",
                    (investigation.id,),
                )
            )
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
