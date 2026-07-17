from dataclasses import dataclass
from typing import cast

from core.graph.models import Verdict
from core.graph.store import GraphStore


@dataclass(frozen=True)
class CaseMetrics:
    """Deterministic per-case measurement for the RQ3 curve. Structural effort is a
    node count (reproducible, independent of the model's verbosity); reuse attributes
    an effort drop to applied skills; quality is graded separately against ground
    truth by a programmatic gate (grade, in the framework), never an LLM judge."""

    hypotheses: int          # structural effort: how many hypotheses were opened
    investigations: int      # structural effort: how many investigation steps were run
    tokens_in: int
    tokens_out: int
    llm_calls: int
    reuse: int               # distinct skills applied in the case (learning attribution)
    verdict_kind: str | None  # the conclusion reached (None if the case never closed)
    feedback: str | None      # the graded outcome, once set


async def case_metrics(store: GraphStore, case_id: str) -> CaseMetrics:
    """Read the deterministic metrics of one closed case. Pure reads over the framework's
    store API (the experiment reads the graph): the case skeleton for the structural
    counts, the cost sums, the reuse count, and the verdict for the outcome."""
    case_map = await store.get_case_map(case_id)
    cost = await store.get_case_cost(case_id)
    reuse = await store.get_case_reuse(case_id)
    verdicts = await store.get_neighbors(case_id, "CONCLUDES", target_label="Verdict")
    verdict = cast(Verdict, verdicts[0]) if verdicts else None
    return CaseMetrics(
        hypotheses=sum(1 for n in case_map["nodes"] if n["type"] == "Hypothesis"),
        investigations=sum(1 for n in case_map["nodes"] if n["type"] == "Investigation"),
        tokens_in=cost["tokens_in"],
        tokens_out=cost["tokens_out"],
        llm_calls=cost["llm_calls"],
        reuse=reuse,
        verdict_kind=verdict.kind if verdict else None,
        feedback=verdict.feedback if verdict else None,
    )
