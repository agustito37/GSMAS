from datetime import datetime
from typing import Literal, cast

from pydantic import BaseModel

from core.graph.models import Case, Hypothesis, NodeBase, Verdict
from core.graph.store import EdgeSpec
from core.learning.learning_role import LearningRole
from core.roles.base import Executor, Reaction

_PROMPT = (
    "You close a security incident case. You are given the case objective, the "
    "hypotheses that were explored, and the evidence that supports or contradicts each. "
    "Weigh the evidence and reach a verdict kind: 'confirmed' (the leading hypothesis "
    "holds / the incident is real), 'refuted' (the evidence rules it out), or "
    "'inconclusive' (the evidence does not settle it: do not force a call). Then write "
    "the verdict: content (the conclusion, citing the decisive evidence) and rationale "
    "(HOW you weighed the evidence to get there). Stay strictly grounded in the evidence "
    "given; do not invent facts."
)


class _VerdictJudgment(BaseModel):
    kind: Literal["confirmed", "refuted", "inconclusive"]
    content: str
    rationale: str


class Synthesizer(LearningRole):
    """Closes the case. Wakes on new Evidence AND on Investigation updates (a skip, or a
    terminal fail - both can be the mutation that leaves the case quiescent); its claim
    decides whether the case is actually ready (every line terminal, no verdict). The
    claim (quiescence) is structural/deterministic; the judgment (weighing the evidence
    into a verdict) is LLM, and it learns how to weigh well. Produces the Verdict and
    closes the Case."""

    name = "synthesizer"

    def reactions(self) -> list[Reaction]:
        """Wakes on every mutation class that can complete quiescence: an evidence lands
        (born WITH its edge), a skip, or a terminal fail."""
        return [
            Reaction(
                {
                    ("node_created", "Evidence"),
                    ("node_updated", "Evidence"),
                    ("node_updated", "Investigation"),
                },
                self._claim_quiescent_case,
                self._synthesize,
            )
        ]

    async def _claim_quiescent_case(self) -> NodeBase | None:
        return await self.store.claim_case_for_synthesis()

    async def _synthesize(self, agent: Executor) -> None:
        case = cast(Case, agent.work)
        judgment = await self.reason(
            agent, system=_PROMPT, user=await self._case_digest(case), schema=_VerdictJudgment
        )
        verdict = Verdict(
            kind=judgment.kind,
            content=judgment.content,
            rationale=judgment.rationale,
            case_id=case.id,
        )
        await self.store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])
        # closing is part of concluding: the Case does not stay 'active' forever
        await self.store.update_node(
            case.id, {"status": "closed", "closed_at": datetime.now().isoformat()}
        )

    async def _case_digest(self, case: Case) -> str:
        """The evidence the verdict must weigh: each hypothesis with its status and the
        evidence that supports or contradicts it. Grounds the judgment in the subgraph."""
        hypotheses = cast(
            list[Hypothesis], await self.store.query_nodes("Hypothesis", {"case_id": case.id})
        )
        lines = [f"Case objective: {case.objective}", "", "Hypotheses and their evidence:"]
        for h in hypotheses:
            lines.append(f"- [{h.status}] {h.description}")
            supporting = await self.store.get_supporting_evidence(h.id)
            refuting = await self.store.get_refuting_evidence(h.id)
            for e in supporting:
                lines.append(f"    supports: {e.content}")
            for e in refuting:
                lines.append(f"    contradicts: {e.content}")
            if not supporting and not refuting:
                lines.append("    (no evidence gathered)")
        return "\n".join(lines)
