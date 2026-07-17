from datetime import datetime
from typing import Literal, cast

from pydantic import BaseModel

from core.graph.models import Case, Hypothesis, NodeBase, Verdict
from core.graph.store import EdgeSpec
from core.learning.learning_role import LearningRole
from core.roles.base import Executor, Reaction

_PROMPT = (
    "You close a security incident case. You are given the case objective, the "
    "hypotheses (each with an id) that were explored, and the evidence that supports "
    "or contradicts each. Weigh the evidence and decide the verdict kind. 'resolved' "
    "requires POSITIVE evidence that supports a definite answer to the objective; "
    "merely ruling out hypotheses is not a resolution, and if NO evidence supports any "
    "hypothesis the verdict is 'unresolved' (an investigation that found nothing did "
    "not settle the case: do not force a call). Then write the verdict: content (the "
    "answer itself: what happened and whether it is malicious, citing the decisive "
    "evidence) and rationale (HOW you weighed the evidence to get there). Finally set "
    "the final disposition so the graph matches your verdict, and ground each one in "
    "the evidence: in 'dispositions', CONFIRM the hypothesis your verdict rests on ONLY "
    "if evidence supports it (its id, status 'confirmed'), and REFUTE a hypothesis ONLY "
    "if evidence DIRECTLY contradicts it (its id, status 'refuted', with the reason). A "
    "hypothesis with no evidence either way is left OUT of the list and stays open; so "
    "is every hypothesis you do not rule on. If the verdict is 'unresolved', confirm "
    "nothing. Stay strictly grounded in the evidence given; do not invent facts."
)


class _Disposition(BaseModel):
    hypothesis_id: str
    status: Literal["confirmed", "refuted"]
    reason: str


class _VerdictJudgment(BaseModel):
    kind: Literal["resolved", "unresolved"]
    content: str
    rationale: str
    dispositions: list[_Disposition]


class Synthesizer(LearningRole):
    """Closes the case. Wakes on new Evidence AND on Investigation updates (a skip, or a
    terminal fail, both can be the mutation that leaves the case quiescent); its claim
    decides whether the case is actually ready (every line terminal, no verdict). The
    claim (quiescence) is structural/deterministic; the judgment (weighing the evidence
    into a verdict) is LLM, and it learns how to weigh well. Produces the Verdict and
    closes the Case."""

    name = "synthesizer"

    def learning_focus(self) -> str:
        return (
            "weigh the gathered evidence into a verdict. Distill a WEIGHING procedure: how "
            "to judge evidence like this: what is decisive, what traps to avoid (e.g. a "
            "user denying their own actions SUPPORTS compromise, it does not refute it), "
            "not how to gather evidence."
        )

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
        hypotheses = cast(
            list[Hypothesis], await self.store.query_nodes("Hypothesis", {"case_id": case.id})
        )
        judgment = await self.reason(
            agent,
            system=_PROMPT,
            user=await self._case_digest(case, hypotheses),
            schema=_VerdictJudgment,
        )
        verdict = Verdict(
            kind=judgment.kind,
            content=judgment.content,
            rationale=judgment.rationale,
            case_id=case.id,
        )
        await self.store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])
        # final disposition: the Synthesizer sees ALL the evidence, so it owns the
        # hypotheses' end state. Confirm what the verdict rests on, refute only a direct
        # contradiction; unlisted ones stay as they are (open, or an alibi the triage
        # already refuted). Nobody reacts to a Hypothesis update, so this wakes no role.
        known = {h.id for h in hypotheses}
        for d in judgment.dispositions:
            if d.hypothesis_id not in known:
                continue  # an invented id
            if d.status == "confirmed" and judgment.kind == "unresolved":
                continue  # an unresolved case confirms nothing
            edits: dict = {"status": d.status}
            if d.status == "refuted":
                edits["refutation_reason"] = d.reason
            await self.store.update_node(d.hypothesis_id, edits)
        # closing is part of concluding: the Case does not stay 'active' forever
        await self.store.update_node(
            case.id, {"status": "closed", "closed_at": datetime.now().isoformat()}
        )

    async def _case_digest(self, case: Case, hypotheses: list[Hypothesis]) -> str:
        """The evidence the verdict must weigh: each hypothesis (by id) with its status
        and the evidence that supports or contradicts it. Grounds both the verdict and
        the final disposition in the subgraph."""
        lines = [f"Case objective: {case.objective}", "", "Hypotheses and their evidence:"]
        for h in hypotheses:
            lines.append(f"- id={h.id} [{h.status}] {h.description}")
            supporting = await self.store.get_supporting_evidence(h.id)
            refuting = await self.store.get_refuting_evidence(h.id)
            for e in supporting:
                lines.append(f"    supports: {e.content}")
            for e in refuting:
                lines.append(f"    contradicts: {e.content}")
            if not supporting and not refuting:
                lines.append("    (no evidence gathered)")
        return "\n".join(lines)
