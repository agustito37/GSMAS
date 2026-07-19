from datetime import datetime
from typing import cast

from pydantic import BaseModel

from core.graph.models import Case, Hypothesis, NodeBase, Verdict
from core.graph.store import EdgeSpec
from core.learning.learning_role import LearningRole
from core.roles.base import Executor, Reaction

_PROMPT = (
    "You conclude a security incident case. You are given the case objective and the "
    "SURVIVING hypotheses (those not refuted, each with an id) with the evidence that "
    "supports or contradicts each. Pick the winner: the ONE hypothesis the evidence "
    "most strongly and DISCRIMINATINGLY supports (winner_id), the answer to the "
    "objective. Discriminating means the evidence distinguishes it from the "
    "alternatives, not merely that it is consistent with it: a login from an unusual "
    "place is consistent with BOTH travel and takeover and so discriminates neither, "
    "whereas a denial of unauthorized changes plus no travel authorization "
    "discriminates takeover. If no surviving hypothesis is a clear discriminating "
    "winner, leave winner_id empty: the case is unresolved, do not force a call. Then "
    "write the verdict: content (the answer itself: what happened and whether it is "
    "malicious, citing the decisive evidence) and rationale (HOW you weighed the "
    "evidence). Stay strictly grounded in the evidence given; do not invent facts."
)


class _Verdict(BaseModel):
    winner_id: str  # the surviving hypothesis the evidence most supports; "" if none wins
    content: str
    rationale: str


class Aggregator(LearningRole):
    """Concludes the case WITHOUT coordinating: it does not assign the hypotheses'
    disposition (the investigators refute locally as evidence lands). It aggregates the
    surviving hypotheses and their evidence into the case answer, reading which
    surviving trail the evidence most strongly and discriminatingly supports, marking
    that one confirmed (the winner) and writing the Verdict. Wakes on the mutation
    classes that can complete quiescence (new Evidence, a skip, a terminal fail); its
    claim decides whether the case is actually ready (every line terminal, no verdict).
    Learns how to read the surviving evidence into a verdict."""

    name = "aggregator"

    def learning_focus(self) -> str:
        return (
            "read the surviving hypotheses and their evidence and conclude the case: "
            "which one the evidence most discriminatingly supports. Distill a WEIGHING "
            "procedure: what makes evidence decisive for a case like this, and what "
            "traps to avoid (e.g. evidence merely CONSISTENT with a hypothesis does not "
            "discriminate it from the alternatives), not how to gather evidence."
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
                self._conclude,
            )
        ]

    async def _claim_quiescent_case(self) -> NodeBase | None:
        return await self.store.claim_quiescent_case()

    async def _conclude(self, agent: Executor) -> None:
        case = cast(Case, agent.work)
        hypotheses = cast(
            list[Hypothesis], await self.store.query_nodes("Hypothesis", {"case_id": case.id})
        )
        survivors = [h for h in hypotheses if h.status != "refuted"]
        judgment = await self.reason(
            agent, system=_PROMPT, user=await self._case_digest(case, survivors), schema=_Verdict
        )
        # the winner emerges here: among the survivors, the one the evidence most
        # discriminatingly supports. Marking it confirmed is a READING of the strongest
        # surviving trail, not a top-down disposition pass (the refutations were local).
        winner = next((h for h in survivors if h.id == judgment.winner_id), None)
        verdict = Verdict(
            kind="resolved" if winner else "unresolved",
            content=judgment.content,
            rationale=judgment.rationale,
            case_id=case.id,
        )
        await self.store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])
        if winner is not None:
            await self.store.update_node(winner.id, {"status": "confirmed"})
        # closing is part of concluding: the Case does not stay 'active' forever
        await self.store.update_node(
            case.id, {"status": "closed", "closed_at": datetime.now().isoformat()}
        )

    async def _case_digest(self, case: Case, survivors: list[Hypothesis]) -> str:
        """The surviving hypotheses (each by id) with the evidence that supports or
        contradicts them: the material the winner is read from. Refuted hypotheses are
        omitted (the investigators already ruled them out from the evidence)."""
        lines = [f"Case objective: {case.objective}", "", "Surviving hypotheses and their evidence:"]
        for h in survivors:
            lines.append(f"- id={h.id} {h.description}")
            supporting = await self.store.get_supporting_evidence(h.id)
            refuting = await self.store.get_refuting_evidence(h.id)
            for e in supporting:
                lines.append(f"    supports: {e.content}")
            for e in refuting:
                lines.append(f"    contradicts: {e.content}")
            if not supporting and not refuting:
                lines.append("    (no evidence gathered)")
        return "\n".join(lines)
