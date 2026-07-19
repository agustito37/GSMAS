from typing import Literal, cast

from pydantic import BaseModel

from core.graph.models import Case, Evidence, Hypothesis, Investigation, NodeBase
from core.graph.store import EdgeSpec
from core.learning.learning_role import LearningRole
from core.roles.base import Executor, Reaction

_PROMPT = (
    "You execute ONE investigation step of an open case: gather the facts for THIS "
    "step only, not the whole investigation. Use the available tools to gather FACTS "
    "(search the telemetry with different keywords: usernames, IPs, hostnames, event "
    "types; a few targeted queries for this step beat one vague one, but do not turn "
    "one step into a full sweep). If a recalled skill is offered, treat it as guidance "
    "and apply only the parts relevant to this step. Be strictly factual: report only "
    "what the tool results show; if the logs show nothing relevant, say so. Judge what "
    "the data MEANS, not its surface: a user DENYING an action they are recorded "
    "taking, or denying a change they did not authorize, SUPPORTS compromise, it does "
    "not contradict it.\n"
    "Emit: content (citing the concrete log entries), rationale (WHY you conclude it), "
    "stance ('supports' / 'contradicts' / 'neutral' toward the hypothesis), and "
    "disposition: weighing ALL the evidence on the hypothesis (the prior evidence shown "
    "PLUS your finding), does it DIRECTLY CONTRADICT the hypothesis ('refuted'), or not "
    "('open')? Refute ONLY on a direct factual contradiction (the hypothesis asserts X "
    "and the evidence proves NOT X); do NOT refute because the evidence is merely weak "
    "or also fits another explanation. You do NOT confirm here: whether a surviving "
    "hypothesis is the answer is decided against the alternatives when the case closes."
)


class _Finding(BaseModel):
    content: str
    rationale: str  # why the data supports this conclusion
    stance: Literal["supports", "contradicts", "neutral"]
    disposition: Literal["refuted", "open"]  # a direct contradiction refutes; else open


class Investigator(LearningRole):
    """THE generic domain investigator. Claims any pending Investigation, works it
    with the common tool catalog (carried by its agents), and produces Evidence born
    with PRODUCES plus SUPPORTS/CONTRADICTS according to its own judgment of the
    finding. Refutation is local: a direct contradiction refutes the hypothesis on the
    spot (and stops its dead line). It does NOT confirm: crowning the winning hypothesis
    is comparative and happens when the case closes. Learns: its investigation
    procedures accumulate as skills."""

    name = "investigator"

    def learning_focus(self) -> str:
        return (
            "run one investigation step, gather the facts and judge what they MEAN for "
            "the hypothesis (the stance, and whether the evidence DIRECTLY contradicts "
            "it). Distill an INVESTIGATING procedure: which queries and sources gather "
            "the decisive evidence for this kind of step, and how to read what a finding "
            "implies (e.g. a subject denying an unauthorized action supports compromise)."
        )

    def reactions(self) -> list[Reaction]:
        trigger = ("node_created", "Investigation")
        return [Reaction({trigger}, self._claim_investigation, self._investigate)]

    async def _claim_investigation(self) -> NodeBase | None:
        return await self.store.claim("Investigation", {})

    async def _investigate(self, agent: Executor) -> None:
        investigation = cast(Investigation, agent.work)
        hypothesis = await self._hypothesis_under_test(investigation.id)
        cases = await self.store.query_nodes("Case", {"case_id": investigation.case_id})
        objective = cast(Case, cases[0]).objective if cases else ""

        user = (
            f"Case objective: {objective}\n"
            f"Hypothesis under test: "
            f"{hypothesis.description if hypothesis else 'unknown'}\n"
            f"{await self._prior_evidence(hypothesis)}"
            f"Your investigation step: {investigation.description}"
        )
        finding = await self.reason(agent, system=_PROMPT, user=user, schema=_Finding)

        evidence = Evidence(
            content=finding.content,
            rationale=finding.rationale,
            case_id=investigation.case_id,
        )
        edges = [EdgeSpec("PRODUCES", investigation.id)]
        if hypothesis is not None and finding.stance != "neutral":
            edge_type = "SUPPORTS" if finding.stance == "supports" else "CONTRADICTS"
            edges.append(EdgeSpec(edge_type, hypothesis.id, direction="out"))
        await self.store.create_node(evidence, "Evidence", edges=edges)

        # local refutation: a direct contradiction kills the hypothesis on the spot.
        # Confirmation is NOT decided here (it is comparative, read at case close), so a
        # supporting or neutral finding just leaves its edge and the hypothesis open.
        if hypothesis is not None and finding.disposition == "refuted":
            await self._refute(hypothesis, finding)

    async def _prior_evidence(self, hypothesis: Hypothesis | None) -> str:
        """The evidence already gathered on this hypothesis, so its disposition is judged
        on the accumulated picture, not on this one finding alone."""
        if hypothesis is None:
            return ""
        supporting = await self.store.get_supporting_evidence(hypothesis.id)
        refuting = await self.store.get_refuting_evidence(hypothesis.id)
        if not supporting and not refuting:
            return "Prior evidence on this hypothesis: none yet.\n"
        lines = ["Prior evidence on this hypothesis:"]
        lines += [f"  supports: {e.content}" for e in supporting]
        lines += [f"  contradicts: {e.content}" for e in refuting]
        return "\n".join(lines) + "\n"

    async def _refute(self, hypothesis: Hypothesis, finding: _Finding) -> None:
        """Refute the hypothesis and stop its dead line: skip its still-pending
        investigations (evidence-gated pruning, so the case converges instead of running
        every branch to the end). skip is a no-op on units already claimed or done, so
        the current step and finished ones are untouched."""
        await self.store.update_node(
            hypothesis.id, {"status": "refuted", "refutation_reason": finding.rationale}
        )
        for inv in await self.store.get_investigations_of_hypothesis(hypothesis.id):
            await self.store.skip(inv.id, f"hypothesis refuted: {finding.rationale}")

    async def _hypothesis_under_test(self, investigation_id: str) -> Hypothesis | None:
        hypotheses = await self.store.get_neighbors(
            investigation_id, "TESTS", direction="in", target_label="Hypothesis"
        )
        return cast(Hypothesis, hypotheses[0]) if hypotheses else None
