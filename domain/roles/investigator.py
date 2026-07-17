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
    "one step into a full sweep). If a recalled skill is offered, treat it as "
    "guidance and apply only the parts relevant to this step; do not run every query "
    "it lists. Be strictly factual: report only what the tool results show; if the "
    "logs show nothing relevant, say so. Then emit your finding: content (the "
    "finding, citing the concrete log entries), rationale (WHY you conclude it from "
    "that data), and stance: 'supports' or 'contradicts' the hypothesis under test, "
    "or 'neutral'."
)


class _Finding(BaseModel):
    content: str
    rationale: str  # why the data supports this conclusion
    stance: Literal["supports", "contradicts", "neutral"]


class Investigator(LearningRole):
    """THE generic domain investigator. Claims any pending Investigation, works it
    with the common tool catalog (carried by its agents), and produces Evidence born
    with PRODUCES plus SUPPORTS/CONTRADICTS according to its own judgment of the
    finding. Learns: its investigation procedures accumulate as skills."""

    name = "investigator"

    def learning_focus(self) -> str:
        return (
            "run one investigation step with the telemetry tools and report a factual "
            "finding. Distill an INVESTIGATING procedure: which queries and sources gather "
            "the decisive evidence for this kind of step."
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

    async def _hypothesis_under_test(self, investigation_id: str) -> Hypothesis | None:
        hypotheses = await self.store.get_neighbors(
            investigation_id, "TESTS", direction="in", target_label="Hypothesis"
        )
        return cast(Hypothesis, hypotheses[0]) if hypotheses else None
