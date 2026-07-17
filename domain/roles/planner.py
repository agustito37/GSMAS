from typing import cast

from pydantic import BaseModel

from core.graph.models import Case, Hypothesis, Investigation, NodeBase
from core.graph.store import EdgeSpec
from core.learning.learning_role import LearningRole
from core.roles.base import Executor, Reaction

_PROMPT = (
    "You plan how to test ONE hypothesis in a security incident investigation. Given "
    "the case objective and the hypothesis, produce a SHORT plan of concrete, targeted "
    "investigation steps (2-4): each a specific thing to check in the available "
    "telemetry that would confirm or refute the hypothesis (a particular log query, a "
    "specific artifact or entity to look up). For each step: description (the concrete "
    "action) and rationale (why it tests this hypothesis). Prefer a few sharp, "
    "discriminating steps over many vague ones."
)


class _Step(BaseModel):
    description: str  # the concrete investigation action
    rationale: str    # why this step tests the hypothesis


class _Plan(BaseModel):
    steps: list[_Step]


class Planner(LearningRole):
    """Plans how to test each hypothesis. On a new Hypothesis it reasons a short plan of
    targeted investigation steps and materializes them as Investigations (TESTS the
    hypothesis). Claims the Hypothesis (its primary consumer) so it is planned exactly
    once. Learns: its planning procedures (a hypothesis kind -> the steps that test it
    well) accumulate as skills."""

    name = "planner"

    def learning_focus(self) -> str:
        return (
            "plan the investigation steps that test one hypothesis. Distill a PLANNING "
            "procedure: for a hypothesis like this, which targeted investigation steps "
            "discriminate it (not how to run them, not how to reach a verdict)."
        )

    def reactions(self) -> list[Reaction]:
        return [Reaction({("node_created", "Hypothesis")}, self._claim_hypothesis, self._plan)]

    async def _claim_hypothesis(self) -> NodeBase | None:
        return await self.store.claim("Hypothesis", {})

    async def _plan(self, agent: Executor) -> None:
        hypothesis = cast(Hypothesis, agent.work)
        if hypothesis.status == "refuted":
            return  # refuted before planning
        cases = await self.store.query_nodes("Case", {"case_id": hypothesis.case_id})
        objective = cast(Case, cases[0]).objective if cases else ""
        user = f"Case objective: {objective}\nHypothesis to test: {hypothesis.description}"
        plan = await self.reason(agent, system=_PROMPT, user=user, schema=_Plan)

        # never leave a non-refuted hypothesis un-tested: it would block quiescence (the
        # case would never close). If the plan came back empty, fall back to one step.
        steps = plan.steps or [
            _Step(
                description=f"Investigate: {hypothesis.description}",
                rationale="direct test of the hypothesis",
            )
        ]
        for step in steps:
            investigation = Investigation(
                description=step.description,
                rationale=step.rationale,
                case_id=hypothesis.case_id,
            )
            await self.store.create_node(
                investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
            )
