from typing import cast

from core.graph.models import Hypothesis, Investigation, NodeBase
from core.graph.store import EdgeSpec
from core.roles.base import Executor, Reaction, Role


class Planner(Role):
    """Minimal InvestigationPlanner: for each Hypothesis, create one Investigation
    that tests it. Deterministic (no LLM) for the end-to-end skeleton; becomes an
    LLM role in Fase 4. Claims the Hypothesis (its primary consumer) so it is
    planned exactly once."""

    def reactions(self) -> list[Reaction]:
        return [Reaction({("node_created", "Hypothesis")}, self._claim_hypothesis, self._plan)]

    async def _claim_hypothesis(self) -> NodeBase | None:
        return await self.store.claim("Hypothesis", {})

    async def _plan(self, agent: Executor) -> None:
        hypothesis = cast(Hypothesis, agent.work)
        if hypothesis.status == "refuted":
            return  # refuted before planning
        investigation = Investigation(
            description=f"Investigate: {hypothesis.description}",
            case_id=hypothesis.case_id,
        )
        await self.store.create_node(
            investigation, "Investigation", edges=[EdgeSpec("TESTS", hypothesis.id)]
        )
