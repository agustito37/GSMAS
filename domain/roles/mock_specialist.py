from typing import cast

from core.agents.base import Agent, DeterministicRole, Reaction
from core.graph.models import Evidence, Investigation, NodeBase
from core.graph.store import EdgeSpec


class MockSpecialist(DeterministicRole):
    """Stub specialist: takes an Investigation and produces dummy Evidence, so the
    flow reaches a Verdict. No LLM, no real analysis. Real Specialists (with the
    tool-calling loop) arrive in Fase 5. Claims ANY pending Investigation: the
    Dispatcher (routing) is deferred until there are several specialists."""

    def reactions(self) -> list[Reaction]:
        trigger = ("node_created", "Investigation")
        return [Reaction({trigger}, self._claim_investigation, self._investigate)]

    async def _claim_investigation(self) -> NodeBase | None:
        return await self.store.claim("Investigation", {})

    async def _investigate(self, agent: Agent) -> None:
        investigation = cast(Investigation, agent.work)
        evidence = Evidence(
            content=f"[mock] no real analysis for: {investigation.description}",
            case_id=investigation.case_id,
        )
        await self.store.create_node(
            evidence, "Evidence", edges=[EdgeSpec("PRODUCES", investigation.id)]
        )
