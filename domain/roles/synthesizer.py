from datetime import datetime
from typing import cast

from core.graph.models import Case, NodeBase, Verdict
from core.graph.store import EdgeSpec
from core.roles.base import Executor, Reaction, Role


class Synthesizer(Role):
    """Closes the case. Wakes on new Evidence AND on Investigation updates (a skip,
    or a terminal fail - both can be the mutation that leaves the case quiescent);
    its claim decides whether the case is actually ready (every line terminal, no
    verdict). Produces the Verdict and closes the Case. Deterministic for the
    skeleton (fixed kind); becomes an LLM role that reads the subgraph and reasons
    the verdict later."""

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
        verdict = Verdict(
            kind="inconclusive",  # deterministic skeleton; an LLM would reason the kind
            content="[mock] synthesized from the case evidence",
            case_id=case.id,
        )
        await self.store.create_node(verdict, "Verdict", edges=[EdgeSpec("CONCLUDES", case.id)])
        # closing is part of concluding: the Case does not stay 'active' forever
        await self.store.update_node(
            case.id, {"status": "closed", "closed_at": datetime.now().isoformat()}
        )
