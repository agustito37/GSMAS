from typing import Literal, cast

from pydantic import BaseModel

from core.agents.base import Agent, Reaction, Role
from core.agents.tool_loop import run_tool_loop
from core.graph.models import Case, Evidence, Hypothesis, Investigation, NodeBase
from core.graph.store import EdgeSpec

_PROMPT = (
    "You execute ONE investigation step of an open case. Use the available tools to "
    "gather FACTS (search the telemetry with different keywords: usernames, IPs, "
    "hostnames, event types; several targeted queries beat one vague one). Be "
    "strictly factual: report only what the tool results show; if the logs show "
    "nothing relevant, say so. Then emit your finding: content (the finding, citing "
    "the concrete log entries), rationale (WHY you conclude it from that data), and "
    "stance: 'supports' or 'contradicts' the hypothesis under test, or 'neutral'."
)


class _Finding(BaseModel):
    content: str
    rationale: str  # why the data supports this conclusion
    stance: Literal["supports", "contradicts", "neutral"]


class Investigator(Role):
    """THE generic domain investigator. Claims any pending Investigation, works it
    with the common tool catalog, and produces Evidence born with PRODUCES plus
    SUPPORTS/CONTRADICTS according to its own judgment of the finding."""

    def __init__(self, store, provider, tools) -> None:
        super().__init__(store)
        self.provider = provider
        self.tools = tools

    def reactions(self) -> list[Reaction]:
        trigger = ("node_created", "Investigation")
        return [Reaction({trigger}, self._claim_investigation, self._investigate)]

    async def _claim_investigation(self) -> NodeBase | None:
        return await self.store.claim("Investigation", {})

    async def _investigate(self, agent: Agent) -> None:
        investigation = cast(Investigation, agent.work)
        hypothesis = await self._hypothesis_under_test(investigation.id)
        cases = await self.store.query_nodes("Case", {"case_id": investigation.case_id})
        objective = cast(Case, cases[0]).objective if cases else ""

        messages = [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": (
                    f"Case objective: {objective}\n"
                    f"Hypothesis under test: "
                    f"{hypothesis.description if hypothesis else 'unknown'}\n"
                    f"Your investigation step: {investigation.description}"
                ),
            },
        ]
        response = await run_tool_loop(
            self.provider, self.tools, agent, messages, response_schema=_Finding
        )
        finding = _Finding.model_validate_json(response.content)

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
