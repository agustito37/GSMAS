from typing import TypeVar

from pydantic import BaseModel

from core.graph.models import NodeBase
from core.learning.recall import SkillCatalog, format_skill_index, recall_skills
from core.learning.retrospective import retrospect
from core.roles.base import Executor, Reaction, Role
from core.tools.base import ToolRegistry

T = TypeVar("T", bound=BaseModel)


class LearningRole(Role):
    """A role that learns. Subclassing this is the single, structural declaration: its
    judgments apply its learned skills (the reason override below), and at case close
    it retrospects to write/refine them (its retrospective reaction, added in the next
    step). A role either learns (subclasses this) or it does not (subclasses Role);
    both behaviors follow from this one class. It IS a Role, not a mixin: store / name
    / reactions come from Role."""

    async def reason(self, agent: Executor, *, system: str, user: str, schema: type[T]) -> T:
        """This role's judgment WITH its learned skills: recall them (by role, at this
        moment), inject their index, and give the agent get_skill to pull the ones it
        applies. Each fetched skill is recorded APPLIED on the work unit (fetched =
        used). Empty LTM -> exactly the plain judgment. The retrospective must NOT use
        this (it reflects to WRITE skills, not apply them): it calls agent.run_llm."""
        skills = await recall_skills(self, agent.work)
        if not skills:
            return await agent.run_llm(system=system, user=user, schema=schema, tools=agent.tools)
        catalog = SkillCatalog(skills)
        tools = (agent.tools or ToolRegistry([])).with_tool(catalog)
        result = await agent.run_llm(
            system=system,
            user=f"{user}\n\n{format_skill_index(skills)}",
            schema=schema,
            tools=tools,
        )
        for skill_id in catalog.fetched:  # fetched = used
            await self.store.mark_skill_applied(agent.work.id, skill_id)
        return result

    def all_reactions(self) -> list[Reaction]:
        """This role's own reactions PLUS its retrospective, auto-attached: subclassing
        LearningRole brings the reflection with no wiring by the concrete role. The
        retrospective wakes when a Verdict gains human feedback (node_updated/Verdict)
        and claims the closed case it has not reflected on yet."""
        return self.reactions() + [
            Reaction(
                {("node_updated", "Verdict")},
                self._claim_for_retrospection,
                self._retrospect,
            )
        ]

    async def _claim_for_retrospection(self) -> NodeBase | None:
        return await self.store.claim_case_for_retrospection(self.name)

    async def _retrospect(self, agent: Executor) -> None:
        await retrospect(self, agent)
