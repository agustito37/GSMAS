from core.graph.models import NodeBase, Skill
from core.roles.base import Role
from core.tools.base import Tool


async def recall_skills(role: Role, work: NodeBase) -> list[Skill]:
    """The active skills of `role` in the workspace of `work`. Empty when nothing has
    been learned yet: the judgment then runs exactly as without learning."""
    workspace = await _workspace_of(role, work)
    return await role.store.get_active_skills(f"{workspace}:{role.name}")


async def _workspace_of(role: Role, work: NodeBase) -> str:
    """The workspace a unit of work belongs to: directly if it carries one (an
    InputSignal), else via its case."""
    workspace = getattr(work, "workspace_id", None)
    if workspace:
        return workspace
    case_id = getattr(work, "case_id", None)
    if case_id:
        cases = await role.store.query_nodes("Case", {"case_id": case_id})
        if cases:
            return getattr(cases[0], "workspace_id", "default")
    return "default"


def format_skill_index(skills: list[Skill]) -> str:
    """The index injected up front: id + summary only, never the full procedure. The
    agent pulls the procedure of the ones it applies with get_skill."""
    if not skills:
        return ""
    header = (
        "You have learned procedures from past cases, indexed below by id and summary. "
        "For each one that fits this task, call get_skill(id) to read it and apply it; "
        "skip the rest."
    )
    body = "\n".join(f"- [{s.id}] {s.summary}" for s in skills)
    return f"{header}\n{body}"


class SkillCatalog(Tool):
    """The role's active skills for one judgment. The summaries are injected up front;
    the agent calls get_skill(id) to read the full procedure of the ones it applies.
    Each distinct fetch is recorded in `fetched`: fetched means used."""

    name = "get_skill"
    description = (
        "Read the full procedure of one learned skill by its id, from the indexed "
        "summaries. Call it for each skill you decide to apply to this task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "skill_id": {"type": "string", "description": "the id of the skill to read"}
        },
        "required": ["skill_id"],
    }

    def __init__(self, skills: list[Skill]) -> None:
        self._by_id = {s.id: s for s in skills}
        self.fetched: list[str] = []

    async def run(self, skill_id: str) -> str:
        skill = self._by_id.get(skill_id)
        if skill is None:
            return f"no skill with id '{skill_id}' in your catalog"
        if skill_id not in self.fetched:
            self.fetched.append(skill_id)
        return skill.content
