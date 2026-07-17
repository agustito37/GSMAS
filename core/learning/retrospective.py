from typing import Literal, cast

from pydantic import BaseModel

from core.graph.models import Case, Skill, Verdict
from core.roles.base import Executor, Role
from core.tools.base import ToolRegistry
from core.tools.graph_read import GraphReadTool

RETIRE_MIN_CASES = 3  # a skill is only eligible to retire after this many feedback cases

_RETRO_PROMPT = (
    "You are the retrospective of the '{role}' role, reviewing a case that just closed "
    "and whose verdict received human feedback (see the outcome).\n"
    "YOUR ROLE'S JOB: {focus}\n"
    "Distill a REUSABLE PROCEDURE for YOUR role's future work; stay STRICTLY within your "
    "role's job above. Do NOT distill another role's procedure: e.g. if your job is to "
    "plan, to hypothesize, or to weigh a verdict, do NOT describe how to query the logs "
    "(that is the investigator's job). From the case map (each node with its effort), the "
    "outcome, and your current skills (some marked as APPLIED in this case), use get_node "
    "to read the nodes that matter. Be conservative: only a clear, generalizable lesson.\n"
    "- If the verdict was CORRECT: distill the effective procedure ('create' a new one, "
    "or 'refine' an existing one to sharpen it).\n"
    "- If the verdict was WRONG: you have the richest signal. Diagnose what went wrong "
    "and 'refine' the applied procedure that misled (fix its steps, add a caveat, or "
    "tighten its 'when' so it does not fire in cases like this), and/or 'create' a "
    "corrective one.\n"
    "Fill each field with its distinct job: when = the trigger, one line (when this "
    "applies); goal = what it achieves and why it works, one line; steps = the ordered "
    "procedure; caveats = pitfalls or what to watch (especially what you just learned "
    "from a failure), omit if none; rationale = from what cases it was learned. A "
    "'refine' RE-STATES the full procedure with your change applied (goal + steps + "
    "caveats), so include what you want to keep. Return an empty list if nothing is "
    "clear."
)


class _Change(BaseModel):
    action: Literal["create", "refine"]
    skill_id: str = ""        # the existing skill, for refine
    when: str = ""            # the trigger: when this applies (becomes the summary)
    goal: str = ""            # one line: what it achieves and why it works (content head)
    steps: list[str] = []     # the ordered procedure (content body)
    caveats: list[str] = []   # pitfalls / what to watch, esp. from failures (content notes)
    rationale: str = ""       # provenance: from what cases it was learned


class _Distillation(BaseModel):
    changes: list[_Change]


async def retrospect(role: Role, agent: Executor, focus: str) -> None:
    """The reflection that writes skills, scoped to the role's `focus` (its function, so
    each role distills ITS procedure, not a generic one). Two separate parts: a
    deterministic vitality
    update (corroborate/refute the skills this case APPLIED, gated by human feedback,
    the threshold deciding retire-vs-refine), and an LLM distillation (create/refine,
    on success and failure alike). Uses agent.run_llm DIRECTLY, not role.reason: it
    reflects to WRITE skills, so it must not get reason's skill-injection loop."""
    case = cast(Case, agent.work)
    role_id = f"{case.workspace_id}:{role.name}"
    verdicts = await role.store.get_neighbors(case.id, "CONCLUDES", target_label="Verdict")
    verdict = cast(Verdict, verdicts[0]) if verdicts else None
    feedback = verdict.feedback if verdict else None
    applied = await role.store.get_case_applied_skills(case.id, role_id)
    skills = await role.store.get_active_skills(role_id)
    known = {s.id for s in skills}

    # 1. LLM distillation (learn from success AND failure; the input carries both)
    case_map = await role.store.get_case_map(case.id)
    tools = (agent.tools or ToolRegistry([])).with_tool(GraphReadTool(role.store))
    distill = await agent.run_llm(
        system=_RETRO_PROMPT.format(role=role.name, focus=focus),
        user=_format_input(case_map, verdict, skills, applied),
        schema=_Distillation,
        tools=tools,
    )

    # 2. deterministic vitality: the threshold decides retire-vs-refine
    retired: set[str] = set()
    if feedback == "incorrect":
        for skill_id in applied:
            await role.store.add_refutation(skill_id, case.id)
            support = await role.store.get_skill_support(skill_id)
            total = support["corroborations"] + support["refutations"]
            if total >= RETIRE_MIN_CASES and support["refutations"] > support["corroborations"]:
                await role.store.retire_skill(skill_id)
                retired.add(skill_id)
    elif feedback == "correct":
        for skill_id in applied:
            await role.store.add_corroboration(skill_id, case.id)
    # partial: neutral. The outcome is ambiguous (the system hedged), so vitality does
    # not move; only correct corroborates and only incorrect refutes. The LLM
    # distillation above still ran, so a partial case can still teach a lesson.

    # 3. apply the distillation (drop a refine of a skill the threshold just retired)
    for change in distill.changes:
        if change.action == "create" and change.when and change.steps:
            skill = Skill(
                role_id=role_id,
                summary=change.when,
                content=_render_content(change.goal, change.steps, change.caveats),
                rationale=change.rationale,
            )
            await role.store.create_skill(skill, case.id)
        elif (
            change.action == "refine"
            and change.skill_id in known
            and change.skill_id not in retired
        ):
            edits: dict = {}
            if change.when:
                edits["summary"] = change.when
            if change.steps:  # a refine re-states the full procedure; steps anchor it
                edits["content"] = _render_content(change.goal, change.steps, change.caveats)
            if change.rationale:
                edits["rationale"] = change.rationale
            if edits:
                await role.store.update_node(change.skill_id, edits)


def _render_content(goal: str, steps: list[str], caveats: list[str]) -> str:
    """The applied procedure the Investigator reads: a short goal, the numbered steps,
    and the pitfalls to watch. Only the present sections are rendered."""
    sections: list[str] = []
    if goal:
        sections.append(f"Goal: {goal}")
    if steps:
        numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
        sections.append(f"Steps:\n{numbered}")
    if caveats:
        notes = "\n".join(f"- {caveat}" for caveat in caveats)
        sections.append(f"Watch out:\n{notes}")
    return "\n\n".join(sections)


def _format_input(
    case_map: dict, verdict: Verdict | None, skills: list[Skill], applied: list[str]
) -> str:
    nodes = "\n".join(
        f"- {n['type']} [{n['id']}] status={n['status']} tokens={n['tokens']}: {n['label']}"
        for n in case_map["nodes"]
    )
    edges = "\n".join(f"- {e['source']} -{e['type']}-> {e['target']}" for e in case_map["edges"])
    outcome = f"{verdict.kind} (feedback: {verdict.feedback})" if verdict else "no verdict"
    if skills:
        current = "\n\n".join(
            f"[{s.id}]{' (APPLIED in this case)' if s.id in applied else ''}\n"
            f"when: {s.summary}\n{s.content}\nrationale: {s.rationale}"
            for s in skills
        )
    else:
        current = "(none yet)"
    return (
        f"CASE MAP\nnodes:\n{nodes}\n\nedges:\n{edges}\n\n"
        f"OUTCOME\n{outcome}\n\nYOUR SKILLS\n{current}"
    )
