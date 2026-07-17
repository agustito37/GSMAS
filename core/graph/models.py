import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class NodeBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    created_by_agent_id: str | None = None

class Claimable(BaseModel):
    """Mixin for work nodes the runtime can claim. `claim_state` is the framework's
    work lifecycle, SEPARATE from any domain status:
        pending -> free | claimed -> an agent holds it | done -> finished | failed -> gave up
    `attempts` counts failed tries; at MAX_ATTEMPTS the node goes 'failed' instead of
    back to 'pending' (no infinite retries). The GraphStore writes all of this
    atomically; roles never touch it. `claimed_by_agent_id` records the holder (for
    orphan recovery), filled once the Agent node lifecycle exists."""

    claim_state: Literal["pending", "claimed", "done", "failed"] = "pending"
    claimed_by_agent_id: str | None = None
    attempts: int = 0

class Measured(BaseModel):
    """Mixin: the cost of processing this node, accumulated by the runtime after each
    agent episode."""

    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    elapsed_ms: float = 0.0

# local nodes (Case-scoped)


class InputSignal(NodeBase, Claimable, Measured):
    raw_content: str
    workspace_id: str = "default"


class CaseNode(NodeBase):
    case_id: str


class Case(CaseNode, Claimable, Measured):
    objective: str
    workspace_id: str = "default"
    rationale: str = ""  # why the Theorist framed the case this way
    context: str | None = None
    status: Literal["active", "closed", "archived"] = "active"
    closed_at: datetime | None = None

    @model_validator(mode="after")
    def _self_reference(self):
        # the Case is the root of its subgraph: case_id points to itself
        if not self.case_id:
            self.case_id = self.id
        return self


class Hypothesis(CaseNode, Claimable, Measured):
    description: str
    rationale: str = ""  # why plausible (or why the evidence suggests it)
    status: Literal["active", "refuted", "confirmed"] = "active"
    root_id: str = ""  # initial hypothesis of its branch; generation is capped per branch
    refutation_reason: str | None = None  # the judgment that refuted it (if refuted)

    @model_validator(mode="after")
    def _own_root(self):
        # an initial hypothesis is its own root; generated ones inherit the parent's
        if not self.root_id:
            self.root_id = self.id
        return self


class Investigation(CaseNode, Claimable, Measured):
    description: str
    rationale: str = ""  # why this step tests the hypothesis
    status: Literal["blocked", "skipped", "validated", "rejected"] | None = None
    assigned_role_id: str | None = None
    condition: str | None = None
    skip_reason: str | None = None


class Evidence(CaseNode, Measured):
    content: str
    rationale: str = ""  # why the agent concluded this finding
    triaged: bool = False  # True once the Theorist judged it (generate/refute/nothing);
    artifact_refs: list[str] = []


class Verdict(CaseNode):
    kind: Literal["resolved", "unresolved"]  # did the investigation reach an answer?
    content: str  # the answer itself (what was concluded; domain-specific)
    rationale: str = ""  # how the evidence was weighed to reach it
    feedback: Literal["correct", "incorrect", "partial"] | None = None


# global nodes (System-scoped)


class Role(NodeBase):
    name: str
    workspace_id: str = ""
    kind: Literal["domain", "system"] = "domain"
    agent_type: Literal["llm", "deterministic"] | None = None


class LTM(NodeBase):
    role_id: str


class Skill(NodeBase):
    role_id: str
    summary: str  # for indexing retrieval
    content: str
    rationale: str = ""  # why the retrospective created/changed this skill
    status: Literal["active", "retired"] = "active"


class Agent(NodeBase):
    role_id: str
    type: Literal["llm", "human"]
    status: Literal["idle", "working", "terminated"] = "idle"


class Workspace(NodeBase):
    """Top-level scope: a container of cases whose per-role skills are isolated."""


LABEL_TO_MODEL: dict[str, type[NodeBase]] = {
    "InputSignal": InputSignal,
    "Case": Case,
    "Hypothesis": Hypothesis,
    "Investigation": Investigation,
    "Evidence": Evidence,
    "Verdict": Verdict,
    "Role": Role,
    "Workspace": Workspace,
    "LTM": LTM,
    "Skill": Skill,
    "Agent": Agent,
}


def to_model(label: str, props: dict) -> NodeBase:
    return LABEL_TO_MODEL[label](**props)
