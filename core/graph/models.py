from datetime import datetime
from typing import Any, Literal
import uuid
from pydantic import BaseModel, Field, model_validator

class NodeBase(BaseModel):
  id: str = Field(default_factory=lambda: str(uuid.uuid4()))
  created_at: datetime = Field(default_factory=datetime.now)
  created_by_agent_id: str | None = None

# local nodes (Case-scoped)

class InputSignal(NodeBase):
  raw_content: str
  metadata: dict[str, Any] = {}

class CaseNode(NodeBase):
  case_id: str

class Case(CaseNode):
  objective: str
  context: str | None = None
  status: Literal["active", "closed", "archived"] = "active"
  closed_at: datetime | None = None

  @model_validator(mode="after")
  def _self_reference(self):
    # the Case is the root of its subgraph: case_id points to itself
    if not self.case_id:
      self.case_id = self.id
    return self

class Hypothesis(CaseNode):
  description: str
  status: Literal["active", "refuted", "confirmed"] = "active"

class Investigation(CaseNode):
    description: str
    status: Literal[
      "pending_dispatch", "blocked", "in_progress",
      "skipped", "validated", "rejected",
    ] = "pending_dispatch"
    assigned_role_id: str | None = None      # id of the Role node
    executor_agent_id: str | None = None     # id of the Agent node that executes it
    condition: str | None = None             # conditional execution (LN)
    skip_reason: str | None = None           # reason when status == "skipped"

class Evidence(CaseNode):
  content: str
  artifact_refs: list[str] = []

class Verdict(CaseNode):
  kind: Literal["confirmed", "refuted", "inconclusive"]
  content: str
  feedback: Literal["correct", "incorrect", "partial"] | None = None

# global nodes (System-scoped)

class Role(NodeBase):
  name: str
  kind: Literal["domain", "system"]
  agent_type: Literal["llm", "deterministic"]
  knowledge_tools: list[str] = []
  operational_tools: list[str] = []
  prompt_template: str | None = None

class LTM(NodeBase):
  role_id: str

class Skill(NodeBase):
  role_id: str
  summary: str # for indexing retrieval
  content: str

class Agent(NodeBase):
  role_id: str
  type: Literal["llm", "human"]
  status: Literal["idle", "working", "terminated"] = "idle"


LABEL_TO_MODEL: dict[str, type[NodeBase]] = {
  "InputSignal": InputSignal,
  "Case": Case,
  "Hypothesis": Hypothesis,
  "Investigation": Investigation,
  "Evidence": Evidence,
  "Verdict": Verdict,
  "Role": Role,
  "LTM": LTM,
  "Skill": Skill,
  "Agent": Agent,
}

def to_model(label: str, props: dict) -> NodeBase:
    return LABEL_TO_MODEL[label](**props)
