import logging
from typing import cast

from pydantic import BaseModel

from core.graph.models import Case, Evidence, Hypothesis, InputSignal, NodeBase
from core.graph.store import EdgeSpec
from core.learning.learning_role import LearningRole
from core.roles.base import Executor, Reaction

logger = logging.getLogger("haive.proposer")

BRANCH_LIMIT = 3  # max hypotheses per branch (root_id); the generation hard stop
CASE_HYPOTHESIS_CAP = 5  # max hypotheses per case; the generative motor stops here (convergence)
TRIAGE_MAX_ATTEMPTS = 3  # failed judgments on one Evidence before giving up on it

_OPEN_PROMPT = (
    "You open an investigation from an input signal. Read the signal and produce: "
    "(1) the investigation's objective in one sentence (what must be resolved); "
    "(2) 1 to 3 candidate hypotheses (plausible explanations) in natural language; "
    "(3) for the objective and for each hypothesis, a brief rationale: WHY you "
    "concluded it from the signal. Reason only from the signal's content."
)

_TRIAGE_PROMPT = (
    "You maintain the hypothesis space of an open investigation. You receive the "
    "case objective, every existing hypothesis (id, status, description) and ONE new "
    "piece of evidence. Your only job is GENERATION: decide new_hypotheses, DEFAULT to "
    "[]. Before proposing anything, compare each candidate against EVERY hypothesis "
    "already listed: if any of them expresses the same idea, even in different words "
    "(e.g. 'account compromised', 'unauthorized access', 'account takeover' are the "
    "SAME hypothesis), do NOT propose it. Only a genuinely DISTINCT explanation the "
    "list does not already cover, with a brief rationale. Most evidence just supports "
    "or weakens existing hypotheses; then return []. Never restate a hypothesis that "
    "is already there. Do NOT judge whether a hypothesis is true or false: that "
    "follows from the evidence the investigators attach. Reason only from the given "
    "content."
)


class _HypothesisOutput(BaseModel):
    description: str
    rationale: str  # why this is a plausible explanation


class _ProposerOutput(BaseModel):
    objective: str
    rationale: str  # why the case is framed this way
    hypotheses: list[_HypothesisOutput]


class _TriageOutput(BaseModel):
    new_hypotheses: list[_HypothesisOutput]


class Proposer(LearningRole):
    """Owner of the hypothesis space, purely generative, with two reactions:
    (1) open: on a new InputSignal, open the Case and derive the initial hypotheses.
    (2) triage (the generative motor): on each new Evidence, judge whether the finding
        suggests a NEW, distinct hypothesis (SUGGESTS edge, parent's branch, capped per
        branch). It does NOT decide whether a hypothesis is true or false: that
        disposition is fixed by the Investigator that finds the deciding evidence. Marks
        the Evidence triaged AFTER judging: the mark's event re-wakes the Aggregator,
        so closure cannot outrun generation.
    Singleton while the scope is one case at a time. Learns: its hypothesizing procedure
    accumulates as skills."""

    name = "proposer"

    def learning_focus(self) -> str:
        return (
            "form the differential: the initial hypotheses from a signal, and new ones "
            "suggested by evidence. Distill a HYPOTHESIZING procedure: for signals/evidence "
            "like this, which candidate explanations to form or drop (not how to "
            "investigate or conclude them)."
        )

    def __init__(self, store) -> None:
        super().__init__(store)
        # transient in-flight dedup (unchanged comment)
        self._triaging: set[str] = set()
        self._triage_attempts: dict[str, int] = {}

    def reactions(self) -> list[Reaction]:
        return [
            Reaction({("node_created", "InputSignal")}, self._claim_signal, self._open_case),
            Reaction({("node_created", "Evidence")}, self._claim_evidence, self._triage_evidence),
        ]

    # ---- reaction 1: open the case ----

    async def _claim_signal(self) -> NodeBase | None:
        return await self.store.claim("InputSignal", {})

    async def _open_case(self, agent: Executor) -> None:
        signal = cast(InputSignal, agent.work)  # the claim only returns InputSignals
        out = await self.reason(
            agent, system=_OPEN_PROMPT, user=signal.raw_content, schema=_ProposerOutput
        )

        case = Case(
            objective=out.objective,
            rationale=out.rationale,
            case_id="",
            workspace_id=signal.workspace_id,  # inherit the workspace from the signal
        )
        await self.store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])

        for hyp in out.hypotheses:
            # initial hypothesis: its own branch root (root_id backfills to self.id)
            hypothesis = Hypothesis(
                description=hyp.description, rationale=hyp.rationale, case_id=case.id
            )
            await self.store.create_node(
                hypothesis, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)]
            )

    # ---- reaction 2: triage the evidence (the generative motor) ----

    async def _claim_evidence(self) -> NodeBase | None:
        # secondary-consumer idempotency: the durable marker is Evidence.triaged;
        # the in-memory set only guards concurrent drains within this process (the
        # check-and-add below is atomic: no await between them).
        for evidence in await self.store.query_nodes("Evidence", {"triaged": False}):
            if evidence.id not in self._triaging:
                self._triaging.add(evidence.id)
                return evidence
        return None

    async def _triage_evidence(self, agent: Executor) -> None:
        evidence = cast(Evidence, agent.work)
        hypotheses = cast(
            list[Hypothesis],
            await self.store.query_nodes("Hypothesis", {"case_id": evidence.case_id}),
        )
        # convergence: stop suggesting once the differential is broad enough (a total
        # cap per case), so the generative motor does not churn out variants forever.
        # Still mark the evidence triaged so the Aggregator's quiescence check proceeds.
        if len(hypotheses) >= CASE_HYPOTHESIS_CAP:
            await self.store.update_node(evidence.id, {"triaged": True})
            self._triaging.discard(evidence.id)
            return
        cases = await self.store.query_nodes("Case", {"case_id": evidence.case_id})
        objective = cast(Case, cases[0]).objective if cases else ""
        parent = await self._parent_hypothesis(evidence.id)

        listing = "\n".join(
            f"- id={h.id} status={h.status}: {h.description}" for h in hypotheses
        )
        out = await self.reason(
            agent,
            system=_TRIAGE_PROMPT,
            user=(
                f"Case objective: {objective}\n\nHypotheses:\n{listing}\n\n"
                f"New evidence (from hypothesis {parent.id if parent else 'unknown'}):\n"
                f"{evidence.content}\nWhy the analyst concluded it: {evidence.rationale}"
            ),
            schema=_TriageOutput,
        )
        logger.info(
            "triage of evidence %s: %d new hypothesis(es)",
            evidence.id[:8],
            len(out.new_hypotheses),
        )

        for new in out.new_hypotheses:
            hypothesis = Hypothesis(
                description=new.description,
                rationale=new.rationale,
                case_id=evidence.case_id,
                # inherit the branch of the hypothesis whose evidence suggested it
                root_id=parent.root_id if parent else "",
            )
            created = await self.store.create_suggested_hypothesis(
                hypothesis,
                evidence.id,
                BRANCH_LIMIT,
                CASE_HYPOTHESIS_CAP,
            )
            if not created:
                logger.info(
                    "cap hit (branch %d / case %d): suggested hypothesis NOT created",
                    BRANCH_LIMIT,
                    CASE_HYPOTHESIS_CAP,
                )

        # mark AFTER judging: this update's event re-wakes the Aggregator, so the
        # quiescence check always runs after the judgment (no closure race)
        await self.store.update_node(evidence.id, {"triaged": True})
        self._triaging.discard(evidence.id)

    async def _parent_hypothesis(self, evidence_id: str) -> Hypothesis | None:
        investigations = await self.store.get_neighbors(
            evidence_id, "PRODUCES", direction="in", target_label="Investigation"
        )
        if not investigations:
            return None
        hypotheses = await self.store.get_neighbors(
            investigations[0].id, "TESTS", direction="in", target_label="Hypothesis"
        )
        return cast(Hypothesis, hypotheses[0]) if hypotheses else None

    async def on_failure(self, work: NodeBase) -> None:
        """Triage recovery (first real user of this hook). Frees the in-flight slot
        so a later event retries the judgment; after TRIAGE_MAX_ATTEMPTS, gives up
        and marks the Evidence triaged anyway (conservative): the case can close, a
        missed generation is acceptable, a hung case is not."""
        if not isinstance(work, Evidence):
            return
        self._triaging.discard(work.id)
        attempts = self._triage_attempts.get(work.id, 0) + 1
        self._triage_attempts[work.id] = attempts
        if attempts >= TRIAGE_MAX_ATTEMPTS:
            logger.warning(
                "giving up on triaging evidence %s after %d attempts: marked triaged",
                work.id[:8],
                attempts,
            )
            await self.store.update_node(work.id, {"triaged": True})
