import logging
from typing import cast

from pydantic import BaseModel

from core.agents.base import Agent, Reaction, Role
from core.graph.models import Case, Evidence, Hypothesis, InputSignal, NodeBase
from core.graph.store import EdgeSpec

logger = logging.getLogger("haive.theorist")

BRANCH_LIMIT = 4  # max hypotheses per branch (root_id); the generation hard stop
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
    "piece of evidence. Judge conservatively:\n"
    "(1) new_hypotheses: ONLY if the evidence reveals a plausible explanation that "
    "no existing hypothesis (of any status) substantially covers, propose it, with "
    "a brief rationale (why the evidence suggests it). Generating is the exception: "
    "most evidence just supports or weakens existing hypotheses - then return [].\n"
    "(2) refuted: ONLY if this evidence conclusively contradicts an ACTIVE "
    "hypothesis, list its id with the rationale. If in doubt, do not refute.\n"
    "Reason only from the given content."
)


class _HypothesisOutput(BaseModel):
    description: str
    rationale: str  # why this is a plausible explanation


class _TheoristOutput(BaseModel):
    objective: str
    rationale: str  # why the case is framed this way
    hypotheses: list[_HypothesisOutput]


class _Refutation(BaseModel):
    hypothesis_id: str
    rationale: str  # why this evidence conclusively refutes it


class _TriageOutput(BaseModel):
    new_hypotheses: list[_HypothesisOutput]
    refuted: list[_Refutation]


class Theorist(Role):
    """Owner of the hypothesis space, with two reactions:
    (1) open: on a new InputSignal, open the Case and derive the initial hypotheses.
    (2) triage (the generative motor): on each new Evidence, judge whether the
        finding suggests a NEW hypothesis (SUGGESTS edge, parent's branch, max 4 per
        branch) and/or conclusively refutes an active one (refuted + skip its pending
        investigations). Marks the Evidence triaged AFTER judging: the mark's event
        re-wakes the Synthesizer, so closure cannot outrun generation.
    Singleton while the scope is one case at a time."""

    def __init__(self, store, provider) -> None:
        super().__init__(store)
        self.provider = provider
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

    async def _open_case(self, agent: Agent) -> None:
        signal = cast(InputSignal, agent.work)  # the claim only returns InputSignals
        messages = [
            {"role": "system", "content": _OPEN_PROMPT},
            {"role": "user", "content": signal.raw_content},
        ]
        response = await self.provider.complete(messages, response_schema=_TheoristOutput)
        out = _TheoristOutput.model_validate_json(response.content)

        case = Case(objective=out.objective, rationale=out.rationale, case_id="")
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

    async def _triage_evidence(self, agent: Agent) -> None:
        evidence = cast(Evidence, agent.work)
        hypotheses = cast(
            list[Hypothesis],
            await self.store.query_nodes("Hypothesis", {"case_id": evidence.case_id}),
        )
        known_ids = {h.id for h in hypotheses}
        cases = await self.store.query_nodes("Case", {"case_id": evidence.case_id})
        objective = cast(Case, cases[0]).objective if cases else ""
        parent = await self._parent_hypothesis(evidence.id)

        listing = "\n".join(
            f"- id={h.id} status={h.status}: {h.description}" for h in hypotheses
        )
        messages = [
            {"role": "system", "content": _TRIAGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Case objective: {objective}\n\nHypotheses:\n{listing}\n\n"
                    f"New evidence (from hypothesis {parent.id if parent else 'unknown'}):\n"
                    f"{evidence.content}\nWhy the analyst concluded it: {evidence.rationale}"
                ),
            },
        ]
        response = await self.provider.complete(messages, response_schema=_TriageOutput)
        out = _TriageOutput.model_validate_json(response.content)
        logger.info(
            "triage of evidence %s: %d new hypothesis(es), %d refutation(s)",
            evidence.id[:8],
            len(out.new_hypotheses),
            len(out.refuted),
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
            )
            if not created:
                logger.warning(
                    "branch %s full (%d): suggested hypothesis NOT created",
                    hypothesis.root_id[:8],
                    BRANCH_LIMIT,
                )

        for refutation in out.refuted:
            if refutation.hypothesis_id not in known_ids:
                continue  # guard against an LLM inventing ids
            await self.store.update_node(
                refutation.hypothesis_id,
                {"status": "refuted", "refutation_reason": refutation.rationale},
            )
            # the refuted line stops consuming work: skip its not-yet-claimed steps
            for investigation in await self.store.get_investigations_of_hypothesis(
                refutation.hypothesis_id
            ):
                await self.store.skip(
                    investigation.id, f"hypothesis refuted: {refutation.rationale}"
                )

        # mark AFTER judging: this update's event re-wakes the Synthesizer, so the
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
        and marks the Evidence triaged anyway - conservative: the case can close, a
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
