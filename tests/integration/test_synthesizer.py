"""Integration test for the LLM Synthesizer: it claims a quiescent case, reasons the
verdict from the evidence digest, and closes the case.

    uv run pytest -m integration
"""

import json

import pytest

from core.agents.base import Agent
from core.graph.models import Case, Evidence, Hypothesis, InputSignal, Investigation
from core.graph.store import EdgeSpec
from core.providers.base import LLMResponse
from core.tools.base import ToolRegistry
from domain.roles.synthesizer import Synthesizer
from tests.mocks.mock_provider import MockProvider


@pytest.mark.integration
async def test_synthesizer_reasons_the_verdict_from_evidence_and_closes(store):
    """A quiescent case (hypothesis tested, evidence produced and triaged): the
    Synthesizer claims it, its prompt carries the supporting evidence, it reasons the
    verdict (kind + content + rationale), and closes the case."""
    signal = InputSignal(raw_content="suspicious login for jdoe")
    await store.create_node(signal, "InputSignal")
    case = Case(objective="determine if the login is malicious", case_id="")
    await store.create_node(case, "Case", edges=[EdgeSpec("OPENS", signal.id)])
    hyp = Hypothesis(description="credential theft", case_id=case.id)
    await store.create_node(hyp, "Hypothesis", edges=[EdgeSpec("DERIVES", case.id)])
    inv = Investigation(description="check auth logs", case_id=case.id)
    await store.create_node(inv, "Investigation", edges=[EdgeSpec("TESTS", hyp.id)])
    ev = Evidence(
        content="login from Belarus at 03:14; the user normally logs in from Argentina",
        case_id=case.id,
        triaged=True,  # quiescence requires every Evidence triaged
    )
    await store.create_node(
        ev, "Evidence",
        edges=[EdgeSpec("PRODUCES", inv.id), EdgeSpec("SUPPORTS", hyp.id, direction="out")],
    )

    verdict_out = json.dumps({
        "kind": "confirmed",
        "content": "credential theft confirmed: anomalous geo-login for jdoe",
        "rationale": "the Belarus login contradicts the user's normal Argentina location",
    })
    synthesizer = Synthesizer(store)
    reaction = synthesizer.reactions()[0]
    work = await reaction.claim()  # claim_case_for_synthesis -> the quiescent case
    assert work is not None and work.id == case.id
    agent = Agent(
        synthesizer, reaction.execute, work,
        provider=MockProvider([LLMResponse(content=verdict_out)]), tools=ToolRegistry([]),
    )
    await agent.run()

    # the evidence reached the prompt (the verdict was reasoned from the digest)
    user_msg = next(m for m in agent.messages if m.get("role") == "user")
    assert "Belarus" in user_msg["content"]
    # the reasoned verdict was created and the case closed
    verdicts = await store.get_neighbors(case.id, "CONCLUDES", target_label="Verdict")
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.kind == "confirmed"
    assert "credential theft" in verdict.content
    assert verdict.rationale != ""
    closed = await store.get_node(case.id)
    assert closed is not None and closed.status == "closed"
