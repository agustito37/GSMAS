from typing import Literal


def grade(
    verdict_kind: str | None, expected_kind: str
) -> Literal["correct", "incorrect", "partial"]:
    """The programmatic feedback source, par of the human one (the dashboard control):
    given the verdict's kind and a KNOWN outcome, emit the same `feedback` the
    retrospective consumes. NO LLM judge - a deterministic, reproducible comparison. A
    hedged 'inconclusive' when the truth was definite is partial credit (not wrong, not
    right); a committed wrong kind is incorrect; no verdict at all is incorrect. The
    known outcome is passed in (`expected_kind`), so the framework depends on no
    dataset; an experiment supplies it from its ground truth."""
    if verdict_kind is None:
        return "incorrect"
    if verdict_kind == expected_kind:
        return "correct"
    if verdict_kind == "inconclusive":
        return "partial"
    return "incorrect"
