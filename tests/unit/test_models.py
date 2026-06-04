"""Unit test for the graph node models.

Run with:
    uv run pytest
"""

import pytest

from core.graph.models import Case, Hypothesis, to_model


@pytest.mark.unit
def test_case_backfills_case_id_with_its_own_id():
    """A Case is the root of its own subgraph: when case_id is left empty, the
    model validator backfills it with the node's own id (see models.Case)."""
    case = Case(objective="investigate the input signal", case_id="")

    assert case.case_id == case.id


@pytest.mark.unit
def test_to_model():
    """to_model maps a label + properties to an instance of the right model type."""
    node = to_model("Hypothesis", {"description": "a candidate explanation", "case_id": "c1"})

    assert isinstance(node, Hypothesis)
    assert node.description == "a candidate explanation"
    assert node.case_id == "c1"
