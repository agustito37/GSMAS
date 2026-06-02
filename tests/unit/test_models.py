"""Unit test for the graph node models.

Run with:
    uv run pytest
"""

from core.graph.models import Case


def test_case_backfills_case_id_with_its_own_id():
    """A Case is the root of its own subgraph: when case_id is left empty, the
    model validator backfills it with the node's own id (see models.Case)."""
    case = Case(objective="investigate the input signal", case_id="")

    assert case.case_id == case.id
