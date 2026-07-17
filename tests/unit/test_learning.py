"""Unit tests for the learning engine's retrieval helpers.

Run with:
    uv run pytest
"""

import pytest

from core.graph.models import Skill
from core.learning.recall import SkillCatalog, format_skill_index


@pytest.mark.unit
def test_format_skill_index_empty_injects_nothing():
    """No skills -> empty string, so the judgment prompt is unchanged (engine off)."""
    assert format_skill_index([]) == ""


@pytest.mark.unit
def test_format_skill_index_lists_ids_and_summaries_not_content():
    """The index carries id + summary for the agent to choose; the procedure stays out
    (it is fetched on demand)."""
    skill = Skill(
        role_id="w:investigator", summary="check MFA first", content="query mfa enrollment"
    )

    index = format_skill_index([skill])

    assert f"[{skill.id}]" in index
    assert "check MFA first" in index
    assert "query mfa enrollment" not in index


@pytest.mark.unit
async def test_skill_catalog_serves_content_and_records_the_fetch():
    """get_skill returns the procedure and records the fetch (fetched = used)."""
    skill = Skill(
        role_id="w:investigator", summary="check MFA first", content="query mfa enrollment"
    )
    catalog = SkillCatalog([skill])

    out = await catalog.run(skill.id)

    assert out == "query mfa enrollment"
    assert catalog.fetched == [skill.id]


@pytest.mark.unit
async def test_skill_catalog_unknown_id_is_text_and_not_recorded():
    """An id outside the catalog comes back as text and is not recorded as used."""
    catalog = SkillCatalog([])

    assert "no skill with id" in await catalog.run("nope")
    assert catalog.fetched == []
