"""A finite public quota is not an unknown request or an empty medical result."""

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.llm_provider.provider import ProviderError
from tests.evaluation_preparation.test_definition_expansion import fixture
from tests.evaluation_preparation.test_targeted_definition_gate import targeted


@pytest.mark.parametrize("kind", ["search", "fetch"])
def test_exhausted_public_group_confirms_no_dispatch_and_is_idempotent(tmp_path, kind):
    e = targeted(fixture(tmp_path))
    for _ in range(2):
        e.reserve_wire("SOURCE:PUBLIC", kind=kind)
    calls = []
    key = "new-group"
    result = e.operation(key, kind + "_group", lambda: calls.append(True))
    assert result == dict(
        status="public_budget_exhausted",
        wire_kind=kind,
        dispatched=False,
        reason="definition_public_request_budget",
    )
    assert calls == []
    count = len(e.operations())
    assert e.operation(key, kind + "_group", lambda: calls.append(True)) == result
    assert len(e.operations()) == count and calls == []


def test_old_pending_group_is_not_reclassified_as_budget_stop(tmp_path):
    e = targeted(fixture(tmp_path))
    e.documents.write(
        "definition:operation:0",
        (
            dict(
                key="unknown",
                kind="fetch_group",
                binding=e.binding,
            ),
        ),
    )
    for _ in range(2):
        e.reserve_wire("SOURCE:PUBLIC", kind="fetch")
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        e.operation("unknown", "fetch_group", lambda: pytest.fail("must not dispatch"))
    assert e.documents.read("unknown:result") is None


def test_exhausted_fetch_returns_explicit_backlog_without_calling_connector(tmp_path):
    e = targeted(fixture(tmp_path))
    for _ in range(2):
        e.reserve_wire("SOURCE:PUBLIC", kind="fetch")
    source = KnowledgeDefinitionSources(
        repository=None,
        store=None,
        search_public=lambda **kw: [dict(reference="one"), dict(reference="two")],
        fetch_hit=lambda hit: pytest.fail("exhausted fetch must not be called"),
    )
    result = source.search(query="public", key="two", limit=2, operation=e.operation)
    assert result["materials"] == []
    assert result["backlog"][0]["reason"] == "definition_public_request_budget"
    assert result["backlog"][0]["unattempted_hits"] == [
        dict(reference="one"),
        dict(reference="two"),
    ]


def test_cancelled_group_cannot_downgrade_to_budget_stop(tmp_path):
    e = targeted(fixture(tmp_path))
    for _ in range(2):
        e.reserve_wire("SOURCE:PUBLIC", kind="fetch")
    (tmp_path / "cancel").touch()
    with pytest.raises(ProviderError, match="cancelled"):
        e.operation("new", "fetch_group", lambda: pytest.fail("must not dispatch"))
    assert e.documents.read("new:result") is None
