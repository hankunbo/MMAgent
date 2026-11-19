"""Confirmed Source results can close locally without spending Review reservation."""

from datetime import UTC, datetime, timedelta

import pytest

from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.definition_expansion import DefinitionExpansion
from tests.evaluation_preparation.test_definition_expansion import fixture


def bounded(tmp_path):
    e = fixture(tmp_path)
    e.campaign.trial = {}
    e.campaign.protocol["max_new_requests"] = 8
    e.policy = e.policy.model_copy(update={"reserve_review_requests": 3})
    e.campaign.usage = lambda: 4 + e.budget.used
    # Configure before construction so the frozen policy binding is not stale.
    return DefinitionExpansion(
        e.campaign,
        documents=e.documents,
        sources=e.sources,
        provider=e.provider,
        budget=e.budget,
        scope=e.scope,
        policy=e.policy.model_dump(mode="json"),
    )


def test_confirmed_source_at_reserve_boundary_closes_and_resumes_without_dispatch(tmp_path):
    e = bounded(tmp_path)
    checkpoint = e.advance(until="definitions")
    assert checkpoint["accepted"] and e.budget.used == 1
    cold = DefinitionExpansion(
        e.campaign,
        documents=e.documents,
        sources=e.sources,
        provider=e.provider,
        budget=e.budget,
        scope=e.scope,
        policy=e.policy.model_dump(mode="json"),
    )
    revision = cold.advance(until="requirements")
    assert revision["added"] and e.campaign.targets == tuple(revision["added"])
    assert e.budget.used == 1 and e.sources.calls == 1
    assert e.advance() == revision


def test_saved_operation_at_reserve_is_read_without_new_claim(tmp_path):
    e = bounded(tmp_path)
    calls = []
    e.documents.write("known:result", ({"confirmed": True},))
    e.campaign.usage = lambda: 5
    assert e.operation("known", "llm", lambda: calls.append(True)) == {"confirmed": True}
    assert not calls and e.operations() == []


def test_new_dispatch_at_reserve_still_refused(tmp_path):
    e = bounded(tmp_path)
    e.campaign.usage = lambda: 5
    calls = []
    with pytest.raises(ProviderError, match="definition_review_reserve"):
        e.operation("new", "llm", lambda: calls.append(True))
    assert not calls and e.operations() == []


def test_unknown_slot_at_reserve_is_never_replayed(tmp_path):
    e = bounded(tmp_path)
    e.documents.write("definition:operation:0", ({"key": "unknown", "kind": "llm"},))
    e.campaign.usage = lambda: 5
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        e.operation("unknown", "llm", lambda: pytest.fail("must not replay"))


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_local_confirmation_does_not_bypass_current_control(tmp_path, stop):
    e = bounded(tmp_path)
    e.documents.write("known:result", ({"confirmed": True},))
    if stop == "cancel":
        (tmp_path / "cancel").touch()
    else:
        e.campaign.protocol["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ProviderError, match="cancelled|deadline"):
        e.operation("known", "llm", lambda: pytest.fail("must not replay"))
