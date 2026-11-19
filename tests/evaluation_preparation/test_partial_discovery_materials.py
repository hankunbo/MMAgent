"""A bounded page limit is not a failed lookup; confirmed hits remain usable."""

import pytest

from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
from tests.evaluation_preparation.test_definition_expansion import fixture
from tests.evaluation_preparation.test_discovery_runtime_repair import resilient


@pytest.mark.parametrize("stop", ["page_limit", "hit_budget"])
def test_confirmed_bounded_partial_hits_are_consumed(tmp_path, monkeypatch, stop):
    from apps import knowledge_literature_tools

    hit = dict(reference=dict(code="SOURCE:KNOWN_HIT"), title="Public fixture")
    result = dict(
        status="partial",
        runs=[dict(outcome=dict(stop_reason=stop, error_category=None))],
        results=[dict(hits=[hit])],
    )
    monkeypatch.setattr(knowledge_literature_tools, "search", lambda *a, **k: result)
    e = resilient(fixture(tmp_path))
    connector = LiteratureDefinitionConnector(None, e, {"public": "rnfl"})
    assert connector.search(query="public", key="one", limit=1) == [hit]


def test_partial_transport_error_still_explicit_backlog(tmp_path, monkeypatch):
    from apps import knowledge_literature_tools

    result = dict(
        status="partial",
        runs=[dict(outcome=dict(stop_reason="timeout", error_category=None))],
        results=[dict(hits=[dict(title="Not assumed complete")])],
    )
    monkeypatch.setattr(knowledge_literature_tools, "search", lambda *a, **k: result)
    e = resilient(fixture(tmp_path))
    value = LiteratureDefinitionConnector(None, e, {"public": "rnfl"}).search(
        query="public", key="one", limit=1
    )
    assert value["status"] == "technical_backlog" and not value["retried"]


@pytest.mark.parametrize("category", ["timeout", "network", "rate_limit"])
def test_fulltext_technical_failure_is_recorded_without_retry(category):
    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
    from packages.knowledge_connectors.ports import ConnectorError, ErrorCategory

    calls = []

    def fetch(hit):
        calls.append(hit)
        raise ConnectorError(ErrorCategory(category))

    owner = KnowledgeDefinitionSources(
        repository=None,
        store=None,
        search_public=lambda **kw: [dict(reference="confirmed-original-hit")],
        fetch_hit=fetch,
        failure_policy="technical_backlog_no_retry",
    )
    result = owner.search(
        query="public", key="one", limit=1, operation=lambda key, kind, call: call()
    )
    assert not result["materials"] and len(calls) == 1
    assert result["backlog"][0]["reason"] == "source_fetch_technical_failure"
    assert result["backlog"][0]["result"]["technical_error"] == category


@pytest.mark.parametrize("category", ["security", "authentication"])
def test_fetch_security_failures_are_not_silently_downgraded(category):
    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
    from packages.knowledge_connectors.ports import ConnectorError, ErrorCategory

    def fetch(hit):
        raise ConnectorError(ErrorCategory(category))

    owner = KnowledgeDefinitionSources(
        repository=None,
        store=None,
        search_public=lambda **kw: [dict(reference="confirmed-original-hit")],
        fetch_hit=fetch,
        failure_policy="technical_backlog_no_retry",
    )
    with pytest.raises(ConnectorError, match=category):
        owner.search(query="public", key="one", limit=1, operation=lambda key, kind, call: call())
