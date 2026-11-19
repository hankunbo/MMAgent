"""Original HTTP errors reach definition retrieval without implicit retry."""

from pathlib import Path

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.knowledge_connectors import ConnectorError, ErrorCategory, load_configuration
from tests.knowledge_connectors.test_http_04b1 import client, get, response


@pytest.mark.parametrize("failure", [TimeoutError(), OSError(), response(429), response(503)])
def test_original_zero_retry_http_failure_becomes_explicit_backlog(failure):
    configs = {
        p.adapter: p
        for p in reversed(load_configuration(Path("configs/knowledge_sources.toml")).providers)
    }
    config = configs["crossref"]
    http, transport = client(
        configs, failure, changes={"http": config.http.model_copy(update={"max_retries": 0})}
    )
    with pytest.raises(ConnectorError) as caught:
        get(http)
    error = caught.value
    assert error.category is ErrorCategory.RETRY_EXHAUSTED
    assert len(transport.requests) == 1
    calls = []

    def fetch(hit):
        calls.append(hit)
        raise error

    owner = KnowledgeDefinitionSources(
        repository=None,
        store=None,
        search_public=lambda **kw: [dict(reference="confirmed-hit")],
        fetch_hit=fetch,
        failure_policy="technical_backlog_no_retry",
    )
    value = owner.search(
        query="public", key="test", limit=1, operation=lambda key, kind, call: call()
    )
    assert len(calls) == 1 and not value["materials"]
    result = value["backlog"][0]
    assert result["reason"] == "source_fetch_technical_failure"
    assert result["result"]["technical_error"] == error.last_error.value
    assert result["result"]["error_category"] == "retry_exhausted"
    assert result["result"]["retried"] is False


@pytest.mark.parametrize("category", ["security", "authentication", "configuration"])
def test_exhausted_envelope_cannot_hide_non_transient_rejection(category):
    def fetch(hit):
        raise ConnectorError(ErrorCategory.RETRY_EXHAUSTED, last_error=ErrorCategory(category))

    owner = KnowledgeDefinitionSources(
        repository=None,
        store=None,
        search_public=lambda **kw: [dict(reference="hit")],
        fetch_hit=fetch,
        failure_policy="technical_backlog_no_retry",
    )
    with pytest.raises(ConnectorError):
        owner.search(query="public", key="test", limit=1, operation=lambda key, kind, call: call())


def test_gap_loop_does_not_repeat_unhealthy_fetch_by_changing_direction(tmp_path):
    from tests.evaluation_preparation.test_gap_expansion import fixture

    expansion, calls, _ = fixture(tmp_path, second_adopts=False)
    original = expansion.sources.search

    def search(**kwargs):
        result = original(**kwargs)
        result["backlog"].append(dict(reason="source_fetch_technical_failure"))
        return result

    expansion.sources.search = search
    revision = expansion.advance()
    assert len(calls) == 1
    assert revision["gap_expansion"]["stop_reason"] == "retrieval_technical_backlog"
    assert revision["gap_expansion"]["next_action"] == "continue_existing_qualified_requirements"
    assert not revision["added"] and not (tmp_path / "cancel").exists()
