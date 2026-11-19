from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.knowledge_connectors import (
    CatalogEntry,
    ConnectorError,
    DeterministicFakeConnector,
    ErrorCategory,
    FixtureExchange,
    KnowledgeSourceConnector,
    ProviderConfig,
    SourceLocator,
    StructuredQuery,
    load_fixture,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fake(configurations, kind):
    config = configurations[kind]
    if kind == "web_fallback":
        config = ProviderConfig.model_validate(config.model_copy(update={"enabled": True}))
    if kind == "guideline":
        config = ProviderConfig.model_validate(
            config.model_copy(
                update={
                    "catalog": (
                        CatalogEntry(
                            record_id="synthetic-04b1",
                            title="Synthetic fixture",
                            url="https://www.nice.org.uk/guidance/synthetic-04b1",
                        ),
                    )
                }
            )
        )
    return DeterministicFakeConnector(config, load_fixture(FIXTURES / (kind + ".json")).exchanges)


@pytest.mark.parametrize("kind", ["pubmed", "europe_pmc", "crossref", "guideline", "web_fallback"])
def test_all_adapters_same_protocol_single_page_and_transient_fetch(configurations, kind):
    connector = fake(configurations, kind)
    assert isinstance(connector, KnowledgeSourceConnector)
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert page.result == "results" and len(page.records) == 1
    assert page.admission_status == page.records[0].admission_status == "draft"
    assert page.records[0].title is not None
    assert all(response.body for response in page.responses)
    if kind == "europe_pmc":
        locator = SourceLocator(record_id="PMC999990001")
    elif kind == "web_fallback":
        locator = SourceLocator(url=page.records[0].url)
    else:
        locator = SourceLocator(record_id=page.records[0].record_id)
    document = connector.fetch(locator)
    assert document.response.byte_count == len(document.response.body)
    assert "body" not in document.model_dump_json()
    assert "SyntheticFixture" not in repr(document)
    assert document.admission_status == "draft"
    connector.transport.assert_consumed()


@pytest.mark.parametrize("kind", ["pubmed", "europe_pmc", "crossref", "guideline", "web_fallback"])
def test_fake_is_repeatable_and_does_not_traverse_pages(configurations, kind):
    first = fake(configurations, kind)
    second = fake(configurations, kind)
    query = StructuredQuery(text="synthetic fixture", page_size=1)
    a, b = first.search(query), second.search(query)
    assert a == b
    assert first.transport.calls == (2 if kind == "pubmed" else 0 if kind == "guideline" else 1)


def test_web_fallback_is_disabled_and_cannot_touch_transport(configurations):
    connector = DeterministicFakeConnector(configurations["web_fallback"], ())
    assert not connector.capabilities().search
    assert connector.healthcheck().category is ErrorCategory.DISABLED
    for call in [
        lambda: connector.search(StructuredQuery(text="fixture")),
        lambda: connector.fetch(SourceLocator(url="https://search.example.invalid/x")),
    ]:
        with pytest.raises(ConnectorError) as caught:
            call()
        assert caught.value.category is ErrorCategory.DISABLED
    assert connector.transport.calls == 0


@pytest.mark.parametrize("change", [{"text": "other query"}, {"page_size": 2}])
def test_cursor_is_bound_to_query_and_page_size(configurations, change):
    connector = fake(configurations, "crossref")
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    query = StructuredQuery(text="synthetic fixture", page_size=1, cursor=page.next_cursor)
    with pytest.raises(ValueError, match="cursor"):
        connector.search(query.model_copy(update=change))
    assert connector.transport.calls == 1


def test_cursor_cannot_cross_provider_or_config(configurations):
    source = fake(configurations, "crossref")
    cursor = source.search(StructuredQuery(text="synthetic fixture", page_size=1)).next_cursor
    target = fake(configurations, "europe_pmc")
    with pytest.raises(ValueError, match="cursor"):
        target.search(StructuredQuery(text="synthetic fixture", page_size=1, cursor=cursor))
    assert target.transport.calls == 0
    altered = cursor.model_copy(update={"config_checksum": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="cursor"):
        source.search(StructuredQuery(text="synthetic fixture", page_size=1, cursor=altered))


def test_next_page_uses_exact_cursor_and_repeated_page_remains_visible(configurations):
    tape = load_fixture(FIXTURES / "crossref.json").exchanges
    first = tape[0]
    second = first.model_copy(
        update={
            "query": tuple(
                (k, "next-synthetic-page" if k == "cursor" else v) for k, v in first.query
            )
        }
    )
    connector = DeterministicFakeConnector(configurations["crossref"], (first, second))
    query = StructuredQuery(text="synthetic fixture", page_size=1)
    page = connector.search(query)
    next_page = connector.search(query.model_copy(update={"cursor": page.next_cursor}))
    assert page.page_checksum == next_page.page_checksum
    assert next_page.next_cursor.page_index == 2
    assert connector.transport.calls == 2  # No aggregation/deduplication in 04B1.


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("pubmed", '{"esearchresult":{"count":"0","idlist":[]}}'),
        ("europe_pmc", '{"hitCount":0,"resultList":{"result":[]}}'),
        ("crossref", '{"status":"ok","message":{"total-results":0,"items":[]}}'),
    ],
)
def test_empty_is_not_error_or_medical_negative(configurations, kind, payload):
    exchange = (
        load_fixture(FIXTURES / (kind + ".json"))
        .exchanges[0]
        .model_copy(update={"body_text": payload})
    )
    connector = DeterministicFakeConnector(configurations[kind], (exchange,))
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert page.result == "empty" and not page.records and page.next_cursor is None


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"status":"ok","status":"bad"}',
        '{"status":"ok","message":null}',
        '{"n":NaN}',
        '{"status":"ok","message":{"total-results":true,"items":[]}}',
    ],
)
def test_invalid_provider_response_is_parse_error(configurations, payload):
    exchange = (
        load_fixture(FIXTURES / "crossref.json")
        .exchanges[0]
        .model_copy(update={"body_text": payload})
    )
    connector = DeterministicFakeConnector(configurations["crossref"], (exchange,))
    with pytest.raises(ConnectorError) as caught:
        connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert caught.value.category is ErrorCategory.PARSE
    assert payload not in str(caught.value)


@pytest.mark.parametrize("value", [True, "1", 1.5, 0, 101])
def test_strict_query_page_size(value):
    with pytest.raises(ValidationError):
        StructuredQuery(text="synthetic", page_size=value)


def test_construct_and_copy_cannot_bypass_input_validation(configurations):
    connector = fake(configurations, "crossref")
    invalid = StructuredQuery.model_construct(text="synthetic", page_size=True, cursor=None)
    with pytest.raises(ValidationError):
        connector.search(invalid)
    with pytest.raises(ValidationError):
        connector.fetch(SourceLocator.model_construct(record_id=None, url=None))
    assert connector.transport.calls == 0


@pytest.mark.parametrize("record", ["1,2", "0", "../x", "PMC9", "abc"])
def test_pubmed_fetch_rejects_batch_or_non_pmid(configurations, record):
    connector = fake(configurations, "pubmed")
    with pytest.raises(ValueError):
        connector.fetch(SourceLocator(record_id=record))
    assert connector.transport.calls == 0


def test_provider_failure_does_not_modify_another_connector(configurations):
    failed = fake(configurations, "crossref")
    failed.transport._exchanges.clear()
    healthy = fake(configurations, "pubmed")
    with pytest.raises(AssertionError, match="no live fallback"):
        failed.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert healthy.search(StructuredQuery(text="synthetic fixture", page_size=1)).records


def test_health_is_http_reachability_and_classifies_failure(configurations):
    config = configurations["crossref"]
    connector = DeterministicFakeConnector(
        config,
        (
            FixtureExchange(
                host="www.crossref.org",
                path="/",
                status=403,
                body_text="denied",
            ),
        ),
    )
    result = connector.healthcheck()
    assert not result.available and result.category is ErrorCategory.AUTHENTICATION


def test_transient_checksum_and_page_response_scope_cannot_be_forged(configurations):
    from packages.knowledge_connectors import SearchPage, TransientResponse

    connector = fake(configurations, "crossref")
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    original = page.responses[0]
    with pytest.raises(ValidationError):
        TransientResponse.model_validate(original.model_copy(update={"body": b"tampered"}))
    metadata = original.metadata.model_copy(update={"provider_version": "9.0.0"})
    bad = original.model_copy(update={"metadata": metadata})
    with pytest.raises(ValidationError):
        SearchPage.model_validate(page.model_copy(update={"responses": (bad,)}))


def test_configured_page_limits_apply_without_fetching_more_pages(configurations):
    config = configurations["crossref"]
    policy = config.http.model_copy(update={"max_pages": 1, "max_page_size": 1})
    config = ProviderConfig.model_validate(config.model_copy(update={"http": policy}))
    connector = DeterministicFakeConnector(
        config, load_fixture(FIXTURES / "crossref.json").exchanges[:1]
    )
    with pytest.raises(ConnectorError) as caught:
        connector.search(StructuredQuery(text="synthetic fixture", page_size=2))
    assert caught.value.category is ErrorCategory.PAGE_LIMIT
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    with pytest.raises(ConnectorError) as caught:
        connector.search(
            StructuredQuery(text="synthetic fixture", page_size=1, cursor=page.next_cursor)
        )
    assert caught.value.category is ErrorCategory.PAGE_LIMIT and connector.transport.calls == 1


def test_pubmed_summary_must_match_search_ids(configurations):
    tape = load_fixture(FIXTURES / "pubmed.json").exchanges
    wrong = tape[1].model_copy(update={"body_text": '{"result":{"uids":["999990002"]}}'})
    connector = DeterministicFakeConnector(configurations["pubmed"], (tape[0], wrong))
    with pytest.raises(ConnectorError) as caught:
        connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert caught.value.category is ErrorCategory.PARSE


def test_successful_healthcheck_is_bounded_response_not_publication(configurations):
    connector = DeterministicFakeConnector(
        configurations["crossref"],
        (
            FixtureExchange(
                host="www.crossref.org",
                path="/",
                headers=(("content-type", "text/html"),),
                body_text="<html>synthetic health fixture</html>",
            ),
        ),
    )
    health = connector.healthcheck()
    assert health.available and health.response.response_checksum.startswith("sha256:")
    assert health.response.metadata.operation == "healthcheck"
