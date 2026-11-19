"""Regression counterexamples: valid nested models must not imply valid result scope."""

import pytest
from pydantic import ValidationError

from packages.knowledge_connectors import (
    DeterministicFakeConnector,
    ErrorCategory,
    FetchedDocument,
    FixtureExchange,
    HealthResult,
    RequestMetadata,
    SearchPage,
    SourceLocator,
    StructuredQuery,
)
from tests.knowledge_connectors.test_connectors_04b1 import fake

OTHER_CONFIG = "sha256:" + "f" * 64
QUERY = StructuredQuery(text="synthetic fixture", page_size=1)


def response_change(response, **changes):
    if changes.get("operation") == "summary":
        changes["search_request_checksum"] = response.metadata.request_checksum
    metadata = RequestMetadata.model_validate(response.metadata.model_copy(update=changes))
    return response.model_copy(update={"metadata": metadata})


def forged(value, changes, mode):
    if mode == "copy":
        return value.model_copy(update=changes)
    return type(value).model_construct(**(vars(value) | changes))


@pytest.mark.parametrize("mode", ["copy", "construct"])
@pytest.mark.parametrize(
    "change",
    [{"operation": "fetch"}, {"operation": "healthcheck"}, {"config_checksum": OTHER_CONFIG}],
)
def test_search_rejects_wrong_operation_or_config(configurations, mode, change):
    page = fake(configurations, "crossref").search(QUERY)
    bad = response_change(page.responses[0], **change)
    with pytest.raises(ValidationError):
        SearchPage.model_validate(forged(page, {"responses": (bad,)}, mode))


@pytest.mark.parametrize("mode", ["copy", "construct"])
@pytest.mark.parametrize("sequence", ["reversed", "summary_only", "duplicate_search"])
def test_pubmed_search_summary_sequence(configurations, mode, sequence):
    page = fake(configurations, "pubmed").search(QUERY)
    search, summary = page.responses
    responses = {
        "reversed": (summary, search),
        "summary_only": (summary,),
        "duplicate_search": (search, search),
    }[sequence]
    with pytest.raises(ValidationError):
        SearchPage.model_validate(forged(page, {"responses": responses}, mode))


@pytest.mark.parametrize("mode", ["copy", "construct"])
@pytest.mark.parametrize("sequence", ["summary_only", "search_summary"])
def test_non_pubmed_cannot_claim_summary(configurations, mode, sequence):
    page = fake(configurations, "crossref").search(QUERY)
    search = page.responses[0]
    summary = response_change(search, operation="summary")
    responses = (summary,) if sequence == "summary_only" else (search, summary)
    with pytest.raises(ValidationError):
        SearchPage.model_validate(forged(page, {"responses": responses}, mode))


@pytest.mark.parametrize("mode", ["copy", "construct"])
@pytest.mark.parametrize(
    "change",
    [
        {"operation": "search"},
        {"operation": "summary"},
        {"operation": "healthcheck"},
        {"config_checksum": OTHER_CONFIG},
    ],
)
def test_fetch_rejects_wrong_operation_or_config(configurations, mode, change):
    connector = fake(configurations, "crossref")
    page = connector.search(QUERY)
    document = connector.fetch(SourceLocator(record_id=page.records[0].record_id))
    bad = response_change(document.response, **change)
    with pytest.raises(ValidationError):
        FetchedDocument.model_validate(forged(document, {"response": bad}, mode))


def healthy(configurations):
    return DeterministicFakeConnector(
        configurations["crossref"],
        (FixtureExchange(host="www.crossref.org", path="/", body_text="{}"),),
    ).healthcheck()


@pytest.mark.parametrize("mode", ["copy", "construct"])
@pytest.mark.parametrize(
    "case",
    [
        "success_no_response",
        "success_category",
        "failure_response",
        "failure_no_category",
        "operation",
        "config",
        "provider",
        "version",
    ],
)
def test_health_result_cannot_contradict_itself_or_scope(configurations, mode, case):
    health = healthy(configurations)
    changes = {
        "success_no_response": {"response": None},
        "success_category": {"category": ErrorCategory.NETWORK},
        "failure_response": {"available": False, "category": ErrorCategory.NETWORK},
        "failure_no_category": {"available": False, "response": None},
        "operation": {"response": response_change(health.response, operation="fetch")},
        "config": {"response": response_change(health.response, config_checksum=OTHER_CONFIG)},
        "provider": {"response": response_change(health.response, provider_code="SOURCE:OTHER")},
        "version": {"response": response_change(health.response, provider_version="9.0.0")},
    }[case]
    with pytest.raises(ValidationError):
        HealthResult.model_validate(forged(health, changes, mode))


@pytest.mark.parametrize("kind", ["guideline", "crossref", "europe_pmc", "pubmed", "web_fallback"])
def test_valid_search_and_fetch_still_validate(configurations, kind):
    connector = fake(configurations, kind)
    page = connector.search(QUERY)
    assert SearchPage.model_validate(page) == page
    expected = (
        () if kind == "guideline" else ("search", "summary") if kind == "pubmed" else ("search",)
    )
    assert tuple(r.metadata.operation for r in page.responses) == expected
    locator = (
        SourceLocator(record_id="PMC999990001")
        if kind == "europe_pmc"
        else SourceLocator(url=page.records[0].url)
        if kind == "web_fallback"
        else SourceLocator(record_id=page.records[0].record_id)
    )
    document = connector.fetch(locator)
    assert FetchedDocument.model_validate(document) == document


def test_valid_health_and_zero_http_empty_catalog(configurations):
    health = healthy(configurations)
    assert HealthResult.model_validate(health) == health
    disabled = DeterministicFakeConnector(configurations["web_fallback"], ()).healthcheck()
    assert HealthResult.model_validate(disabled) == disabled
    page = DeterministicFakeConnector(configurations["guideline"], ()).search(QUERY)
    assert page.result == "empty" and page.responses == ()
    assert SearchPage.model_validate(page) == page


@pytest.mark.parametrize("part", ["record", "cursor", "result", "summary_parent"])
def test_all_page_components_bind_configuration_and_summary_parent(configurations, part):
    page = fake(configurations, "pubmed").search(QUERY)
    changes = {
        "record": {
            "records": (page.records[0].model_copy(update={"config_checksum": OTHER_CONFIG}),)
        },
        "cursor": {
            "next_cursor": page.next_cursor.model_copy(update={"config_checksum": OTHER_CONFIG})
        },
        "result": {"config_checksum": OTHER_CONFIG},
        "summary_parent": {
            "responses": (
                page.responses[0],
                response_change(page.responses[1], search_request_checksum=OTHER_CONFIG),
            )
        },
    }[part]
    with pytest.raises(ValidationError):
        SearchPage.model_validate(page.model_copy(update=changes))


def test_capabilities_bind_exact_configuration(configurations):
    config = configurations["crossref"]
    result = DeterministicFakeConnector(config, ()).capabilities()
    assert result.config_checksum == config.checksum() and result.adapter == "crossref"
    with pytest.raises(ValidationError):
        type(result).model_validate(result.model_copy(update={"config_checksum": "latest"}))
