"""Provider-v2.1 ranking is pinned on every page, without rewriting v2.0."""

import json
from pathlib import Path

import pytest

from packages.knowledge_connectors import (
    ConnectorError,
    DeterministicFakeConnector,
    ErrorCategory,
    ProviderConfig,
    SourceLocator,
    StructuredQuery,
    load_fixture,
)


def assembly(configurations, version):
    config = ProviderConfig.model_validate(
        configurations["crossref"].model_copy(update={"provider_version": version})
    )
    original = load_fixture(Path(__file__).parent / "fixtures/crossref.json").exchanges
    first = original[0]
    if version == "2.1.0":
        first = first.model_copy(
            update={"query": (*first.query, ("sort", "score"), ("order", "desc"))}
        )
    second = first.model_copy(
        update={
            "query": tuple(
                (key, "next-synthetic-page" if key == "cursor" else value)
                for key, value in first.query
            )
        }
    )
    return DeterministicFakeConnector(config, (first, second, original[1]))


@pytest.mark.parametrize("version", ("1.0.0", "2.0.0", "2.1.0"))
def test_ranked_or_legacy_exact_wire_parameters_across_cursor_pages(configurations, version):
    connector = assembly(configurations, version)
    query = StructuredQuery(text="synthetic fixture", page_size=1)
    first = connector.search(query)
    assert first.provider_version == version
    assert first.next_cursor.provider_version == version
    second = connector.search(query.model_copy(update={"cursor": first.next_cursor}))
    assert second.next_cursor.page_index == 2
    assert second.next_cursor.provider_version == version
    # Ranking affects discovery only: a DOI fetch remains the original request.
    document = connector.fetch(SourceLocator(record_id="10.9999/synthetic-04b1"))
    assert document.provider_version == version
    assert connector.transport.calls == 3
    connector.transport.assert_consumed()


def test_legacy_cursor_cannot_silently_upgrade_ranking(configurations):
    old = assembly(configurations, "2.0.0")
    new = assembly(configurations, "2.1.0")
    query = StructuredQuery(text="synthetic fixture", page_size=1)
    cursor = old.search(query).next_cursor
    with pytest.raises(ValueError, match="cursor provider/configuration/query mismatch"):
        new.search(query.model_copy(update={"cursor": cursor}))
    assert new.transport.calls == 0


def test_ranking_version_changes_request_and_configuration_identity(configurations):
    query = StructuredQuery(text="synthetic fixture", page_size=1)
    old = assembly(configurations, "2.0.0").search(query)
    new = assembly(configurations, "2.1.0").search(query)
    assert old.config_checksum != new.config_checksum
    assert old.responses[0].metadata.request_checksum != new.responses[0].metadata.request_checksum
    assert old.next_cursor.config_checksum != new.next_cursor.config_checksum


@pytest.mark.parametrize(
    "author,expected",
    [
        ({"name": "Synthetic Research Institute"}, ("Synthetic Research Institute",)),
        ({"given": "Sample", "family": "Author"}, ("Sample Author",)),
        (
            {"name": "Synthetic Research Institute", "given": None, "family": None},
            ("Synthetic Research Institute",),
        ),
    ],
)
def test_crossref_corporate_and_person_authors(configurations, author, expected):
    fixture = load_fixture(Path(__file__).parent / "fixtures/crossref.json").exchanges[0]
    body = json.loads(fixture.body_text)
    body["message"]["items"][0]["author"] = [author]
    fixture = fixture.model_copy(update={"body_text": json.dumps(body)})
    connector = DeterministicFakeConnector(configurations["crossref"], (fixture,))
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert page.records[0].authors == expected


@pytest.mark.parametrize("author", [{}, {"name": 7}, {"name": ""}, {"given": 7}])
def test_crossref_invalid_author_stays_parse_error(configurations, author):
    fixture = load_fixture(Path(__file__).parent / "fixtures/crossref.json").exchanges[0]
    body = json.loads(fixture.body_text)
    body["message"]["items"][0]["author"] = [author]
    fixture = fixture.model_copy(update={"body_text": json.dumps(body)})
    connector = DeterministicFakeConnector(configurations["crossref"], (fixture,))
    with pytest.raises(ConnectorError) as caught:
        connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert caught.value.category is ErrorCategory.PARSE
