"""Second repair counterexamples; typed locators are not clinical vocabulary."""

import pytest

from packages.knowledge_search import QueryIdentifier
from packages.knowledge_search.privacy import safe_url
from packages.knowledge_search.provider import configuration_proof, default_rate_policy


@pytest.mark.parametrize(
    "value,canonical",
    [
        ("10.1000/xyz123", "10.1000/xyz123"),
        ("DOI:10.5555/Abc.2026", "10.5555/abc.2026"),
        ("https://doi.org/10.1000/foo%2Fbar", "10.1000/foo/bar"),
        ("10.1002/abc(2026)123", "10.1002/abc(2026)123"),
    ],
)
def test_second_repair_doi(value, canonical):
    assert QueryIdentifier(kind="DOI", value=value).value == canonical


@pytest.mark.parametrize(
    "value,canonical",
    [
        ("https://doi.org/10.1000/foo%2fbar", "https://doi.org/10.1000/foo/bar"),
        (
            "https://www.nice.org.uk/guidance/caf%C3%A9",
            "https://www.nice.org.uk/guidance/caf%C3%A9",
        ),
        (
            "https://www.nice.org.uk/guidance/cafe\u0301",
            "https://www.nice.org.uk/guidance/caf%C3%A9",
        ),
    ],
)
def test_second_repair_url(value, canonical):
    assert safe_url(value) == canonical


def test_second_repair_url_default_host_allowlist():
    with pytest.raises(ValueError):
        safe_url("https://unknown.example.invalid/article/123")


@pytest.mark.parametrize("kind,value", [("PMID", "12345678"), ("PMCID", "PMC12345678")])
def test_numeric_query_identifiers_explicitly_unadmitted(kind, value):
    with pytest.raises(ValueError):
        QueryIdentifier(kind=kind, value=value)


@pytest.mark.parametrize(
    "value",
    [
        "9.1000/abc",
        "10.12/abc",
        "doi:PMID:123",
        "10.1000/" + "x" * 513,
        "10.1000/abc\x00",
        "10.1000/foo%252Fbar",
        "10.1000/%65mail%3Dx%40example.org",
        "10.1000/Authorization%3Asecret",
        "10.1000/../abc",
        "10.1000/a%ZZ",
        "10.1000/a%",
        "10.1000/mrn",
        "10.1000/date%20of%20birth%201970-01-02",
    ],
)
def test_identifier_privacy_and_syntax_rejection(value):
    with pytest.raises(ValueError):
        QueryIdentifier(kind="DOI", value=value)


@pytest.mark.parametrize(
    "value",
    [
        "https://user:secret@doi.org/a",
        "https://doi.org/a?",
        "https://doi.org/a#",
        "https://doi.org/a?mailto=x",
        "http://doi.org/a",
        "https://doi.org:444/a",
        "https://doi.org/a%252Fb",
        "https://doi.org/a%2f%2fb",
        "https://doi.org/a/%2e%2e/b",
        "https://doi.org/a%5Cb",
        "https://doi.org/a%ZZ",
        "https://doi.org/a%",
        "https://doi.org/%65mail%3Dx%40example.org",
        "https://doi.org/Authorization%3Asecret",
        "https://doi.org/ｍａｉｌｔｏ：secret",
        "https://doi.org/&#109;ailto:secret",
        "https://doi.org/ab\ncd",
        "https://doi.org/a%00b",
    ],
)
def test_url_policy_rejects_ambiguous_or_private_paths(value):
    with pytest.raises(ValueError):
        safe_url(value)


@pytest.mark.parametrize("kind", ["pubmed", "europe_pmc", "crossref", "guideline"])
def test_connector_generated_discovery_urls_use_shared_policy(kind):
    from pathlib import Path

    from packages.knowledge_connectors import StructuredQuery, load_configuration
    from packages.knowledge_search import FederatedSearch
    from tests.knowledge_connectors.test_connectors_04b1 import fake

    configs = {
        c.adapter: c
        for c in reversed(load_configuration(Path("configs/knowledge_sources.toml")).providers)
    }
    connector = fake(configs, kind)
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    hit = FederatedSearch._candidate(
        page.records[0],
        page,
        0,
        0,
        configuration_proof(connector._connector.config, default_rate_policy()),
    )
    assert hit.identifiers[0].value == safe_url(page.records[0].url)
    assert safe_url(hit.identifiers[0].value) == hit.identifiers[0].value


def test_connector_crossref_encoded_doi_path_and_guideline_unicode_path():
    import json
    from pathlib import Path

    from packages.knowledge_connectors import (
        CatalogEntry,
        DeterministicFakeConnector,
        StructuredQuery,
        load_fixture,
    )
    from packages.knowledge_search import FederatedSearch
    from tests.knowledge_search.support import config

    tape = load_fixture(Path("tests/knowledge_connectors/fixtures/crossref.json")).exchanges[0]
    body = json.loads(tape.body_text)
    body["message"]["items"][0]["DOI"] = "10.1000/foo/bar"
    body["message"]["items"][0]["URL"] = "https://doi.org/10.1000/foo%2Fbar"
    tape = tape.model_copy(update={"body_text": json.dumps(body)})
    connector = DeterministicFakeConnector(config("crossref"), (tape,))
    page = connector.search(StructuredQuery(text="synthetic fixture", page_size=1))
    assert (
        FederatedSearch._candidate(
            page.records[0],
            page,
            0,
            0,
            configuration_proof(connector._connector.config, default_rate_policy()),
        )
        .identifiers[0]
        .value
        == "https://doi.org/10.1000/foo/bar"
    )
    cfg = config(
        "guideline",
        catalog=(
            CatalogEntry(
                record_id="general",
                title="Synthetic fixture",
                url="https://www.nice.org.uk/guidance/caf%C3%A9",
            ),
        ),
    )
    page = DeterministicFakeConnector(cfg, ()).search(
        StructuredQuery(text="synthetic fixture", page_size=1)
    )
    assert (
        FederatedSearch._candidate(
            page.records[0], page, 0, 0, configuration_proof(cfg, default_rate_policy())
        )
        .identifiers[0]
        .value.endswith("caf%C3%A9")
    )


def test_shared_template_migration_snapshot_has_no_drift():
    from db.knowledge.validation.build import TARGET, artifact

    assert TARGET.read_text() == artifact()


def test_identifier_policy_revalidation_and_checksum_canonicalization():
    from packages.knowledge_search import KnowledgeQueryPlan
    from tests.knowledge_search.support import config, plan

    first = QueryIdentifier(kind="DOI", value="DOI:10.1000/Foo%2fBar")
    second = QueryIdentifier(kind="DOI", value="10.1000/foo/bar")
    assert first == second
    assert (
        plan(config(), identifiers=(first,)).checksum()
        == plan(config(), identifiers=(second,)).checksum()
    )
    for malformed in (
        first.model_copy(update={"source": "case_observation"}),
        QueryIdentifier.model_construct(kind="DOI", value="10.1000/mrn"),
    ):
        with pytest.raises(ValueError):
            QueryIdentifier.model_validate(malformed)
    raw = plan(config()).model_dump(mode="json")
    raw["query_plan_version"] = "2.0.0"
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate_json(__import__("json").dumps(raw))
