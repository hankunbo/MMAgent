"""Literature CLI contract tests: memory repository and recorded HTTP, never live IO."""

import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from apps import knowledge_literature_tools as app
from db.knowledge import KnowledgeSnapshot, KnowledgeStatus
from db.knowledge.repository import KnowledgeConflictError, KnowledgeNotFoundError
from packages.knowledge_connectors import FakeClock, SafeHTTPClient, load_configuration
from packages.knowledge_connectors.http import WireResponse
from tests.knowledge_connectors.test_http_04b1 import ScriptedTransport


@pytest.fixture(autouse=True)
def isolate_operator_contact(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CONTACT_FILE", tmp_path / "knowledge_contact.json")
    monkeypatch.delenv("MMAGENT_NCBI_EMAIL", raising=False)


class MemoryRepository:
    """Exercise real application/connector logic, not PostgreSQL persistence claims."""

    def __init__(self):
        self.rows = {}
        self.searches = {}

    @staticmethod
    def identity(reference):
        return reference.code, reference.version, reference.checksum

    def put(self, record):
        key = self.identity(record.reference())
        self.rows.setdefault(
            key,
            KnowledgeSnapshot(record=record, status=KnowledgeStatus.DRAFT, revision=0),
        )
        return self.rows[key]

    def get(self, reference):
        try:
            return self.rows[self.identity(reference)]
        except KeyError:
            raise KnowledgeNotFoundError("synthetic repository: exact reference missing") from None

    def lookup_search(self, key, binding):
        entry = self.searches.get(key)
        if entry is None:
            return None
        if entry[0] != binding:
            raise KnowledgeConflictError("synthetic repository: search key content changed")
        return entry[1]

    def commit_search(self, records, key, binding):
        existing = self.lookup_search(key, binding)
        if existing is not None:
            return existing
        saved = tuple(self.put(record) for record in records)
        self.searches[key] = binding, saved
        return saved


@pytest.fixture
def registered():
    repo = MemoryRepository()
    app.register(repo)
    return repo


def recorded_factory(*responses, production_identity=False):
    transports = []

    def factory(config, governor=None):
        transport = ScriptedTransport(*responses)
        if production_identity:
            transport.contact_mode = "production"
        transports.append(transport)
        return SafeHTTPClient(
            config,
            transport,
            clock=governor.clock if governor else FakeClock(),
            governor=governor,
            resolver=lambda _: ("93.184.216.34",),
            environment=lambda _: None,
        )

    return factory, transports


def test_five_providers_include_exact_new_aao_catalog_without_mutating_baseline():
    configs = app.configuration().providers
    assert len(configs) == len({c.provider_code for c in configs}) == 5
    aao = app.provider_config("aao")
    assert aao.provider_version == "2.2.1"
    assert aao.health_endpoint == aao.catalog[0].url
    assert aao.adapter == "guideline" and aao.source_type == "GUIDELINE"
    assert {entry.record_id for entry in aao.catalog} == {
        "AAO_POAG_PPP_2026",
        "AAO_POAG_SUSPECT_PPP_2026",
        "AAO_ANGLE_CLOSURE_PPP_2026",
    }
    assert all(urlsplit(entry.url).hostname == "www.aao.org" for entry in aao.catalog)
    assert all("/Assets/" in entry.url for entry in aao.catalog)
    baseline = app.aao_baseline()
    assert baseline.provider_version == "2.1.0"
    assert all(urlsplit(e.url).hostname == "www.aaojournal.org" for e in baseline.catalog)
    assert app.provider_record(baseline).checksum == (
        "sha256:e0e99c47366c06d3b38d9b5dcbe2e151b9ea494450fd2230349226862c1dc2e5"
    )
    old = next(
        c
        for c in load_configuration(Path("configs/knowledge_sources.toml")).providers
        if c.provider_code == aao.provider_code
    )
    assert old.provider_version == "2.0.0" and not old.catalog
    cochrane = app.provider_config("cochrane")
    assert cochrane.source_type == "WEB" and len(cochrane.catalog) == 1
    crossref = app.provider_config("crossref")
    assert crossref.provider_version == "2.1.0"
    old_crossref = next(
        c
        for c in load_configuration(Path("configs/knowledge_sources.toml")).providers
        if c.provider_code == crossref.provider_code
    )
    assert old_crossref.provider_version == "2.0.0"
    assert crossref.adapter == old_crossref.adapter == "crossref"


def test_registration_and_listing_do_not_claim_live_fetch_or_medical_publication(monkeypatch):
    monkeypatch.delenv("MMAGENT_NCBI_EMAIL", raising=False)
    repo = MemoryRepository()
    assert not any(row["registered"] for row in app.listing(repo))
    first = app.register(repo)
    assert app.register(repo) == first and len(repo.rows) == 8
    old_aao = app.provider_record(app.aao_baseline())
    current_aao = app.provider_record(app.provider_config("aao"))
    intermediate_aao = app.provider_record(app.aao_pdf_baseline())
    assert current_aao.supersedes == intermediate_aao.reference()
    assert intermediate_aao.supersedes == old_aao.reference()
    assert repo.get(old_aao.reference()).record == old_aao
    assert repo.get(current_aao.reference()).record == current_aao
    old_crossref = app.provider_record(app.crossref_baseline())
    current_crossref = app.provider_record(app.provider_config("crossref"))
    assert current_crossref.supersedes == old_crossref.reference()
    assert repo.get(old_crossref.reference()).record == old_crossref
    assert repo.get(current_crossref.reference()).record == current_crossref
    assert all(row["registered"] and row["enabled"] for row in first)
    assert all(not row["synthetic"] for row in first)
    assert {row["live_network_status"] for row in first} == {"not_checked_by_listing"}
    assert all(s.status is KnowledgeStatus.DRAFT for s in repo.rows.values())
    pubmed = next(row for row in first if row["provider"] == "pubmed")
    assert pubmed["contact_environment"] == "MMAGENT_NCBI_EMAIL"
    assert pubmed["contact_configured"] is False


def test_pubmed_and_aao_use_declared_hostname_proxy():
    from packages.knowledge_connectors import ExplicitProxyHTTPSTransport, PinnedHTTPSTransport

    for key in app.PROVIDERS:
        config = app.provider_config(key)
        http = app.client(config)
        assert isinstance(http.transport, ExplicitProxyHTTPSTransport)
        expected = "proxy_hostname" if key in ("pubmed", "aao") else "local_public_ip_pinned"
        assert http.transport.policy.target_resolution == expected
        assert http.transport.policy.receipt()["tls_verification"] is True
        assert http.transport.policy.receipt()["direct_fallback"] is False
        assert type(app.client(config, transport="direct").transport) is PinnedHTTPSTransport
    assert app.provider_config("pubmed").provider_version == "2.0.0"


def test_saved_contact_is_loaded_without_export_and_not_shown(registered, monkeypatch):
    email = "synthetic-contact@example.org"
    app.CONTACT_FILE.write_text(
        json.dumps({"version": "knowledge-contact/1.0.0", "ncbi_email": email})
    )
    assert app.contact_environment("MMAGENT_NCBI_EMAIL") == email
    assert app.client(app.provider_config("pubmed")).environment("MMAGENT_NCBI_EMAIL") == email
    rows = app.listing(registered)
    assert next(r for r in rows if r["provider"] == "pubmed")["contact_configured"] is True
    assert email not in json.dumps(rows)
    monkeypatch.setenv("MMAGENT_NCBI_EMAIL", "override@example.org")
    assert app.contact_environment("MMAGENT_NCBI_EMAIL") == "override@example.org"
    monkeypatch.setenv("MMAGENT_NCBI_EMAIL", "")
    assert app.contact_environment("MMAGENT_NCBI_EMAIL") == ""
    monkeypatch.setenv("OTHER_CONTACT", "other@example.org")
    assert app.contact_environment("OTHER_CONTACT") == "other@example.org"


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        '{"version":"wrong","ncbi_email":"private"}',
        '{"version":"knowledge-contact/1.0.0","ncbi_email":"private"}',
        "x" * 4097,
    ],
)
def test_invalid_saved_contact_has_sanitized_error(raw):
    from packages.knowledge_connectors import ConnectorError, ErrorCategory

    app.CONTACT_FILE.write_text(raw)
    with pytest.raises(ConnectorError) as caught:
        app.contact_environment("MMAGENT_NCBI_EMAIL")
    assert caught.value.category is ErrorCategory.CONFIGURATION
    assert "private" not in str(caught.value)


def test_other_identity_does_not_load_ncbi_file():
    app.CONTACT_FILE.write_text("not-json")
    assert app.contact_environment("UNCONFIGURED_OTHER_CONTACT") is None


@pytest.mark.parametrize("damage", ["withdrawn", "wrong_record", "published"])
def test_unavailable_registration_rejects_fetch_before_http(registered, damage):
    config = app.provider_config("aao")
    identity = registered.identity(app.provider_record(config).reference())
    original = registered.rows[identity]
    changes = {
        "withdrawn": {"withdrawn": True},
        "wrong_record": {"record": app.provider_record(app.provider_config("crossref"))},
        "published": {"status": KnowledgeStatus.PUBLISHED},
    }[damage]
    registered.rows[identity] = original.model_copy(update=changes)
    factory, transports = recorded_factory()
    with pytest.raises(ValueError, match="registered_literature_provider_unavailable"):
        app.fetch(registered, "aao", config.catalog[0].record_id, client_factory=factory)
    assert not transports
    assert not next(row for row in app.listing(registered) if row["provider"] == "aao")["enabled"]


def test_pubmed_missing_contact_fails_without_synthetic_identity_fallback(registered):
    factory, transports = recorded_factory(production_identity=True)
    result = app.fetch(registered, "pubmed", "12345678", client_factory=factory)
    assert result["status"] == "failed" and result["category"] == "configuration"
    assert not result["archived"] and "excerpt" not in result
    assert len(transports) == 1 and not transports[0].requests


def test_html_fetch_uses_exact_catalog_and_returns_bounded_unarchived_excerpt(registered):
    body = (
        b"<nav>hidden menu</nav><script>hidden script</script>"
        b"<main><p>Synthetic public guideline body</p></main><footer>hidden footer</footer>"
    )
    factory, transports = recorded_factory(
        WireResponse(200, (("content-type", "text/html"),), body)
    )
    entry = app.provider_config("cochrane").catalog[0]
    result = app.fetch(
        registered, "cochrane", entry.record_id, client_factory=factory, max_chars=10
    )
    assert result["status"] == "succeeded" and not result["archived"]
    assert result["excerpt"] == "Synthetic " and result["excerpt_truncated"]
    assert "hidden" not in result["excerpt"] and "<" not in result["excerpt"]
    assert result["byte_count"] == len(body)
    assert transports[0].requests[0][0].url == entry.url


def test_remote_403_remains_failed_fetch_not_empty_success(registered):
    factory, transports = recorded_factory(
        WireResponse(403, (("content-type", "text/html"),), b"Denied")
    )
    result = app.fetch(
        registered,
        "aao",
        app.provider_config("aao").catalog[0].record_id,
        client_factory=factory,
    )
    assert result["status"] == "failed" and not result["archived"]
    assert "excerpt" not in result
    assert result["metadata"]["attempts"][-1]["http_status"] == 403
    assert len(transports[0].requests) == 1


def test_chinese_glaucoma_cli_uses_curated_search_and_same_key_reuses_records(
    registered, monkeypatch, capsys
):
    factory, transports = recorded_factory()
    monkeypatch.setattr(app, "repository", lambda capability: registered)
    monkeypatch.setattr(app, "client", lambda c, g=None, **kwargs: factory(c, g))
    argv = ["search", "--providers", "aao", "--topic", "青光眼", "--key", "fixture-cli"]
    assert app.main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["topic"] == "glaucoma" and first["status"] == "succeeded"
    # The angle-closure title has no "glaucoma" token; this is literal catalog
    # matching, not a fabricated remote result or implicit semantic expansion.
    assert first["provider_hit_count"] == 2
    assert first["queries"][0]["text"] == "glaucoma"
    assert not first["cache_hit"] and first["database_written"]
    assert app.main(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["cache_hit"] and not second["database_written"]
    assert first["results"] == second["results"]
    assert all(not transport.requests for transport in transports)


def test_403_in_search_is_provider_failure_not_no_literature(registered):
    factory, transports = recorded_factory(
        WireResponse(403, (("content-type", "application/json"),), b"{}")
    )
    result = app.search(
        registered, ("crossref",), "glaucoma", key="fixture-denied", client_factory=factory
    )
    assert result["status"] == "failed" and result["provider_hit_count"] == 0
    assert result["runs"][0]["outcome"]["stop_reason"] == "provider_error"
    assert result["runs"][0]["outcome"]["error_category"] is not None
    assert len(transports[0].requests) == 1


def test_duplicate_groups_merge_transitive_exact_identities_and_keep_provenance():
    def hit(provider, **identifiers):
        return {
            "provider": provider,
            "title": "Same title does not prove identical paper",
            "identifiers": [
                {"identifier_type": kind, "value": value} for kind, value in identifiers.items()
            ],
        }

    hits = [
        hit("crossref", DOI="10.1234/AbCd"),
        hit("pubmed", PMID="12345678"),
        hit("europepmc", DOI="10.1234/abcd", PMID="12345678"),
        hit("other", DOI="10.1234/different"),
    ]
    grouped = app.duplicate_groups(hits)
    assert sorted(len(group["hits"]) for group in grouped) == [1, 3]
    assert sum(len(group["hits"]) for group in grouped) == len(hits)
    merged = next(group for group in grouped if len(group["hits"]) == 3)
    assert {h["provider"] for h in merged["hits"]} == {"crossref", "pubmed", "europepmc"}
    assert {item["value"] for item in merged["identifiers"]} == {"10.1234/abcd", "12345678"}
    assert len(app.duplicate_groups([hit("one"), hit("two")])) == 2


@pytest.mark.parametrize("topic", ["patient John Smith", "86岁病人左眼照片", "unknown-topic"])
def test_arbitrary_patient_free_text_not_forwarded_to_search(registered, topic):
    factory, transports = recorded_factory()
    with pytest.raises(ValueError):
        app.search(registered, ("crossref",), topic, client_factory=factory)
    assert not transports


def crossref_response(status=200, *, title="Synthetic glaucoma measurement study"):
    body = {
        "status": "ok",
        "message": {
            "total-results": 1,
            "items": [
                {
                    "DOI": "10.9999/synthetic-literature-sweep",
                    "title": [title],
                    "URL": "https://doi.org/10.9999/synthetic-literature-sweep",
                }
            ],
        },
    }
    return WireResponse(status, (("content-type", "application/json"),), json.dumps(body).encode())


def test_sweep_searches_each_topic_separately_and_deduplicates_exact_papers(
    registered, monkeypatch
):
    factory, transports = recorded_factory(crossref_response())
    monkeypatch.setattr(app, "client", lambda c, g=None, **kwargs: factory(c, g))
    result = app.sweep(
        registered,
        ("crossref",),
        topics=("cup_disc_ratio", "rim"),
        key="fixture-topic-batch",
    )
    assert result["status"] == "succeeded" and result["topic_count"] == 2
    first, second = result["topic_results"]
    assert first["request_key"] == "fixture-topic-batch:per-source-v1:cup_disc_ratio:crossref"
    assert second["request_key"] == "fixture-topic-batch:per-source-v1:rim:crossref"
    first_text, second_text = first["queries"][0]["text"], second["queries"][0]["text"]
    assert "glaucoma" in first_text and "cup disc ratio" in first_text
    assert "neuroretinal rim" not in first_text
    assert "glaucoma" in second_text and "neuroretinal rim" in second_text
    assert "cup disc ratio" not in second_text
    assert result["provider_hit_count"] == 2 and result["unique_identity_group_count"] == 1
    assert len(transports) == 2 and all(len(t.requests) == 1 for t in transports)
    # Grouping is presentation only; neither of the exact per-topic facts disappears.
    first_ref = first["results"][0]["hits"][0]["reference"]
    second_ref = second["results"][0]["hits"][0]["reference"]
    assert first_ref != second_ref and len(registered.searches) == 2
    # DOI comes from retained discovery metadata, not only persisted URL identity.
    assert {
        "identifier_type": "DOI",
        "value": "10.9999/synthetic-literature-sweep",
    } in first["results"][0]["hits"][0]["identifiers"]
    assert result["search_task_count"] == 2


def test_sweep_preserves_successful_topic_when_another_site_request_is_denied(
    registered, monkeypatch
):
    transports = []

    def client(config, governor=None, **kwargs):
        factory, created = recorded_factory(crossref_response(200 if not transports else 403))
        result = factory(config, governor)
        transports.extend(created)
        return result

    monkeypatch.setattr(app, "client", client)
    result = app.sweep(
        registered, ("crossref",), topics=("cup_disc_ratio", "rim"), key="fixture-partial"
    )
    assert result["status"] == "partial"
    assert [r["status"] for r in result["topic_results"]] == ["succeeded", "failed"]
    assert result["provider_hit_count"] == result["unique_identity_group_count"] == 1
    assert result["topic_results"][1]["runs"][0]["outcome"]["stop_reason"] == "provider_error"


@pytest.mark.parametrize("command", ["register", "search", "sweep"])
def test_existing_output_rejected_before_database_or_network(command, tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    original = b'{"prior_experiment":"must remain unchanged"}\n'
    output.write_bytes(original)
    calls = []

    def repository(*args):
        calls.append(args)
        raise AssertionError("existing output must be rejected before database assembly")

    monkeypatch.setattr(app, "repository", repository)
    with pytest.raises(SystemExit) as caught:
        app.main([command, "--output", str(output)])
    assert caught.value.code == 2 and not calls
    assert output.read_bytes() == original


def test_output_export_contains_full_result_not_just_stdout_receipt(tmp_path, capsys):
    output = tmp_path / "topics.json"
    assert app.main(["topics", "--output", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text())
    assert receipt == {"output": str(output), "status": "saved"}
    assert "cup_disc_ratio" in saved and "rnfl" in saved
    assert "status" not in saved


def test_sweep_metadata_rejection_is_local_to_source_and_successful_retry_is_reused(
    registered, monkeypatch, caplog, capsys
):
    rejected = "Synthetic metadata patient id FORBIDDEN_FIXTURE_123"
    factory, transports = recorded_factory(crossref_response(title=rejected))
    monkeypatch.setattr(app, "client", lambda c, g=None, **kwargs: factory(c, g))
    options = {
        "topics": ("rim",),
        "key": "fixture-source-local-rejection",
    }
    first = app.sweep(registered, ("crossref", "aao"), **options)
    assert first["status"] == "partial"
    assert first["topic_count"] == 1 and first["search_task_count"] == 2
    failed, succeeded = first["topic_results"]
    assert failed["status"] == "rejected_metadata" and not failed["database_written"]
    assert failed["provider"] == "crossref" and failed["results"] == []
    assert succeeded["status"] == "succeeded" and succeeded["database_written"]
    assert not succeeded["cache_hit"] and len(registered.searches) == 1
    before = dict(registered.rows)
    second = app.sweep(registered, ("crossref", "aao"), **options)
    assert second["topic_results"][0]["status"] == "rejected_metadata"
    assert second["topic_results"][1]["cache_hit"]
    assert not second["topic_results"][1]["database_written"]
    assert registered.rows == before and len(registered.searches) == 1
    assert first["topic_results"][1]["results"] == second["topic_results"][1]["results"]
    assert sum(len(t.requests) for t in transports) == 2
    captured = capsys.readouterr()
    combined = json.dumps(
        [first, second, [s.model_dump(mode="json") for s in registered.rows.values()]]
    )
    assert rejected not in combined + captured.out + captured.err + caplog.text
    assert "FORBIDDEN_FIXTURE_123" not in combined + captured.out + captured.err + caplog.text


def test_sweep_all_metadata_rejected_is_failed_not_empty_and_writes_no_searches(
    registered, monkeypatch, caplog, capsys
):
    rejected = "Synthetic patient id ALL_REJECTED_FIXTURE_123"
    factory, _ = recorded_factory(crossref_response(title=rejected))
    monkeypatch.setattr(app, "client", lambda c, g=None, **kwargs: factory(c, g))
    before = dict(registered.rows)
    result = app.sweep(
        registered,
        ("crossref",),
        topics=("cup_disc_ratio", "rim"),
        key="fixture-all-rejected",
    )
    assert result["status"] == "failed" and result["provider_hit_count"] == 0
    assert result["unique_identity_group_count"] == 0
    assert all(r["status"] == "rejected_metadata" for r in result["topic_results"])
    assert all(not r["database_written"] for r in result["topic_results"])
    assert registered.rows == before and not registered.searches
    captured = capsys.readouterr()
    assert "ALL_REJECTED_FIXTURE_123" not in (
        json.dumps(result) + captured.out + captured.err + caplog.text
    )


def test_url_path_case_is_not_a_deduplication_equivalence():
    hits = [
        {
            "provider": "fixture",
            "title": "Matching titles are insufficient",
            "identifiers": [{"identifier_type": "URL", "value": url}],
        }
        for url in ("https://example.org/Study", "https://example.org/study")
    ]
    groups = app.duplicate_groups(hits)
    assert len(groups) == 2
    assert {g["identifiers"][0]["value"] for g in groups} == {
        "https://example.org/Study",
        "https://example.org/study",
    }


def test_controlled_multi_topic_search_reuses_original_query_and_persistence(registered):
    factory, transports = recorded_factory(crossref_response())
    result = app.search(
        registered,
        ("crossref",),
        "glaucoma",
        topics=("rim", "slo"),
        key="fixture-targeted-rim",
        client_factory=factory,
    )
    assert result["topics"] == ["rim", "slo"] and result["topic"] is None
    assert result["database_written"] and len(transports) == 1
    query = result["queries"][0]["text"]
    assert "neuroretinal rim" in query and "scanning laser ophthalmoscopy" in query
    before = sum(len(t.requests) for t in transports)
    cached = app.search(
        registered,
        ("crossref",),
        "glaucoma",
        topics=("rim", "slo"),
        key="fixture-targeted-rim",
        client_factory=factory,
    )
    assert cached["cache_hit"] and sum(len(t.requests) for t in transports) == before
    with pytest.raises(KnowledgeConflictError):
        app.search(
            registered,
            ("crossref",),
            "glaucoma",
            topics=("rnfl",),
            key="fixture-targeted-rim",
            client_factory=factory,
        )


def test_multi_topic_does_not_open_free_patient_query_input(registered):
    factory, transports = recorded_factory()
    with pytest.raises(ValueError, match="public_discovery_topic"):
        app.search(
            registered,
            ("crossref",),
            "glaucoma",
            topics=("patient_identifier_123",),
            key="fixture-private-topic",
            client_factory=factory,
        )
    assert not transports and not registered.searches
