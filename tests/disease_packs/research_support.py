"""B03 recorded connector and actual B02 RULE pipeline. 模拟真实内容模式，仅隔离测试。"""

from uuid import uuid4

from db.knowledge import models as m
from packages.knowledge_claims.ingestion import RULE_CONFIG_CHECKSUM, source_binding
from packages.knowledge_claims.models import CandidateRequest, StructuredOutput
from tests.database.test_knowledge_claims_04b5 import budget


def research_claim(database, store, index_config, *, synthetic):
    """Synthetic bytes, simulated non-synthetic storage mode; never reviewed medical content."""
    from apps.knowledge_federated_search import KnowledgeSearchService
    from apps.knowledge_provider_registration import register_source_provider
    from db.knowledge.repository.claims import read_evidence
    from packages.knowledge_archive import Bibliography, FetchInput
    from packages.knowledge_connectors import (
        FakeClock,
        FixtureExchange,
        RecordedTransport,
        SafeHTTPClient,
        SourceLocator,
    )
    from packages.knowledge_connectors.configuration import CatalogEntry
    from packages.knowledge_connectors.connectors import GuidelineSiteConnector
    from packages.knowledge_index.models import Filters
    from tests.knowledge_claims.support import output
    from tests.knowledge_search.support import assembly, config, plan

    token = uuid4().hex
    url = "https://www.nice.org.uk/guidance/synthetic-b03-fixture-" + token
    cfg = config(
        "guideline",
        provider_code="SOURCE:B03:PROVIDER:" + token.upper(),
        catalog=(
            CatalogEntry(
                record_id="fixture-b03-" + token,
                title="Synthetic condition association adults fixture " + token,
                url=url,
            ),
        ),
    )
    provider = register_source_provider(cfg, database.editor, synthetic=synthetic)
    query = plan(cfg, source_types=("guideline",))
    executor, transports = assembly((cfg,), {cfg.provider_code: ()})
    with executor:
        receipt = KnowledgeSearchService(
            executor, database.editor, (provider,), synthetic=synthetic
        ).search(query, idempotency_key="b03-" + token)
    search_hit = next(s.record for s in receipt.records if type(s.record) is m.SearchHit)
    assert transports[cfg.provider_code].calls == 0  # catalog discovery has no HTTP receipt
    text = "OCT result is positive."
    tape = FixtureExchange(
        host="www.nice.org.uk",
        path="/guidance/synthetic-b03-fixture-" + token,
        headers=(("content-type", "text/html"),),
        body_text="<html><body><p>" + text + "</p></body></html><!--" + token + "-->",
    )
    transport = RecordedTransport((tape,))
    connector = GuidelineSiteConnector(
        cfg,
        SafeHTTPClient(cfg, transport, resolver=lambda host: ("93.184.216.34",), clock=FakeClock()),
    )
    fetched = connector.fetch(SourceLocator(url=url))
    transport.assert_consumed()
    archived = database.editor.archive_hit(
        search_hit.reference(),
        FetchInput(
            document=fetched,
            metadata_response_checksum=fetched.response.response_checksum,
            bibliography=Bibliography(language="en"),
        ),
        store,
    )
    applicability = m.seal(
        m.ApplicabilityCondition,
        code="CLAIM:B03:" + token.upper(),
        version="1.0.0",
        synthetic=synthetic,
        population=("synthetic_fixture_only",),
        conditions=("software_test",),
        modalities=("synthetic",),
        scope_note="Synthetic software fixture, simulated real content mode only.",
    )
    database.editor.put(applicability)
    built = database.editor.build_document_index(
        archived,
        store,
        index_config,
        applicability_refs=(applicability.reference(),),
    )
    scope = Filters(
        population=("synthetic_fixture_only",),
        conditions=("software_test",),
        modalities=("synthetic",),
    )
    hit = database.editor.query_candidate_index(built["key"], "OCT", scope)["hits"][0]
    with database.editor._connect() as conn:
        evidence = read_evidence(conn, hit, scope, store)
    request = CandidateRequest(
        request_key="recorded-fixture-" + token,
        rule_config_checksum=RULE_CONFIG_CHECKSUM,
        source=source_binding(evidence),
        candidate=StructuredOutput.model_validate_json(
            output(text, hit["position"]["span_start"])
        ).candidates[0],
    )
    result = database.editor.ingest_candidate(
        request, store, budget(), producer="recorded-fixture-producer", filters=scope
    )
    assert result["status"] == "succeeded", result
    assert all(
        s.record.synthetic == synthetic and s.status == m.KnowledgeStatus.DRAFT
        for s in result["records"]
    )
    claim = next(s.record for s in result["records"] if type(s.record) is m.ClinicalClaim)
    database.editor.admit_claim(claim.reference(), hit, store, filters=scope)
    assert database.reviewer.review_bundle(claim.reference())["records"]
    assert database.editor.get(claim.reference()).status == m.KnowledgeStatus.DRAFT
    return claim, hit, built, applicability
