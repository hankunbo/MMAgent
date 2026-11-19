"""Approved EGS WEB content through original discovery/archive/index repositories."""

import argparse
import json

from apps.glaucoma_evidence_bindings import ROOT, dump, ref, write_new
from db.knowledge import models as k
from packages.knowledge_connectors import (
    PinnedHTTPSTransport,
    SafeHTTPClient,
    SourceLocator,
    WebSearchFallbackConnector,
    load_configuration,
)
from packages.object_store import LocalObjectStore
from packages.object_store import checksum as byte_checksum

CONFIG = ROOT / "configs/knowledge_sources_egs.toml"
DESTINATION = ROOT / "docs/experiments/glaucoma-egs-content-v2"
LINKS = {
    "EGS-HEMORRHAGE-2024": {
        "nodes": ["HEMORRHAGE", "DISC_BLEED", "ALTERNATIVES"],
        "terms": ["hemorrhage", "detachment"],
        "interpretation": "盘周出血可与青光眼相关，也有非青光眼解释；不能单独确诊。",
    },
    "EGS-ONH-2019": {
        "nodes": ["CUPDISC", "CUP_RIM_MEASURE", "DISC_SIZE", "DISC_GEOMETRY"],
        "terms": ["disc size", "rim", "cup"],
        "interpretation": "杯盘比需结合视盘大小和盘沿形态解释，不是单指标诊断公式。",
    },
    "EGS-RNFL-2017": {
        "nodes": ["RNFL", "RNFL_THICKNESS", "CHANGE", "TEMPORAL_CHANGE"],
        "terms": ["progression", "floor", "thickness"],
        "interpretation": "OCT结构测量可辅助早期进展评估；晚期存在测量底限，不授权SLO输出厚度。",
    },
    "EGS-OCT-2020": {
        "nodes": ["RNFL", "RNFL_THICKNESS", "ALTERNATIVES"],
        "terms": ["diagnosis", "normative", "clinical"],
        "interpretation": "OCT参照数据库异常需要临床背景解释，不能单凭颜色/异常报告确诊。",
    },
}


def persist_discovery(repo, *, client_factory=None, synthetic=False):
    from apps.knowledge_federated_search import KnowledgeSearchService
    from apps.knowledge_provider_registration import register_source_provider
    from packages.knowledge_search import (
        CodedTerm,
        FederatedSearch,
        KnowledgeQueryPlan,
        ProviderBudget,
        load_rate_policy,
    )

    configuration = load_configuration(CONFIG)
    policy = load_rate_policy(CONFIG.with_name("knowledge_search.toml"))
    # Preserve the first attempted listing deployment as an exact predecessor.
    old_config = configuration.providers[0].model_copy(
        update={"provider_version": "1.0.0", "search_endpoint": "https://www.eugs.org/news"}
    )
    previous = register_source_provider(old_config, repo, synthetic=synthetic, rate_policy=policy)
    provider = register_source_provider(
        configuration.providers[0],
        repo,
        supersedes=previous.record.reference(),
        synthetic=synthetic,
        rate_policy=policy,
    )
    plan = KnowledgeQueryPlan(
        goal="GOAL:KNOWLEDGE:EVIDENCE",
        claim="CLAIM:KNOWLEDGE:ASSOCIATION",
        evidence_need="diagnostic_criteria",
        diseases=(CodedTerm(code="CONCEPT:DISEASE:GLAUCOMA", label="glaucoma"),),
        population=("research cohort",),
        source_types=("web",),
        providers=(
            ProviderBudget(
                provider_code=provider.record.code,
                page_size=10,
                max_hits=10,
                max_pages=1,
                max_requests=1,
                max_retries=0,
            ),
        ),
        total_max_hits=10,
        total_max_requests=1,
        max_concurrency=1,
    )

    def factory(config, governor):
        return SafeHTTPClient(
            config, PinnedHTTPSTransport(), clock=governor.clock, governor=governor
        )

    with FederatedSearch(configuration, policy, client_factory or factory) as executor:
        receipt = KnowledgeSearchService(executor, repo, (provider,), synthetic=synthetic).search(
            plan, idempotency_key="egs-four-curated-pages-20260916-v4"
        )
    return [x for x in receipt.records if x.record.kind is k.KnowledgeKind.SEARCH_HIT]


def ingest(repo, destination=DESTINATION):
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from packages.knowledge_archive import Bibliography, FetchInput
    from packages.knowledge_index.models import IndexConfig

    check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
    store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
    config = load_configuration(CONFIG).providers[0]
    connector = WebSearchFallbackConnector(config, SafeHTTPClient(config, PinnedHTTPSTransport()))
    hits = persist_discovery(repo)
    if len(hits) != 4:
        raise ValueError("egs_discovery_incomplete")
    pages = []
    for hit_snapshot in hits:
        hit = hit_snapshot.record
        key = hit.discovery.provider_record_id
        entry = next(e for e in config.catalog if e.record_id == key)
        path = destination / (key + ".json")
        if path.exists():
            page = json.loads(path.read_text())
            verify_page(repo, store, page)
            pages.append(page)
            continue
        previous = [
            o
            for o in repo.archive_catalog()["observations"]
            if o["hit_ref"] == dump(hit.reference()) and o["pipeline_version"] == "1.1.0"
        ]
        if previous:
            if len(previous) != 1:
                raise ValueError("egs_multiple_observations_require_selection")
            receipt = previous[0]
        else:
            fetched = connector.fetch(SourceLocator(url=entry.url))
            receipt = repo.archive_hit(
                hit.reference(),
                FetchInput(
                    pipeline_version="1.1.0",
                    document=fetched,
                    metadata_response_checksum=fetched.response.response_checksum,
                    bibliography=Bibliography(
                        language="en",
                        institutions=("European Glaucoma Society",),
                        canonical_url=entry.url,
                    ),
                ),
                store,
            )
        context = k.seal(
            k.ApplicabilityCondition,
            code="EVIDENCE:GLAUCOMA:EGS_AUTHORING_CONTEXT",
            version="1.0.0",
            population=("research cohort",),
            conditions=("glaucoma",),
            scope_note="Education authoring; not patient eligibility or approved clinical rules",
        )
        repo.put(context)
        resolved = {
            r["candidate_key"]
            for r in repo.archive_catalog().get("equivalences", ())
            if r["observation_key"] == receipt["key"]
            and r["source_version"] == receipt["version_ref"]
        }
        built = (
            {
                "status": "blocked_identity_conflict",
                "chunk_refs": [],
                "reason": "archive_merge_candidate_requires_resolution",
            }
            if set(receipt["candidate_keys"]) - resolved
            else repo.build_document_index(
                receipt,
                store,
                IndexConfig(
                    mode="fts_only",
                    fts_weight=1.0,
                    vector_weight=0.0,
                    vector_unavailable_reason="real_embedding_provider_not_configured_for_authoring",
                ),
                applicability_refs=(context.reference(),),
                targets=("glaucoma_authoring",),
            )
        )
        if built["status"] not in {"indexed", "blocked_identity_conflict"}:
            raise ValueError("egs_index_failed:" + key)
        chunks = []
        for chunk_ref in built["chunk_refs"]:
            c = repo.get(ref(chunk_ref)).record
            text = store.read_bounded(c.object_uri, 16384)
            if byte_checksum(text) != c.content_checksum or len(text) != c.size_bytes:
                raise ValueError("egs_chunk_bytes_mismatch")
            chunks.append(
                {
                    "ref": chunk_ref,
                    "position": dump(c.derivation.position),
                    "object_uri": c.object_uri,
                    "text_checksum": c.content_checksum,
                    "text": text.decode(),
                }
            )
        from packages.knowledge_index.parsers import parse

        source = repo.get(ref(receipt["version_ref"])).record
        body = store.read_bounded(source.object_uri, 2_000_000)
        preview = parse(body, source.media_type, language="en")
        if preview.status != "parsed":
            raise ValueError("egs_preview_parse_failed")
        page = {
            "version": "egs-source-binding/1.0.0",
            "id": key,
            "title": entry.title,
            "url": entry.url,
            "source_type": "WEB",
            "content_class": "EGS education / tip / pearl, not a clinical guideline",
            "provider_ref": dump(hit.provider_ref),
            "hit_ref": dump(hit.reference()),
            "source_ref": receipt["version_ref"],
            "archive": receipt,
            "fetch": receipt["http"],
            "index": built,
            "chunks": chunks,
            "preview_blocks_not_indexed_evidence": [
                dump(b)
                for b in preview.document.blocks
                if len(b.text) > 100 and any(t in b.text.casefold() for t in LINKS[key]["terms"])
            ],
            "binding": LINKS[key],
            "published": False,
            "medical_weight": None,
            "interpretation_owner": "assistant research interpretation; not approved clinical rule",
        }
        write_new(path, page)
        pages.append(page)
    result = {"version": "egs-content-set/1.0.0", "pages": pages, "published": False}
    write_new(destination / "content.json", result)
    return result


def verify_page(repo, store, page):
    from packages.knowledge_archive.approved_web import PAGES

    if (
        page["id"] not in LINKS
        or page["binding"] != LINKS[page["id"]]
        or page["url"] != PAGES[page["id"]]
    ):
        raise ValueError("egs_page_definition_mismatch")
    snapshot = repo.get(ref(page["source_ref"]))
    if snapshot.withdrawn or snapshot.status is k.KnowledgeStatus.DEPRECATED:
        raise ValueError("egs_source_unavailable")
    source = snapshot.record
    body = store.read_bounded(source.object_uri, 2_000_000)
    if byte_checksum(body) != source.content_checksum or len(body) != source.size_bytes:
        raise ValueError("egs_source_bytes_mismatch")
    if page["index"]["status"] == "indexed":
        from db.knowledge.repository.index import current_source

        current_source(repo.archive_catalog(), page["archive"])
    from packages.knowledge_index.parsers import parse

    parsed = parse(body, source.media_type, language="en")
    if parsed.status != "parsed" or page["preview_blocks_not_indexed_evidence"] != [
        dump(b)
        for b in parsed.document.blocks
        if len(b.text) > 100 and any(t in b.text.casefold() for t in LINKS[page["id"]]["terms"])
    ]:
        raise ValueError("egs_preview_binding_mismatch")
    for chunk in page["chunks"]:
        snap = repo.get(ref(chunk["ref"]))
        if snap.withdrawn or snap.status is k.KnowledgeStatus.DEPRECATED:
            raise ValueError("egs_chunk_unavailable")
        c = snap.record
        text = store.read_bounded(c.object_uri, 16384)
        if (
            c.source_version_ref != source.reference()
            or byte_checksum(text) != c.content_checksum
            or chunk["text_checksum"] != c.content_checksum
            or chunk["position"] != dump(c.derivation.position)
            or chunk["object_uri"] != c.object_uri
            or len(text) != c.size_bytes
            or text.decode() != chunk["text"]
        ):
            raise ValueError("egs_chunk_binding_mismatch")


def main():
    from apps.knowledge_source_tools import repository
    from apps.patient_service_assembly import OBJECT_ROOTS
    from packages.knowledge_index.models import Filters

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("ingest", "show", "search"))
    parser.add_argument("--query", default="glaucoma")
    args = parser.parse_args()
    result = (
        ingest(repository("editor"))
        if args.action == "ingest"
        else json.loads((DESTINATION / "content.json").read_text())
    )
    repo = repository("editor")
    store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
    for page in result["pages"]:
        verify_page(repo, store, page)
    if args.action == "search":
        results = []
        for page in result["pages"]:
            if page["index"]["status"] != "indexed":
                results.append({"page": page["id"], "status": page["index"]["status"]})
                continue
            retrieved = repo.query_candidate_index(
                page["index"]["key"],
                args.query,
                Filters(
                    population=("research cohort",),
                    conditions=("glaucoma",),
                    targets=("glaucoma_authoring",),
                ),
            )
            results.append({"page": page["id"], "result": retrieved})
        print(
            json.dumps(
                {"channel": "draft_authoring", "results": results}, ensure_ascii=False, indent=2
            )
        )
        return
    print(
        json.dumps(
            {
                "pages": [
                    {
                        "id": p["id"],
                        "source_ref": p["source_ref"],
                        "chunks": len(p["chunks"]),
                        "index_status": p["index"]["status"],
                        "url": p["url"],
                    }
                    for p in result["pages"]
                ],
                "published": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
