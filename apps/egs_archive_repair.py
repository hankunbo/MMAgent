"""Resolve the archived EGS duplicate, index original bytes, preserve v1 manifests."""

import json

from apps.glaucoma_egs_content import DESTINATION, LINKS, verify_page
from apps.glaucoma_evidence_bindings import ROOT, dump, ref, write_new
from db.knowledge import models as k
from packages.knowledge_index.models import Filters, IndexConfig
from packages.object_store import checksum


def repair(repo, store):
    previous = ROOT / "docs/experiments/glaucoma-egs-content-v1/content.json"
    result = json.loads(previous.read_text())
    page = next(p for p in result["pages"] if p["id"] == "EGS-HEMORRHAGE-2024")
    receipt = page["archive"]
    catalog = repo.archive_catalog()
    comparisons = [
        o
        for o in catalog["observations"]
        if o["hit_ref"] == receipt["hit_ref"]
        and not o["candidate_keys"]
        and o["key"] != receipt["key"]
    ]
    if len(comparisons) != 1 or len(receipt["candidate_keys"]) != 1:
        raise ValueError("exact_single_duplicate_required")
    proof = repo.resolve_archive_equivalence(
        receipt["candidate_keys"][0], receipt["key"], comparisons[0]["key"], store
    )
    context = k.seal(
        k.ApplicabilityCondition,
        code="EVIDENCE:GLAUCOMA:EGS_AUTHORING_CONTEXT",
        version="1.0.0",
        population=("research cohort",),
        conditions=("glaucoma",),
        scope_note="Education authoring; not patient eligibility or approved clinical rules",
    )
    if repo.get(context.reference()).record != context:
        raise ValueError("authoring_context_mismatch")
    index = repo.build_document_index(
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
    if index["status"] != "indexed":
        raise ValueError("index_not_completed")
    chunks = []
    for reference in index["chunk_refs"]:
        c = repo.get(ref(reference)).record
        text = store.read_bounded(c.object_uri, 16384)
        if checksum(text) != c.content_checksum or len(text) != c.size_bytes:
            raise ValueError("chunk_content_mismatch")
        chunks.append(
            {
                "ref": reference,
                "position": dump(c.derivation.position),
                "object_uri": c.object_uri,
                "text_checksum": c.content_checksum,
                "text": text.decode(),
            }
        )
    page.update(index=index, chunks=chunks, equivalence=proof)
    filters = Filters(
        population=("research cohort",), conditions=("glaucoma",), targets=("glaucoma_authoring",)
    )
    retrieval = repo.query_candidate_index(index["key"], "hemorrhage", filters)
    if not retrieval["hits"] or not any(
        len(c["text"]) > 100 and any(t in c["text"].casefold() for t in LINKS[page["id"]]["terms"])
        for c in chunks
    ):
        raise ValueError("substantive_fts_readback_missing")
    after = repo.archive_catalog()
    if after["records"] != catalog["records"] or after["observations"] != catalog["observations"]:
        raise ValueError("archive_history_changed")
    for p in result["pages"]:
        verify_page(repo, store, p)
        write_new(DESTINATION / (p["id"] + ".json"), p)
    write_new(DESTINATION / "content.json", result)
    evidence = {
        "equivalence": proof,
        "index": index,
        "retrieval": retrieval,
        "old_archive_unchanged": True,
        "active_candidate_keys": [c["key"] for c in after["candidates"]],
        "historical_candidate_keys": [c["key"] for c in after["candidate_history"]],
        "all_pages_indexed": all(p["index"]["status"] == "indexed" for p in result["pages"]),
        "total_chunks": sum(len(p["chunks"]) for p in result["pages"]),
    }
    write_new(DESTINATION / "repair-verification.json", evidence)
    return {
        "page": page["id"],
        "chunks": len(chunks),
        "fts_hits": len(retrieval["hits"]),
        "all_pages_indexed": evidence["all_pages_indexed"],
        "total_chunks": evidence["total_chunks"],
    }


def main():
    from apps.knowledge_source_tools import repository
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from packages.object_store import LocalObjectStore

    check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
    print(json.dumps(repair(repository("editor"), LocalObjectStore(OBJECT_ROOTS["knowledge"]))))


if __name__ == "__main__":
    main()
