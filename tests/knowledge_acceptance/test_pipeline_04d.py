"""Independent real-PG 04D composition, source graphs and immutable replay probes."""

import json
import random

import pytest

from apps.knowledge_retrieval import PublishedKnowledgeService, SourceCandidateService
from db.knowledge import models as m
from db.knowledge.repository import KnowledgeConflictError, KnowledgeNotFoundError
from packages.knowledge_archive import Bibliography
from packages.knowledge_index.models import IndexConfig
from packages.knowledge_index.ports import FakeEmbedder, FakeReranker
from packages.object_store import LocalObjectStore, checksum
from packages.retrieval.knowledge.models import RetrievalContext, RetrievalIntegrityError
from tests.database.knowledge_support_04a import free_port, prepare_database
from tests.database.test_knowledge_archive_04b3 import fetch, hit, other_provider_hit, version
from tests.knowledge_retrieval.support import assemble


@pytest.fixture(scope="module")
def database(database_lease):
    with database_lease.cluster(free_port()):
        yield prepare_database()


def fts_config():
    return IndexConfig(mode="fts_only", vector_weight=0.0, vector_unavailable_reason="04D_fixture")


def test_repeated_discovery_reuses_canonical_source_index(database, tmp_path):
    store = LocalObjectStore(tmp_path / "objects")
    h = hit(database, title="04D exact repeated discovery")
    bib = Bibliography(language="en")
    first = database.editor.archive_hit(h.reference(), fetch(h, bibliography=bib), store)
    second = database.editor.archive_hit(
        h.reference(), fetch(h, bibliography=bib, seconds=10), store
    )
    assert second["decision"] == "reused"
    assert first["version_ref"] == second["version_ref"]
    assert first["object_uri"] != second["object_uri"]
    condition = m.seal(
        m.ApplicabilityCondition,
        code="SOURCE:04D:REPEATED_CONDITION",
        version="1.0.0",
        synthetic=True,
        population=("synthetic_fixture_only",),
        conditions=("software_test",),
        modalities=("synthetic",),
        scope_note="Synthetic source reuse, no clinical interpretation",
    )
    database.editor.put(condition)
    args = dict(applicability_refs=(condition.reference(),))
    original = database.editor.build_document_index(first, store, fts_config(), **args)
    repeated = database.editor.build_document_index(second, store, fts_config(), **args)
    assert repeated == original
    assert store.read(first["object_uri"]) == store.read(second["object_uri"])
    with pytest.raises(KnowledgeConflictError, match="exact persisted"):
        database.editor.build_document_index(
            second | {"object_uri": first["object_uri"]}, store, fts_config(), **args
        )
    source = version(database, first)
    canonical_path = store.root / source.object_uri.removeprefix("object://knowledge/")
    body = canonical_path.read_bytes()
    canonical_path.write_bytes(bytes([body[0] ^ 1]) + body[1:])
    try:
        with pytest.raises(KnowledgeConflictError, match="actual source bytes"):
            database.editor.build_document_index(second, store, fts_config(), **args)
    finally:
        canonical_path.write_bytes(body)


def test_two_providers_and_reversed_visits_share_one_index(database, tmp_path):
    store = LocalObjectStore(tmp_path / "objects")
    a = hit(database, identifiers={"pmid": "999990001", "doi": "10.9999/synthetic-04b1"})
    b = other_provider_hit(database, "pubmed")
    bib = Bibliography(title="Independent 04D two-provider identity", language="en")
    condition = m.seal(
        m.ApplicabilityCondition,
        code="SOURCE:04D:MULTIPROVIDER_CONDITION",
        version="1.0.0",
        synthetic=True,
        population=("synthetic_fixture_only",),
        conditions=("software_test",),
        scope_note="Synthetic multi-provider source scope",
    )
    database.editor.put(condition)
    receipts, builds = [], []
    for h in (a, b, b, a):
        receipt = database.editor.archive_hit(
            h.reference(),
            fetch(h, bibliography=bib, body=b"<article>independent two-provider source</article>"),
            store,
        )
        receipts.append(receipt)
        builds.append(
            database.editor.build_document_index(
                receipt, store, fts_config(), applicability_refs=(condition.reference(),)
            )
        )
    assert all(build == builds[0] for build in builds)
    assert receipts[0]["provider_ref"] != receipts[1]["provider_ref"]
    assert receipts[0]["version_ref"] == receipts[1]["version_ref"]
    assert receipts[0]["object_uri"] != receipts[1]["object_uri"]


@pytest.mark.parametrize("seed", [401, 402, 403])
def test_random_small_source_discovery_graph_keeps_exact_identity(database, tmp_path, seed):
    rng = random.Random(seed)
    store = LocalObjectStore(tmp_path / "objects")
    identities = [str(seed * 1000 + i) for i in range(rng.randint(2, 4))]
    visits = identities * 2
    rng.shuffle(visits)
    observed = {}
    for identity in visits:
        h = hit(
            database,
            identifiers={"pmid": identity},
            title=f"Synthetic independent graph {identity}",
        )
        body = f"<article><p>synthetic source {identity}</p></article>".encode()
        receipt = database.editor.archive_hit(h.reference(), fetch(h, body=body), store)
        source = version(database, receipt)
        expected = observed.setdefault(identity, source.reference())
        assert source.reference() == expected
        assert source.content_checksum == checksum(body)
        assert store.read(source.object_uri) == body
    assert len(set(observed.values())) == len(identities)
    catalog = database.editor.archive_catalog()
    for ref in observed.values():
        paths = [
            o for o in catalog["observations"] if o["version_ref"] == ref.model_dump(mode="json")
        ]
        assert len(paths) == 2
        assert len({p["hit_ref"]["code"] for p in paths}) == 2


def test_four_fake_pipeline_scope_cache_and_one_byte_corruption(database, tmp_path):
    store = LocalObjectStore(tmp_path / "objects")
    embedder, reranker = FakeEmbedder(), FakeReranker()
    cfg = IndexConfig(
        mode="hybrid", embedder=embedder.spec, reranker=reranker.spec, reranker_mode="enabled"
    )
    # assemble uses recorded Fake Connector, FakeStructuredExtractor and real services/SQL.
    data = assemble(database, store, cfg, scoped=True, graph=True, embedder=embedder)
    ctx = RetrievalContext(
        population=("synthetic_fixture_only", "unspecified_in_source"),
        conditions=("software_test",),
        modalities=("synthetic",),
        anatomy=("synthetic_retina",),
        targets=("synthetic-target",),
        disease_packs=("synthetic-pack",),
        sexes=("female",),
        age_years=18,
        exams=("OCT",),
        times=("baseline",),
        lateralities=("RIGHT",),
        devices=("synthetic-device",),
    )
    reader = PublishedKnowledgeService(database.reader, store)
    history = SourceCandidateService(database.editor, store)
    ref = data["release"].reference()
    before = history.replay_release(ref)
    expected = {data[name].reference() for name in ("claim", "template", "rule")}
    for age in (18, 65):
        for population in (ctx.population, tuple(reversed(ctx.population))):
            request = ctx.model_copy(update={"age_years": age, "population": population})
            result = reader.retrieve_published_knowledge(
                ref, "OCT", request, cfg, embedder=embedder, reranker=reranker
            )
            assert {h["artifact"].record.reference() for h in result["hits"]} == expected
            assert all(h["components"]["reranker"] is not None for h in result["hits"])
    for age in (17, 66, None):
        rejected = reader.retrieve_published_knowledge(
            ref,
            "OCT",
            ctx.model_copy(update={"age_years": age}),
            cfg,
            embedder=embedder,
            reranker=reranker,
        )
        assert rejected["hits"] == () and rejected["outcome"] == "no_applicable_knowledge"
    # A self-consistent candidate cache entry is still not a publication capability.
    from packages.knowledge_index.models import digest

    for cached in reader._cache.values():
        cached["hits"] = [{"channel": "candidate", "synthetic_poison": True}]
        cached["checksum"] = digest(cached["hits"])
    result = reader.retrieve_published_knowledge(
        ref, "OCT", ctx, cfg, embedder=embedder, reranker=reranker
    )
    assert {h["artifact"].record.reference() for h in result["hits"]} == expected
    claim = data["claim"]
    assert claim.candidate.provenance.method == "LLM"
    source = database.editor.replay(claim.source_refs[0]).record
    path = store.root / source.object_uri.removeprefix("object://knowledge/")
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    try:
        with pytest.raises(RetrievalIntegrityError):
            history.replay_release(ref)
        with pytest.raises(RetrievalIntegrityError):
            reader.retrieve_published_knowledge(
                ref, "OCT", ctx, cfg, embedder=embedder, reranker=reranker
            )
    finally:
        path.write_bytes(original)
    after = history.replay_release(ref)
    assert before["material"] == after["material"] and before["checksum"] == after["checksum"]
    database.publisher.withdraw(source.reference(), "RULE:SYNTHETIC_04D_WITHDRAWAL")
    with pytest.raises(KnowledgeNotFoundError):
        reader.retrieve_published_knowledge(
            ref, "OCT", ctx, cfg, embedder=embedder, reranker=reranker
        )
    withdrawn = history.replay_release(ref)
    assert withdrawn["material"] == before["material"]
    assert not withdrawn["current"]["available"]
    (tmp_path / "04d-e2e.json").write_text(
        json.dumps(
            {
                "synthetic_only": True,
                "release": ref.model_dump(mode="json"),
                "manifest_checksum": before["checksum"],
                "material": before["material"],
                "config": cfg.model_dump(mode="json"),
                "objects_verified": True,
                "current_withdrawal_blocks_read": True,
                "frozen_history_preserved": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
