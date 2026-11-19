"""Intent coverage and confirmed index addresses do not confer medical authority."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.grounding.models import digest
from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.evaluation_preparation.test_contextual_definition_retrieval import source


def plan():
    return dict(
        schema_version="definition-passage-plan/2.0.0",
        queries=["advanced", "cup", "differential"],
        query_intents={"advanced": "target", "cup": "examination", "differential": "alternative"},
        max_per_document=3,
        context_radius=0,
        max_characters=12000,
    )


def owner():
    rows = {c: source(c) for c in ("target", "citation", "cup", "alternative")}
    rows["citation"]["position"]["section_path"] = ["References"]
    candidates = {
        "advanced": ["target"],
        "cup": ["citation", "cup"],
        "differential": ["alternative"],
    }
    reads = []

    def query(k, q, filters, **options):
        assert options == dict(fts_single_pass=True)
        reads.append((k, q))
        return dict(hits=[rows[c]["hit"] for c in candidates[q]])

    repo = SimpleNamespace(
        query_candidate_index=query,
        read_definition_material=lambda hit, *a, **kw: rows[hit["chunk_ref"]["code"]],
        read_definition_context=lambda *a, **kw: pytest.fail("context disabled"),
    )
    s = KnowledgeDefinitionSources(
        repository=repo, store=None, index_keys=("original",), passage_plans={"q": plan()}
    )
    return s, rows, reads


def test_intent_coverage_precedes_quota_and_bad_bibliography_does_not_consume_turn():
    s, rows, reads = owner()
    value = s.search(query="q", key="q1", limit=3, operation=None)
    assert [s["code"] for s in value["materials"]] == ["cup", "target", "alternative"]
    assert value["acquired_indexes"] == []
    assert value["schema_version"] == "research-definition-search/1.1.0"
    assert len(reads) == 3
    assert value["retrieval"]["selections"][0]["query_matches"][0]["query"] == "cup"
    assert all(not s["intent_is_entailment"] for s in value["retrieval"]["selections"])
    assert any(b["reason"] == "bibliography_not_definition_text" for b in value["backlog"])
    assert rows["cup"]["text"] == value["materials"][0]["text"]


def test_old_plan_retains_original_query_order_and_quota_semantics():
    s, _, _ = owner()
    raw = plan()
    raw.pop("query_intents")
    raw.update(schema_version="definition-passage-plan/1.0.0", max_per_document=1)
    old = KnowledgeDefinitionSources(
        repository=s.repository, store=None, index_keys=("original",), passage_plans={"q": raw}
    )
    value = old.search(query="q", key="q1", limit=3, operation=None)
    assert [x["code"] for x in value["materials"]] == ["target"]
    assert "acquired_indexes" not in value


def test_exact_intent_map_and_versioned_expansion_policy():
    p = plan()
    p["query_intents"].pop("cup")
    with pytest.raises(ValueError, match="intent_binding"):
        KnowledgeDefinitionSources(repository=None, store=None, passage_plans={"q": p})
    raw = json.loads(Path("configs/research/multimodal_definition_gate_v13.json").read_bytes())
    current = read_expansion_policy(raw)
    assert current.schema_version == "campaign-definition-expansion/1.13.0"
    assert current.max_fetch_requests == current.max_search_requests == 2
    second = current.passage_plans[current.public_queries[1]]
    assert set(second.query_intents.values()) == {"examination", "alternative"}
    # Five paragraph slots are allocated 3+2 by the original contextual owner.
    # Do not spend the second question's two slots on a repeated target intent.
    assert current.max_candidates == 5
    old = json.loads(Path("configs/research/multimodal_definition_gate_v12.json").read_bytes())
    assert read_expansion_policy(old).schema_version == old["schema_version"]
    with pytest.raises(ValueError):
        read_expansion_policy({**raw, "passage_plans": old["passage_plans"]})


def test_confirmed_pool_cold_restore_is_exact_and_current_retrieval_still_runs():
    s, _, reads = owner()
    receipt = dict(
        status="indexed",
        key="obtained",
        source_version_ref=dict(code="SOURCE:V"),
        chunk_refs=[dict(code="CHUNK:A")],
        config_checksum=digest("frozen"),
    )
    calls = []
    s.repository.replay_document_index = lambda key, store: (
        calls.append(key) or copy.deepcopy(receipt)
    )
    s.restore_indexes([receipt])
    assert s.index_keys == ("original", "obtained")
    s.restore_indexes([receipt])
    assert s.index_keys == ("original", "obtained")
    cold, _, _ = owner()
    cold.repository.replay_document_index = s.repository.replay_document_index
    cold.restore_indexes([receipt])
    assert cold.index_keys == s.index_keys and len(calls) == 3
    bad = {**receipt, "config_checksum": digest("tampered")}
    with pytest.raises(ValueError, match="receipt_binding"):
        s.restore_indexes([bad])
    s.search(query="q", key="second", limit=3, operation=None)
    assert ("obtained", "cup") in reads

    def withdrawn(*a, **kw):
        raise ValueError("original_current_source_withdrawn")

    s.repository.query_candidate_index = withdrawn
    with pytest.raises(ValueError, match="original_current_source_withdrawn"):
        s.search(query="q", key="later", limit=3, operation=None)


def test_zero_capacity_and_cancel_never_query_pool_or_remote():
    s, _, reads = owner()
    with pytest.raises(ValueError, match="material_limit"):
        s.search(query="q", key="zero", limit=0, operation=None)

    def cancelled():
        raise RuntimeError("cancelled")

    s.before_read = cancelled
    with pytest.raises(RuntimeError, match="cancelled"):
        s.search(query="q", key="cancel", limit=3, operation=None)
    assert not reads
