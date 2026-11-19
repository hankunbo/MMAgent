"""More current source context, not more independent evidence or desired answers."""

import copy
import json
from pathlib import Path

import pytest

from packages.evidence_reuse.definition_sources import read_passage_plan
from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.evaluation_preparation.test_definition_source_balance import fixture


def owner():
    source, rows, queries, reads = fixture(
        ("old-a", "old-b", "new-c"), "definition-passage-plan/6.0.0"
    )
    for row in rows.values():
        row["position"]["kind"] = "paragraph"
        row["hit"]["position"] = dict(row["position"])
    return source, rows, queries, reads


def policy():
    value = json.loads(
        Path("configs/research/multimodal_definition_numeric_interpretation_v1.json").read_bytes()
    )
    value.update(
        schema_version="campaign-definition-expansion/1.38.0",
        prompt_version="definition-draft-6.8.0",
        retrieval_scope="confirmed-index-only/1.0.0",
        unresolved_relation_retrieval=None,
        max_candidates=16,
    )
    for plan in value["passage_plans"].values():
        plan.update(schema_version="definition-passage-plan/6.0.0", max_per_document=8)
    return value


def test_current_material_pool_retains_multiple_paragraphs_without_new_http():
    source, _, _, reads = owner()
    source.search_public = lambda **kw: pytest.fail("no new public search")
    result = source.reconcile_pool(queries=["q"], limit=16)
    assert len(result["materials"]) == 9
    assert len(reads) == len(set(reads)) == 9
    assert {m["source_version"]["code"] for m in result["materials"]} == {"old-a", "old-b", "new-c"}
    assert result["retrieval"]["medical_entailment_proven"] is False
    assert all(not s["intent_is_entailment"] for s in result["retrieval"]["selections"])


def test_old_material_limit_and_explicit_new_bounds_remain_enforced():
    old, _, _, _ = fixture(("old-a",))
    with pytest.raises(ValueError, match="definition_material_limit"):
        old.reconcile_pool(queries=["q"], limit=6)
    source, _, _, _ = owner()
    for limit in (0, 17, True):
        with pytest.raises(ValueError, match="definition_material_limit"):
            source.reconcile_pool(queries=["q"], limit=limit)
    raw = source.passage_plans["q"].model_dump(mode="json")
    for change in ({"max_per_document": 9}, {"max_characters": 16001}):
        with pytest.raises(ValueError):
            read_passage_plan(dict(raw, **change))


def test_empty_extended_search_cannot_fall_through_to_http_or_discovery():
    source, _, _, _ = owner()
    source.repository.query_candidate_index = lambda *a, **kw: dict(hits=[])
    source.search_public = source.fetch_hit = lambda **kw: pytest.fail("local-only plan")
    value = source.search(query="q", key="local", limit=16, operation=None)
    assert value["materials"] == [] and value["reused"] is True
    with pytest.raises(ValueError, match="definition_bounded_pool_local_only"):
        source.search(
            query="q",
            key="bad",
            limit=16,
            operation=None,
            retrieval_mode="unseen-first-for-owner-gap/1.0.0",
        )


def test_character_bound_and_current_source_failure_not_bypassed():
    source, rows, _, _ = owner()
    for row in rows.values():
        row["text"] = "source context " * 250
    value = source.reconcile_pool(queries=["q"], limit=16)
    assert sum(len(m["text"]) for m in value["materials"]) <= 12000
    assert any(x["reason"] == "source_context_character_budget" for x in value["backlog"])

    def changed(*a, **kw):
        raise RuntimeError("source withdrawn")

    source.repository.read_definition_material = changed
    with pytest.raises(RuntimeError, match="source withdrawn"):
        source.reconcile_pool(queries=["q"], limit=16)


def test_mixed_versions_cannot_expand_old_pool_or_silently_change_old_plan():
    source, _, _, _ = owner()
    raw = source.passage_plans["q"].model_dump(mode="json")
    raw["schema_version"] = "definition-passage-plan/5.0.0"
    source.passage_plans["old"] = read_passage_plan(raw)
    with pytest.raises(ValueError):
        source.reconcile_pool(queries=["q", "old"], limit=16)


def test_owner_oversized_material_is_recorded_not_truncated_or_accepted():
    from packages.knowledge_claims.evidence import Rejected

    source, _, _, _ = owner()
    original = source.repository.read_definition_material

    def read(hit, *a, **kw):
        if hit["chunk_ref"]["code"] == "old-a:0":
            raise Rejected("claim_text_limit")
        return original(hit, *a, **kw)

    source.repository.read_definition_material = read
    result = source.reconcile_pool(queries=["q"], limit=16)
    assert len(result["materials"]) == 8
    assert "old-a:0" not in {s["code"] for s in result["materials"]}
    assert any(b["reason"] == "source_chunk_claim_text_limit" for b in result["backlog"])


def test_policy_is_explicit_index_only_with_original_numeric_and_literal_owners():
    from packages.multimodal_research.campaign_renewal import (
        CURRENT_INCREMENT_POLICIES,
        INDEX_ONLY_POLICIES,
    )
    from packages.multimodal_research.definition_expansion import NumericInterpretationPublicPolicy

    raw = policy()
    value = read_expansion_policy(raw)
    assert isinstance(value, NumericInterpretationPublicPolicy)
    assert value.schema_version in INDEX_ONLY_POLICIES & CURRENT_INCREMENT_POLICIES
    assert read_expansion_policy(value.model_dump(mode="json")) == value
    for change in (
        {"schema_version": "campaign-definition-expansion/1.37.0"},
        {"max_candidates": 17},
        {"unresolved_relation_retrieval": "unseen-first-for-owner-gap/1.0.0"},
    ):
        with pytest.raises(ValueError):
            read_expansion_policy(dict(raw, **change))


def test_production_checkpoint_and_literal_projection_preserve_context_on_recovery(tmp_path):
    from packages.llm_provider.definition_passages import project
    from tests.evaluation_preparation.test_definition_expansion import fixture as stage_fixture

    stage = stage_fixture(tmp_path)
    stage.policy = read_expansion_policy(policy())
    source, _, _, _ = owner()
    source.search = lambda **kw: dict(
        schema_version="research-definition-search/1.1.0",
        materials=[],
        acquired_indexes=[],
        backlog=[],
    )
    stage.sources = source
    stage.query_plan = lambda: [dict(query="q", selected=True, retrieval_mode=None)]
    result = stage._advance(until="sources")
    assert len(result["materials"]) == 9
    payload = dict(target="GOAL:EXAMPLE", source_materials=result["materials"])
    original = copy.deepcopy(payload)
    wire = project(payload)
    assert payload == original
    passages = wire["definition_passage_contract"]["passages"]
    for material in result["materials"]:
        assert (
            "".join(p["text"] for p in passages if p["source_code"] == material["code"])
            == material["text"]
        )
    source.reconcile_pool = lambda **kw: pytest.fail("confirmed material must not be reselected")
    assert stage._advance(until="sources") == result
