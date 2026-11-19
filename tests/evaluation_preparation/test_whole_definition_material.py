"""Research context capacity never implies medical entailment or claim publication."""

import copy
from types import SimpleNamespace

import pytest

from db.knowledge.repository import claims
from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources, read_passage_plan
from packages.knowledge_claims.evidence import Rejected
from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.evaluation_preparation.test_bounded_definition_material_pool import policy
from tests.evaluation_preparation.test_definition_source_balance import fixture


def new_policy():
    value = policy()
    value["schema_version"] = "campaign-definition-expansion/1.39.0"
    for plan in value["passage_plans"].values():
        plan["schema_version"] = "definition-passage-plan/7.0.0"
    return value


def test_explicit_versions_preserve_prior_policy_and_dispatch():
    from packages.multimodal_research.campaign_renewal import (
        INDEX_ONLY_POLICIES,
        REVISED_ACQUISITION_POLICIES,
    )
    from packages.multimodal_research.definition_expansion import NumericInterpretationPublicPolicy

    old = policy()
    before = copy.deepcopy(old)
    assert read_expansion_policy(old).schema_version == "campaign-definition-expansion/1.38.0"
    assert old == before
    current = read_expansion_policy(new_policy())
    assert isinstance(current, NumericInterpretationPublicPolicy)
    assert current.schema_version in INDEX_ONLY_POLICIES & REVISED_ACQUISITION_POLICIES
    assert {p.schema_version for p in current.passage_plans.values()} == {
        "definition-passage-plan/7.0.0"
    }
    changed = new_policy()
    next(iter(changed["passage_plans"].values()))["schema_version"] = (
        "definition-passage-plan/6.0.0"
    )
    with pytest.raises(ValueError):
        read_expansion_policy(changed)


def test_new_selection_and_revalidation_use_same_original_reader_profile():
    source, rows, _, _ = fixture(("old-a",), "definition-passage-plan/7.0.0")
    calls = []
    for row in rows.values():
        row["position"]["kind"] = "paragraph"
        row["hit"]["position"] = dict(row["position"])
        row["text"] = "Observation context. " * 130

    def read(hit, store, *, filters, material_profile):
        calls.append((hit["chunk_ref"]["code"], material_profile))
        assert material_profile == "research-definition-material/1.1.0"
        return rows[hit["chunk_ref"]["code"]]

    source.repository.read_definition_material = read
    source.repository.read_definition_materials = lambda hits, store, **kw: [
        read(h, store, **kw) for h in hits
    ]
    pool = source.reconcile_pool(queries=["q"], limit=3)
    assert len(pool["materials"]) == 3
    assert all(len(s["text"]) > 2000 for s in pool["materials"])
    assert pool["retrieval"]["material_reader"]["complete_blocks_only"] is True
    assert pool["retrieval"]["medical_entailment_proven"] is False
    codes = [m["code"] for m in pool["materials"]]
    source.restore(pool["materials"])
    assert source.read(codes[0]) == rows[codes[0]]
    assert source.read_many(codes) == pool["materials"]
    assert len(calls) == 7  # Fresh original-owner calls, not cached qualification.


def test_whole_block_overflow_is_explicit_backlog_not_truncation():
    source, rows, _, _ = fixture(("old-a",), "definition-passage-plan/7.0.0")
    for row in rows.values():
        row["position"]["kind"] = "paragraph"
        row["hit"]["position"] = dict(row["position"])

    def read(hit, store, **kw):
        if hit["chunk_ref"]["code"] == "old-a:0":
            raise Rejected("definition_source_requires_bounded_chunk")
        return rows[hit["chunk_ref"]["code"]]

    source.repository.read_definition_material = read
    result = source.reconcile_pool(queries=["q"], limit=3)
    assert len(result["materials"]) == 2
    assert any(
        x["reason"] == "source_chunk_definition_source_requires_bounded_chunk"
        for x in result["backlog"]
    )


@pytest.mark.parametrize(
    "version", ["definition-passage-plan/6.0.0", "definition-passage-plan/7.0.0"]
)
def test_invalid_profile_cannot_reach_database_and_old_reader_not_promoted(version):
    source, _, _, _ = fixture(("old-a",), version)
    assert bool(source._material_options()) == version.endswith("/7.0.0")
    repo = SimpleNamespace(_role="mmagent_knowledge_editor")
    with pytest.raises(ValueError, match="definition_material_profile"):
        claims.read_definition_materials(repo, [{}], None, filters={}, material_profile="unbounded")


def test_mixed_reader_profiles_refused_before_source_consumption():
    source, _, _, _ = fixture(("old-a",), "definition-passage-plan/7.0.0")
    raw = source.passage_plans["q"].model_dump(mode="json")
    raw["schema_version"] = "definition-passage-plan/6.0.0"
    with pytest.raises(ValueError, match="definition_mixed_passage_plan_versions"):
        KnowledgeDefinitionSources(
            repository=None,
            store=None,
            index_keys=(),
            passage_plans={"q": source.passage_plans["q"], "other": read_passage_plan(raw)},
        )
