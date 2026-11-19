"""Source-balanced recall is not source qualification or medical entailment."""

import json
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from tests.evaluation_preparation.test_contextual_definition_retrieval import source


def fixture(order, version="definition-passage-plan/4.0.0"):
    rows = {}
    for document in ("old-a", "old-b", "new-c"):
        for index in range(3):
            code = f"{document}:{index}"
            row = source(code, index + 1)
            row["source_version"] = dict(code=document)
            row["hit"]["source_version_ref"] = row["source_version"]
            row["position"]["section_path"] = ["Discussion"]
            rows[code] = row
    queries, reads = [], []

    def query(key, text, filters, **options):
        assert options == dict(fts_single_pass=True)
        queries.append((key, text))
        return dict(hits=[r["hit"] for r in rows.values() if r["source_version"]["code"] == key])

    def read(hit, store, *, filters):
        code = hit["chunk_ref"]["code"]
        reads.append(code)
        return rows[code]

    plan = dict(
        schema_version=version,
        queries=["examination", "target", "alternative"],
        query_intents={key: key for key in ("examination", "target", "alternative")},
        max_per_document=3,
        context_radius=0,
        max_characters=12000,
    )
    owner = KnowledgeDefinitionSources(
        repository=SimpleNamespace(query_candidate_index=query, read_definition_material=read),
        store=None,
        index_keys=tuple(order),
        passage_plans={"q": plan},
    )
    return owner, rows, queries, reads


def test_balanced_selection_is_index_order_invariant_and_covers_sources():
    selections = []
    for order in permutations(("old-a", "old-b", "new-c")):
        owner, _, queries, reads = fixture(order)
        value = owner.reconcile_pool(queries=["q"], limit=5)
        selections.append([m["code"] for m in value["materials"]])
        assert len(queries) == 9
        assert len(reads) == 5
        assert {m["source_version"]["code"] for m in value["materials"]} == set(order)
        assert value["retrieval"]["medical_entailment_proven"] is False
        decisions = value["retrieval"]["candidate_decisions"]
        assert len(decisions) == 9
        assert sum(d["reason"] == "selected" for d in decisions) == 5
        assert sum(d["reason"] == "material_limit_not_evaluated" for d in decisions) == 4
        assert all(not d["current_material_read"] for d in decisions if d["reason"] != "selected")
    assert all(s == selections[0] for s in selections)


def test_old_plan_retains_index_order_for_historical_replay():
    owner, _, _, _ = fixture(("old-a", "old-b", "new-c"), "definition-passage-plan/3.0.0")
    value = owner.reconcile_pool(queries=["q"], limit=5)
    assert not any(m["code"].startswith("new-c:") for m in value["materials"])
    assert "candidate_decisions" not in value["retrieval"]


def test_balanced_selection_preserves_current_owner_failure():
    owner, _, _, _ = fixture(("old-a", "old-b", "new-c"))

    def unavailable(*args, **kwargs):
        raise RuntimeError("current_source_unavailable")

    owner.repository.read_definition_material = unavailable
    with pytest.raises(RuntimeError, match="current_source_unavailable"):
        owner.reconcile_pool(queries=["q"], limit=5)


def test_balanced_selection_records_non_discourse_not_a_medical_rejection():
    owner, rows, _, _ = fixture(("old-a", "old-b", "new-c"))
    for row in rows.values():
        row["position"]["section_path"] = ["References"]
    value = owner.reconcile_pool(queries=["q"], limit=5)
    assert value["materials"] == []
    assert len(value["retrieval"]["candidate_decisions"]) == 9
    assert all(
        d["reason"] == "bibliography_not_definition_text"
        for d in value["retrieval"]["candidate_decisions"]
    )


def test_balanced_selection_requires_bound_source_identity():
    owner, rows, _, _ = fixture(("old-a", "old-b", "new-c"))
    for row in rows.values():
        row["hit"].pop("source_version_ref")
    with pytest.raises(ValueError, match="definition_pool_source_identity"):
        owner.reconcile_pool(queries=["q"], limit=5)


def test_new_trial_policy_is_explicit_cold_readable_and_budget_preserving(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial
    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_trial_entry_window import prepare_entry

    _, spec = prepare_entry(tmp_path, monkeypatch)
    path = "configs/research/multimodal_definition_gate_v20.json"
    policy = read_expansion_policy(json.loads(Path(path).read_bytes()))
    assert {p.schema_version for p in policy.passage_plans.values()} == {
        "definition-passage-plan/4.0.0"
    }
    value = definition_trial.prepare(
        tmp_path,
        expansion_policy=path,
        authorization="synthetic source balance binding",
        seconds=5400,
        entry_time_window=True,
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.16.0"
    assert value["max_new_requests"] == 12
    assert value["expansion_policy"]["max_fetch_requests"] == 2
    assert definition_trial.read(tmp_path) == value
    value["schema_version"] = "multimodal-definition-trial/1.15.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        definition_trial.checked(value, spec)


def test_mixed_new_old_pool_plans_reject():
    from packages.evidence_reuse.definition_sources import read_passage_plan

    owner, _, _, _ = fixture(("old-a", "old-b", "new-c"))
    old = owner.passage_plans["q"].model_dump(mode="json")
    old["schema_version"] = "definition-passage-plan/3.0.0"
    owner.passage_plans["old"] = read_passage_plan(old)
    with pytest.raises(ValueError, match="definition_pool_plan_version"):
        owner.reconcile_pool(queries=["q", "old"], limit=5)


def test_original_stage_consumes_new_pool_and_confirmed_replay_does_not_query(tmp_path):
    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    expansion = expansion_fixture(tmp_path)
    expansion.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v20.json").read_bytes())
    )
    owner, _, _, _ = fixture(("old-a", "old-b", "new-c"))
    owner.search = lambda **kw: dict(
        schema_version="research-definition-search/1.1.0",
        materials=[],
        backlog=[],
        acquired_indexes=[],
    )
    expansion.sources = owner
    expansion.query_plan = lambda: [dict(query="q", selected=True, retrieval_mode=None)]
    result = expansion._advance(until="sources")
    assert len(result["materials"]) == 5
    assert len({m["source_version"]["code"] for m in result["materials"]}) == 3
    assert len(result["backlog"][-1]["retrieval"]["candidate_decisions"]) == 9
    operations = expansion.operations()
    owner.reconcile_pool = lambda **kw: pytest.fail("confirmed pool must not repeat")
    assert expansion._advance(until="sources") == result
    assert expansion.operations() == operations
