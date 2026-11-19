"""New pool selection semantics; historical intent plans keep their old bytes."""

from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from tests.evaluation_preparation.test_contextual_definition_retrieval import source


def make_sources(version):
    rows = {code: source(code, i + 1) for i, code in enumerate(("abbreviation", "relation"))}
    rows["abbreviation"]["position"]["section_path"] = ["Abbreviations"]
    rows["relation"]["position"]["section_path"] = ["Discussion"]
    rows["relation"]["text"] = "A complete discussion includes both applicability and limitations."
    repo = SimpleNamespace(
        query_candidate_index=lambda *a, **kw: dict(hits=[r["hit"] for r in rows.values()]),
        read_definition_material=lambda hit, *a, **kw: rows[hit["chunk_ref"]["code"]],
    )
    plan = dict(
        schema_version=version,
        queries=["public target"],
        query_intents={"public target": "target"},
        max_per_document=1,
        context_radius=0,
        max_characters=12000,
    )
    return KnowledgeDefinitionSources(
        repository=repo, store=None, index_keys=("original",), passage_plans={"q": plan}
    ), rows


def test_new_discourse_plan_does_not_spend_target_slot_on_abbreviation_section():
    owner, rows = make_sources("definition-passage-plan/3.0.0")
    result = owner.search(query="q", key="test", limit=1, operation=None)
    assert result["materials"] == [rows["relation"]]
    assert any(
        b["reason"] == "non_discourse_section_not_definition_text" and b["code"] == "abbreviation"
        for b in result["backlog"]
    )
    assert result["retrieval"]["selections"][0]["intent_is_entailment"] is False


def test_old_intent_plan_preserves_abbreviation_result_for_historical_compatibility():
    owner, rows = make_sources("definition-passage-plan/2.0.0")
    result = owner.search(query="q", key="test", limit=1, operation=None)
    assert result["materials"] == [rows["abbreviation"]]


def test_complete_pool_applies_earlier_target_query_to_late_source_without_http():
    owner, rows = make_sources("definition-passage-plan/3.0.0")
    seen = []

    def query(key, text, *args, **kwargs):
        seen.append((key, text))
        return dict(hits=[rows["relation"]["hit"]] if key == "late" else [])

    owner.repository.query_candidate_index = query
    owner.index_keys = ("original", "late")
    owner.search_public = lambda **kw: pytest.fail("no HTTP in pool reconciliation")
    result = owner.reconcile_pool(queries=["q"], limit=1)
    assert seen == [("original", "public target"), ("late", "public target")]
    assert result["materials"] == [rows["relation"]]
    assert result["retrieval"]["medical_entailment_proven"] is False


def test_pool_rechecks_current_owner_and_does_not_turn_error_into_empty_materials():
    owner, _ = make_sources("definition-passage-plan/3.0.0")

    def unavailable(*args, **kwargs):
        raise RuntimeError("current_source_unavailable")

    owner.repository.read_definition_material = unavailable
    with pytest.raises(RuntimeError, match="current_source_unavailable"):
        owner.reconcile_pool(queries=["q"], limit=1)


def test_pool_guard_and_old_version_are_not_bypassed():
    owner, _ = make_sources("definition-passage-plan/2.0.0")
    with pytest.raises(ValueError, match="pool_plan_version"):
        owner.reconcile_pool(queries=["q"], limit=1)
    owner, _ = make_sources("definition-passage-plan/3.0.0")

    def cancelled():
        raise RuntimeError("cancelled")

    owner.before_read = cancelled
    with pytest.raises(RuntimeError, match="cancelled"):
        owner.reconcile_pool(queries=["q"], limit=1)


def test_complete_pool_trial_has_explicit_binding_and_no_extra_remote_budget(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial
    from tests.evaluation_preparation.test_component_trial import envelope

    value, spec = envelope(
        tmp_path, monkeypatch, "configs/research/multimodal_definition_gate_v18.json"
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.13.0"
    assert value["expansion_policy"]["schema_version"] == "campaign-definition-expansion/1.18.0"
    assert value["expansion_policy"]["max_search_requests"] == 2
    assert value["expansion_policy"]["max_fetch_requests"] == 2
    assert definition_trial.checked(value, spec) == value


def test_production_stage_checkpoints_reconciled_materials_and_replays_local_receipt(tmp_path):
    import json
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture

    expansion = fixture(tmp_path)
    expansion.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v18.json").read_bytes())
    )
    owner, rows = make_sources("definition-passage-plan/3.0.0")
    owner.search = lambda **kw: dict(
        schema_version="research-definition-search/1.1.0",
        materials=[rows["abbreviation"]],
        backlog=[],
        acquired_indexes=[],
    )
    expansion.sources = owner
    expansion.query_plan = lambda: [dict(query="q", selected=True, retrieval_mode=None)]
    result = expansion._advance(until="sources")
    assert result["materials"] == [rows["relation"]]
    revision = result["backlog"][-1]
    assert revision["previous_codes"] == ["abbreviation"]
    assert revision["selected_codes"] == ["relation"]
    operations = expansion.operations()
    assert any(o["kind"] == "local_pool_reconciliation" for o in operations)
    owner.reconcile_pool = lambda **kw: pytest.fail("confirmed operation must not repeat")
    assert expansion._advance(until="sources") == result
    assert expansion.operations() == operations


@pytest.mark.parametrize("kind", ["local_pool_reconciliation", "llm", "fetch_group"])
def test_pool_pending_recovery_is_local_only(tmp_path, kind):
    import json
    from pathlib import Path

    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture

    expansion = fixture(tmp_path)
    expansion.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v18.json").read_bytes())
    )
    expansion.documents.write(
        "definition:operation:0", (dict(key="pending", kind=kind, binding=expansion.binding),)
    )
    calls = []
    if kind != "local_pool_reconciliation":
        with pytest.raises(ProviderError, match="unknown_no_repeat"):
            expansion.operation("pending", kind, lambda: calls.append(True))
        assert calls == []
    else:
        assert expansion.operation("pending", kind, lambda: dict(local=True)) == dict(local=True)
        assert len(expansion.operations()) == 1
        assert expansion.operation("pending", kind, lambda: calls.append(True)) == dict(local=True)
        assert calls == []


@pytest.mark.parametrize("capacity", [1, 5])
def test_small_final_material_quota_does_not_skip_second_planned_source(tmp_path, capacity):
    import json
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture

    expansion = fixture(tmp_path)
    raw = json.loads(Path("configs/research/multimodal_definition_gate_v18.json").read_bytes())
    raw["max_candidates"] = capacity
    expansion.policy = read_expansion_policy(raw)
    owner, rows = make_sources("definition-passage-plan/3.0.0")
    owner.passage_plans["second"] = owner.passage_plans["q"]
    called = []

    def search(**kwargs):
        called.append(kwargs["query"])
        return dict(
            schema_version="research-definition-search/1.1.0",
            materials=[source(f"{kwargs['query']}-{i}") for i in range(kwargs["limit"])],
            backlog=[],
            acquired_indexes=[],
        )

    owner.search = search
    expansion.sources = owner
    expansion.query_plan = lambda: [
        dict(query=q, selected=True, retrieval_mode=None) for q in ("q", "second")
    ]
    result = expansion._advance(until="sources")
    assert called == ["q", "second"]
    assert len(result["materials"]) == 1
