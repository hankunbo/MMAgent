"""Unresolved owner relations permit bounded discovery, not medical acceptance."""

import json
from pathlib import Path

import pytest

from tests.evaluation_preparation.test_intent_definition_retrieval import owner


def test_local_material_does_not_suppress_explicit_gap_discovery():
    sources, _, _ = owner()
    calls = []
    sources.search_public = lambda **kw: calls.append(kw) or []
    sources.fetch_hit = lambda _: pytest.fail("no hits to fetch")
    value = sources.search(
        query="q",
        key="gap",
        limit=3,
        operation=lambda key, kind, call: call(),
        retrieval_mode="local-and-public-for-owner-gap/1.0.0",
    )
    assert len(calls) == 1
    assert [row["code"] for row in value["materials"]] == ["cup", "target", "alternative"]
    assert value["discovery"]["local_materials"] == 3
    assert value["discovery"]["medical_entailment_proven"] is False


def test_legacy_local_first_does_not_dispatch():
    sources, _, _ = owner()
    sources.search_public = lambda **kw: pytest.fail("legacy local-first")
    sources.fetch_hit = lambda _: pytest.fail("legacy local-first")
    result = sources.search(query="q", key="old", limit=3, operation=None)
    assert result["reused"] and "discovery" not in result


@pytest.mark.parametrize("status", ["public_budget_exhausted", "technical_backlog"])
def test_new_gap_budget_or_technical_stop_keeps_local_materials(status):
    sources, _, _ = owner()
    sources.search_public = lambda **kw: dict(status=status)
    sources.fetch_hit = lambda _: pytest.fail("not dispatched")
    value = sources.search(
        query="q",
        key="gap",
        limit=2,
        operation=lambda key, kind, call: call(),
        retrieval_mode="local-and-public-for-owner-gap/1.0.0",
    )
    assert len(value["materials"]) == 2
    assert any(b.get("status") == status for b in value["backlog"])


def test_gap_discovery_keeps_unknown_as_error():
    sources, _, _ = owner()
    sources.search_public = lambda **kw: pytest.fail("unknown never repeats")
    sources.fetch_hit = lambda _: pytest.fail("unknown never repeats")

    def unknown(*_):
        raise RuntimeError("definition_response_unknown_no_repeat")

    with pytest.raises(RuntimeError, match="unknown_no_repeat"):
        sources.search(
            query="q",
            key="gap",
            limit=2,
            operation=unknown,
            retrieval_mode="local-and-public-for-owner-gap/1.0.0",
        )


def test_new_trial_and_policy_cannot_be_mixed_with_old_version(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial as trial
    from tests.evaluation_preparation.test_component_trial import envelope

    value, spec = envelope(
        tmp_path, monkeypatch, "configs/research/multimodal_definition_gate_v15.json"
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.10.0"
    assert trial.read(tmp_path) == value
    value["schema_version"] = "multimodal-definition-trial/1.9.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        trial.checked(value, spec)


def test_owner_query_plan_selects_discovery_only_for_unbound_target(tmp_path):
    from types import SimpleNamespace

    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import fixture

    expansion = fixture(tmp_path)
    raw = json.loads(Path("configs/research/multimodal_definition_gate_v15.json").read_bytes())
    expansion.policy = read_expansion_policy(raw)
    expansion.original_rows = [
        dict(
            id="active:cup_geometry",
            input_modalities=["new_image"],
            definition=SimpleNamespace(interpretation_status="installed"),
        )
    ]
    plan = expansion.query_plan()[0]
    assert plan["target_relation_gap_ids"] == ["active:cup_geometry"]
    assert plan["retrieval_mode"] == "local-and-public-for-owner-gap/1.0.0"
    expansion.original_rows = []
    plan = expansion.query_plan()[0]
    assert plan["missing_definition_ids"] == ["active:cup_geometry"]
    assert plan["retrieval_mode"] is None


def test_new_policy_schema_and_config_frozen():
    from packages.multimodal_research.definition_expansion import GapDiscoveryExpansionPolicy
    from packages.multimodal_research.definition_trial import GapDiscoveryTrial
    from packages.multimodal_research.implementation import freeze

    assert "configs/research/multimodal_definition_gate_v15.json" in freeze()
    for model in (GapDiscoveryExpansionPolicy, GapDiscoveryTrial):
        path = Path("packages/llm_provider/schemas") / (model.__name__ + ".schema.json")
        assert json.loads(path.read_bytes()) == model.model_json_schema()


def test_production_sources_stage_consumes_owner_plan_and_cold_replays(tmp_path):
    from packages.multimodal_research.definition_expansion import DefinitionExpansion
    from tests.evaluation_preparation.test_definition_expansion import fixture
    from tests.evaluation_preparation.test_directional_review import fixture as review_fixture
    from tests.evaluation_preparation.test_intent_definition_retrieval import plan

    seed = fixture(tmp_path)
    definition = review_fixture().owner.current().policy.items[0].definition
    seed.campaign.base.ledger = [
        dict(
            id="active:cup_geometry",
            input_modalities=["new_image"],
            definition=definition,
        )
    ]
    raw = json.loads(Path("configs/research/multimodal_definition_gate_v15.json").read_bytes())
    raw.update(
        public_queries=["q"],
        query_requirements={"q": ["active:cup_geometry"]},
        public_topic_sets={"q": ["rim"]},
        passage_plans={"q": plan()},
    )
    seed.campaign.base.task.target_code = raw["public_target"]
    sources, _, _ = owner()
    calls = []
    sources.search_public = lambda **kw: calls.append(kw) or []
    sources.fetch_hit = lambda _: pytest.fail("no hits")

    def assembled():
        return DefinitionExpansion(
            seed.campaign,
            documents=seed.documents,
            sources=sources,
            provider=seed.provider,
            budget=seed.budget,
            scope=seed.scope,
            policy=raw,
        )

    expansion = assembled()
    result = expansion.advance(until="sources")
    assert len(calls) == 1
    assert result["stage"] == "sources" and result["materials"]
    trace = next(b for b in result["backlog"] if b["reason"] == "owner_gap_retrieval_plan")
    assert trace["plan"]["target_relation_gap_ids"] == ["active:cup_geometry"]
    assert trace["discovery"]["local_materials"] == 3
    assert assembled().advance(until="sources") == result
    assert len(calls) == 1 and seed.budget.used == 0
