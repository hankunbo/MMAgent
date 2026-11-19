"""Explicit bounded component consumption plus definition increment; offline."""

import json
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.multimodal_research import definition_trial as trial
from tests.evaluation_preparation.test_component_consumer import config


def envelope(tmp_path, monkeypatch, policy="configs/research/multimodal_definition_gate_v11.json"):
    monkeypatch.setattr(trial, "freeze", lambda: {"frozen": "same"})
    task, _ = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")
    path, selection = config(tmp_path, task=task)
    spec = dict(
        schema_version="multimodal-live-entry/1.14.0",
        implementation=trial.freeze(),
        intent=task.model_dump(mode="json"),
        input_checksum=selection.input_checksum,
        component_consumer=dict(path=str(path), checksum=digest(selection)),
    )
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    value = trial.prepare(
        tmp_path,
        expansion_policy=policy,
        authorization="synthetic explicit component review window",
        max_new_requests=8,
        seconds=1800,
    )
    return value, spec


def test_preparation_selects_new_version_without_registration(tmp_path, monkeypatch):
    value, _ = envelope(tmp_path, monkeypatch)
    assert value["schema_version"] == "multimodal-definition-trial/1.7.0"
    assert value["review_scope"] == "definition_increment_and_confirmed_component"
    assert trial.read(tmp_path) == value
    assert not (tmp_path / "protocol.json").exists()


@pytest.mark.parametrize("change", ["entry", "old_trial", "task", "input", "selection", "policy"])
def test_component_trial_rejects_cross_identity_or_old_qualification(tmp_path, monkeypatch, change):
    value, spec = envelope(tmp_path, monkeypatch)
    if change == "entry":
        spec["schema_version"] = "multimodal-live-entry/1.13.0"
    elif change == "old_trial":
        value["schema_version"] = "multimodal-definition-trial/1.6.0"
        value["review_scope"] = "definition_increment_only"
    elif change == "task":
        spec["intent"]["target_code"] = "GOAL:OTHER"
    elif change == "input":
        spec["input_checksum"] = digest("other")
    elif change == "selection":
        spec["component_consumer"]["checksum"] = digest("other")
    else:
        value["expansion_policy"]["schema_version"] = "campaign-definition-expansion/1.11.0"
    value["entry_checksum"] = digest(spec)
    with pytest.raises(ValueError):
        trial.checked(value, spec)


@pytest.mark.parametrize("allowed", [True, False])
def test_partial_review_selection_never_redispatches_component(allowed):
    calls = []
    c = object.__new__(trial.DefinitionTrialCampaign)
    c.base = SimpleNamespace(
        component_observations=SimpleNamespace(
            target="component", check=lambda: calls.append("current")
        ),
        observations={"component": "confirmed"},
        ledger=[dict(id="component", review_allowed=allowed)],
    )
    c.targets = ("component", "added")
    selected = c.select_component_review(("added",), {})
    assert selected == (("added", "component") if allowed else ("added",))
    assert c.targets == ("added",)
    assert c.base.review_target_ids == selected
    assert calls == ["current"]


def test_new_trial_schema_exact():
    from pathlib import Path

    assert (
        json.loads(Path("packages/llm_provider/schemas/ComponentTrial.schema.json").read_bytes())
        == trial.ComponentTrial.model_json_schema()
    )


def test_only_new_trial_increases_review_capacity(monkeypatch):
    from packages.multimodal_research.campaign import MultimodalCampaign

    monkeypatch.setattr(
        MultimodalCampaign, "__init__", lambda self: setattr(self, "base", SimpleNamespace())
    )
    for version, expected in [("1.6.0", 2048), ("1.7.0", 4096), ("1.8.0", 4096)]:
        campaign = trial.DefinitionTrialCampaign(
            trial=dict(schema_version="multimodal-definition-trial/" + version, expansion_policy={})
        )
        config = campaign.config.model_dump(mode="json")
        assert (
            next(r for r in config["roles"] if r["role"] == "review")["max_output_tokens"]
            == expected
        )
        assert config["review_batching"]["batch_size"] == 2


def test_intent_component_trial_explicit_version_and_schema(tmp_path, monkeypatch):
    from pathlib import Path

    value, spec = envelope(
        tmp_path, monkeypatch, "configs/research/multimodal_definition_gate_v13.json"
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.8.0"
    assert trial.read(tmp_path) == value
    assert (
        json.loads(
            Path("packages/llm_provider/schemas/IntentComponentTrial.schema.json").read_bytes()
        )
        == trial.IntentComponentTrial.model_json_schema()
    )
    value["schema_version"] = "multimodal-definition-trial/1.7.0"
    with pytest.raises(ValueError):
        trial.checked(value, spec)


@pytest.mark.parametrize("version", ["1.7.0", "1.8.0"])
def test_component_protocol_version_freezes_and_cold_reads(tmp_path, monkeypatch, version):
    from packages.multimodal_research import implementation
    from packages.multimodal_research.campaign import MultimodalCampaign

    monkeypatch.setattr(implementation, "freeze", lambda: {})
    c = object.__new__(MultimodalCampaign)
    c.output, c.campaign_id = tmp_path, "synthetic-component-version"
    checksum = SimpleNamespace(checksum=digest("synthetic"))
    c.base = SimpleNamespace(
        manifest=checksum,
        visible_value=checksum,
        task=checksum,
        view_checksum=digest("views"),
        sources=SimpleNamespace(initial=[]),
    )
    c.config, c.targets = {}, ("component",)
    c.previous_attempts = lambda: []
    c.trial = dict(schema_version="multimodal-definition-trial/" + version, max_new_requests=12)
    actual = c.freeze()
    assert actual["schema_version"] == "multimodal-live-campaign/1.6.0"
    assert c.freeze() == actual


def test_component_campaign_stage_dispatch_and_partial_memory(tmp_path, monkeypatch):
    from packages.multimodal_research import memory, stages

    (tmp_path / "protocol.json").write_text(
        json.dumps({"schema_version": "multimodal-live-campaign/1.6.0"})
    )
    sentinel = object()
    monkeypatch.setattr(trial, "cold", lambda output: sentinel)
    assert stages.assemble_for_stage(tmp_path) is sentinel
    campaign = SimpleNamespace(protocol={"definition_increment_only": True})
    assert memory.archive(campaign, {}) == dict(
        status="not_archived_whole_case_increment_only", eligible_for_planning=False
    )


def test_public_topic_selection_and_configuration_freeze():
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from packages.multimodal_research.implementation import freeze
    from packages.object_store import checksum

    path = Path("configs/research/multimodal_definition_gate_v12.json")
    policy = read_expansion_policy(json.loads(path.read_bytes()))
    assert policy.max_definitions == policy.max_new_observations == policy.max_rounds == 1
    assert policy.max_search_requests == policy.max_fetch_requests == 2
    assert policy.reserve_review_requests == 3
    assert freeze()[str(path)] == checksum(path.read_bytes())


@pytest.mark.parametrize("mode", ["distinct", "same", "conflict"])
def test_confirmed_component_and_stage_facts_are_not_lost_or_duplicated(mode):
    from packages.multimodal_research.report import acquired_observations

    baseline = {"component": {"status": "obtained", "value": 0.5}}
    stage = (
        {"new": {"status": "obtained", "value": 0.2}}
        if mode == "distinct"
        else {"component": {"status": "obtained", "value": 0.5 if mode == "same" else 0.6}}
    )
    campaign = SimpleNamespace(
        base=SimpleNamespace(observations=baseline), accepted_observations=lambda: stage
    )
    if mode == "conflict":
        with pytest.raises(ValueError, match="acquired_observation_conflict"):
            acquired_observations(campaign)
    else:
        actual = acquired_observations(campaign)
        assert len(actual) == (2 if mode == "distinct" else 1)
        assert actual["component"] == baseline["component"]


def test_partial_scope_survives_saved_result_html_without_score_change():
    from packages.evaluation_preparation.evidence_tree_v3 import render
    from packages.multimodal_research.campaign import review_observations
    from tests.evaluation_preparation.test_unified_view import report

    value = report()
    original = digest(value["scores"])
    value["review_scope"] = dict(
        kind="definition_increment_and_confirmed_component",
        explanation="限定组件与增量，不是整例评分 <unsafe>",
    )
    page = render(value, input_checksum=digest(value))
    assert "限定组件与增量，不是整例评分 &lt;unsafe&gt;" in page
    assert digest(value["scores"]) == original
    assert review_observations(SimpleNamespace(review_target_ids=("a",)), {"a": 1, "b": 2}) == {
        "a": 1
    }
