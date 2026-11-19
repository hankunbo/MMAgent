"""Model selection changes identity, not methods or historical UNKNOWN requests."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config
from packages.llm_provider.model_selection import apply_selection, load_selection
from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research.campaign import configuration

SELECTION = "configs/agents/relay_existing_model_selection_v1.json"


@pytest.fixture(autouse=True)
def legacy_model_origin(monkeypatch):
    """Exercise the original 4o -> 6-sol contract, not today's mutable default model."""
    from packages.multimodal_research import context_review

    original_configuration = context_review.configuration

    def legacy(origin, *, profile=context_review.PROFILE):
        selected = original_configuration(origin, profile=profile)
        if profile == "configs/agents/relayrouter_snapshot_evidence_v1.json":
            raw = selected.model_dump(mode="json")
            raw["transport"]["model"] = "gpt-4o-2024-08-06"
            return read_provider_config(raw)
        return selected

    monkeypatch.setattr(context_review, "configuration", legacy)


def original():
    return configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route=prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"}),
    )


def test_model_only_delta_and_original_bytes():
    before = original()
    raw = before.model_dump(mode="json")
    selected = apply_selection(before, load_selection(SELECTION))
    expected = copy.deepcopy(raw)
    expected["transport"]["model"] = "gpt-6-sol"
    assert selected.model_dump(mode="json") == expected
    assert before.model_dump(mode="json") == raw
    assert selected.roles == before.roles
    assert selected.transport.max_retries == before.transport.max_retries
    assert digest(before) != digest(selected)


@pytest.mark.parametrize("change", ["model", "base_url", "api_key_env", "route", "binding"])
def test_unapproved_changes_rejected(change):
    config, binding = original(), load_selection(SELECTION)
    raw = config.model_dump(mode="json")
    if change == "binding":
        binding["profile_checksum"] = digest("changed")
    elif change == "route":
        raw.pop("network_route")
        raw["schema_version"] = "agent-provider-config/9.0.0"
    else:
        raw["transport"][change] = (
            "https://invalid.example/v1"
            if change == "base_url"
            else "DIFFERENT_ENV"
            if change == "api_key_env"
            else "different"
        )
    with pytest.raises(ValueError, match="model_selection"):
        apply_selection(read_provider_config(raw), binding)


def test_private_path_refused_before_read(tmp_path):
    with pytest.raises(ValueError, match="public_configuration"):
        load_selection(tmp_path / "private.json")


def test_original_prepare_preflight_cold_restore_and_successor(tmp_path, monkeypatch):
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research import entry
    from packages.multimodal_research.discovery_continuation import definition_parent
    from packages.multimodal_research.report_recovery import restore_execution_context
    from tests.dataset_adapters.test_multimodal import gdp
    from tests.evaluation_preparation.test_routed_source_backlog import routed_parent

    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8888")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("packages.llm_provider.network_route.load_route", lambda path: route)
    monkeypatch.setattr(
        "packages.multimodal_research.entry_preflight.check_local_route", lambda *a: "offline"
    )
    common = dict(
        dataset="harvard_gdp",
        root=gdp(tmp_path / "data"),
        sample="data_0001.npz",
        task="advanced_glaucoma",
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_harvard_visual_priority_v5.json",
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="harvard-gdp-v1",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route="configs/agents/environment_https_route_v1.json",
        model_selection=SELECTION,
    )
    out = tmp_path / "fresh"
    result = entry.prepare_entry(**common, output=out)
    spec = json.loads((out / "entry.json").read_bytes())
    assert result["api_requests"] == result["database_writes"] == 0
    assert spec["provider"]["transport"]["model"] == "gpt-6-sol"
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    frozen = dict(
        schema_version="multimodal-live-campaign/1.16.0",
        execution_profile=spec["execution_profile"],
        root_direction_review=spec["root_direction_review"],
        review_batch_size=3,
        model_selection=spec["model_selection"],
    )
    base = SimpleNamespace(task=load_intent(common["task"], common["registry"])[0])
    restore_execution_context(base, spec, frozen)
    assert base.model_selection == spec["model_selection"]
    assert base.network_route == spec["network_route"]
    with pytest.raises(ValueError, match="model_selection_protocol"):
        restore_execution_context(base, spec, dict(frozen, model_selection=None))

    # Actual Source-failure parent owner validates the same model extension.
    _, ref, _, plan, failure, _, _ = routed_parent(tmp_path)
    plan.update(
        input=spec["input_checksum"],
        implementation=spec["implementation"],
        provider=digest(spec["provider"]),
        model_selection=spec["model_selection"],
    )
    failure.update(input_checksum=spec["input_checksum"], protocol_checksum=digest(plan))
    parent = Path(ref["parent"])
    for name, value in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (parent / (name + ".json")).write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in parent.glob("*.json")}
    child = tmp_path / "model-successor"
    entry.prepare_entry(
        **common,
        output=child,
        definition_failure_of=parent,
        defer_failed_definition_expansion=True,
        source_fallback_seconds=1800,
        source_fallback_authorization="test-no-source-replay",
    )
    successor = json.loads((child / "entry.json").read_bytes())
    assert successor["provider"] == spec["provider"]
    assert successor["model_selection"] == spec["model_selection"]
    assert entry.assemble_entry(child, preflight_only=True)["status"] == "local_preflight_passed"
    assert {p.name: p.read_bytes() for p in parent.glob("*.json")} == before
    # A changed binding cannot use the parent qualification.
    plan.pop("model_selection")
    (parent / "protocol.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="model_selection_protocol"):
        definition_parent(
            parent,
            checksum=spec["input_checksum"],
            intent=spec["intent"],
            defer=True,
            root_backlog=True,
            routed_backlog=True,
        )


def test_fair_profile_current_config_and_old_defaults():
    from packages.evaluation_preparation.generic_input import (
        ALTERNATE_PROFILE,
        ROUTED_PROFILE,
        ROUTED_PROVIDER,
        owner_key,
        provider_config,
    )

    old = read_provider_config(json.loads(Path(ROUTED_PROVIDER).read_bytes()))
    base = dict(
        schema_version="fairvlmed-run-input/1.7.0",
        provider_profile=ROUTED_PROFILE,
        provider_configuration=ROUTED_PROVIDER,
        provider_configuration_checksum=digest(old),
        owner_key_policy="content-bound-owner-key/1.0.0",
    )
    assert provider_config(base) == old
    binding = load_selection(SELECTION)
    new = dict(
        base,
        schema_version="fairvlmed-run-input/1.8.0",
        provider_profile=ALTERNATE_PROFILE,
        model_selection=binding,
        provider_configuration_checksum=digest(apply_selection(old, binding)),
    )
    assert provider_config(new) == apply_selection(old, binding)
    assert owner_key(new, "phase") != owner_key(base, "phase")
    with pytest.raises(ValueError, match="generic_model_selection_version"):
        provider_config(dict(base, model_selection=binding))
    with pytest.raises(ValueError, match="generic_model_selection_version"):
        provider_config({k: v for k, v in new.items() if k != "model_selection"})


def test_terminal_selection_preserved_in_original_commands(tmp_path):
    from packages.evaluation_preparation import terminal_cohort as batch
    from tests.evaluation_preparation.test_terminal_routed_recovery import routed_manifest

    raw = routed_manifest().model_dump(mode="json")
    raw.update(schema_version="terminal-case-cohort/1.4.0", model_selection=SELECTION)
    manifest = batch.read_manifest(raw)
    for case in manifest.cases:
        argv = batch.commands(manifest, case, tmp_path, "prepare")[0]
        if case.dataset == "fairvlmed":
            assert argv[argv.index("--provider-profile") + 1].endswith("-alternate-v1")
        else:
            assert argv[argv.index("--model-selection") + 1] == SELECTION
            child = batch.commands(manifest, case, tmp_path, "source-prepare")[0]
            assert child[child.index("--model-selection") + 1] == SELECTION
            assert "--defer-failed-definition-expansion" in child
    # An old manifest cannot silently acquire the model selection.
    with pytest.raises(ValueError):
        batch.read_manifest(dict(raw, schema_version="terminal-case-cohort/1.3.0"))


def test_actual_campaign_constructor_and_protocol_keep_selected_model(tmp_path, monkeypatch):
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.single_proposition import root_direction_spec

    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    binding = load_selection(SELECTION)
    identity = SimpleNamespace(
        sample="synthetic", adapter_version="multimodal-adapters/1.2.0", checksum=digest("input")
    )
    base = SimpleNamespace(
        output=tmp_path,
        manifest=identity,
        visible_value=identity,
        task=task,
        inputs=object(),
        definition_expansion_policy={},
        resilient_vision=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route=original().network_route.model_dump(mode="json"),
        model_selection=binding,
        root_direction_review=root_direction_spec(task),
        sources=SimpleNamespace(initial=[]),
        view_checksum=digest("views"),
    )
    monkeypatch.setattr(
        "db.runtime.repository.postgres.validate_runtime_principal", lambda target: None
    )
    monkeypatch.setattr(MultimodalCampaign, "available_targets", staticmethod(lambda base: ()))
    campaign = MultimodalCampaign(base, object(), campaign_id="model-selection-test")
    expected = configuration(
        directional=True,
        resilient_vision=True,
        exact_primary_status=True,
        qualified_observations=True,
        execution_profile=base.execution_profile,
        review_batch_size=3,
        network_route=base.network_route,
        model_selection=binding,
    )
    assert campaign.config == expected
    frozen = campaign.freeze()
    assert frozen["model_selection"] == binding
    assert frozen["provider"] == digest(expected)
    assert campaign.freeze() == frozen


@pytest.mark.parametrize("lock_profile", [None, "runtime-case-host-resource/1.0.0"])
def test_fair_prepare_load_and_original_config_are_consistent(tmp_path, monkeypatch, lock_profile):
    from packages.evaluation_preparation import generic_input as g
    from tests.evaluation_preparation.test_generic_input import data

    data(tmp_path / "data", "new_input.npz")
    monkeypatch.setattr(g, "directory", lambda key: tmp_path / key)
    # No network operation: this only verifies the installed public route.
    selected = g.prepare(
        dataset="fairvlmed",
        root=tmp_path / "data",
        sample="new_input.npz",
        key="selected-model",
        evidence_profile="source-complete-view/1.0.0",
        upstream_profile="evidence-reuse/1.0.0",
        report_profile="case-report-production/1.0.0",
        provider_profile=g.ALTERNATE_PROFILE,
        execution_lock_profile=lock_profile,
    )
    assert selected["schema_version"] == (
        "fairvlmed-run-input/1.9.0" if lock_profile else "fairvlmed-run-input/1.8.0"
    )
    assert g.load(selected["key"]) == selected
    config = g.provider_config(selected)
    assert config.transport.model == "gpt-6-sol"
    assert (
        config.roles == read_provider_config(json.loads(Path(g.ROUTED_PROVIDER).read_bytes())).roles
    )
    assert selected["max_requests"] == 40 and selected["seconds"] == 5400
    assert not list(tmp_path.rglob("binding.json"))
