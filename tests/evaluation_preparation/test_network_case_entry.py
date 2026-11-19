"""Network selection is explicit at original terminal entries, not a hidden retry."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import terminal_cohort
from packages.grounding.models import digest
from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research import entry
from tests.dataset_adapters.test_multimodal import gdp
from tests.evaluation_preparation.test_generic_provider_profile import prepared
from tests.evaluation_preparation.test_terminal_cohort import manifest

ROUTE = "configs/agents/environment_https_route_v1.json"


def test_actual_multimodal_prepare_preflight_and_route_binding(tmp_path, monkeypatch):
    root = gdp(tmp_path / "input")
    out = tmp_path / "entry"
    policy = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8888")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("packages.llm_provider.network_route.load_route", lambda path: policy)
    entry.prepare_entry(
        dataset="harvard_gdp",
        root=root,
        sample="data_0001.npz",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_harvard_visual_priority_v2.json",
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="harvard-gdp-v1",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route=ROUTE,
    )
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("preflight reached database"))
    monkeypatch.setattr(
        "packages.multimodal_research.entry_preflight.check_local_route", lambda *a: "offline"
    )
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    spec_path = out / "entry.json"
    spec = json.loads(spec_path.read_bytes())
    assert spec["schema_version"] == "multimodal-live-entry/1.29.0"
    assert spec["provider"]["schema_version"] == "agent-provider-config/9.1.0"
    assert spec["network_route"] == spec["provider"]["network_route"]
    assert spec["max_requests"] == 40 and spec["seconds"] == 5400
    assert spec["root_direction_review"]["title"] == "晚期青光眼"
    spec_path.write_text(json.dumps({**spec, "schema_version": "multimodal-live-entry/1.26.0"}))
    with pytest.raises(ValueError, match="network_route_entry_version"):
        entry.assemble_entry(out, preflight_only=True)
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8889")
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        entry.assemble_entry(out, preflight_only=True)


def test_route_cannot_enable_old_failed_entry_replay(tmp_path):
    with pytest.raises(ValueError, match="network_route_requires_fresh_entry_not_replay"):
        entry.prepare_entry(
            dataset="gamma",
            root="/not-read",
            sample="0001",
            task="advanced_glaucoma",
            output=tmp_path,
            execution_profile="relay-root-question-v1",
            review_batch_size=3,
            network_route=ROUTE,
            definition_failure_of="old-failure",
        )


def test_fair_input_new_route_uses_same_original_prompt_without_mutating_old(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_input as g

    routed_bytes = Path(g.ROUTED_PROVIDER).read_bytes()
    shape_bytes = Path(g.SHAPE_PROVIDER).read_bytes()
    old, options = prepared(tmp_path, monkeypatch)
    before = (g.directory(old["key"]) / "input.json").read_bytes()
    routed = json.loads(routed_bytes)
    policy = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    routed["network_route"] = policy.model_dump(mode="json")
    Path(g.ROUTED_PROVIDER).write_text(json.dumps(routed))
    Path(g.SHAPE_PROVIDER).write_bytes(shape_bytes)
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8888")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    new = g.prepare(**options, key="explicit-network", provider_profile=g.ROUTED_PROFILE)
    assert new["schema_version"] == "fairvlmed-run-input/1.7.0"
    assert g.load(new["key"]) == new
    config = g.provider_config(new)
    assert config.schema_version == "agent-provider-config/7.1.0"
    assert digest(config) == new["provider_configuration_checksum"]
    assert new["max_requests"] == 40 and new["seconds"] == 5400
    assert g.owner_key(new, "phase", "review") != g.owner_key(old, "phase", "review")
    assert (g.directory(old["key"]) / "input.json").read_bytes() == before
    assert not list(tmp_path.rglob("binding.json"))
    from packages.evaluation_preparation import generic_run

    monkeypatch.setattr(generic_run, "start", lambda *a: pytest.fail("started despite route drift"))
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8889")
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        generic_run.run(new["key"])


def test_routed_batch_cli_is_explicit_and_legacy_manifest_bytes_unchanged(tmp_path):
    old = terminal_cohort.read_manifest(manifest())
    old_bytes = old.model_dump_json()
    raw = manifest()
    raw.update(schema_version="terminal-case-cohort/1.2.0", network_route=ROUTE)
    new = terminal_cohort.read_manifest(raw)
    argv = terminal_cohort.commands(new, new.cases[0], tmp_path, "prepare")[0]
    assert argv[argv.index("--network-route") + 1] == ROUTE
    assert (
        "--network-route" not in terminal_cohort.commands(old, old.cases[0], tmp_path, "prepare")[0]
    )
    assert old.model_dump_json() == old_bytes


def test_new_root_report_recovery_preserves_question_without_network():
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.report_recovery import restore_execution_context
    from packages.multimodal_research.single_proposition import root_direction_spec

    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"}).model_dump(mode="json")
    spec = dict(
        schema_version="multimodal-live-entry/1.29.0",
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        root_direction_review=root_direction_spec(task),
        network_route=route,
        provider=dict(network_route=route),
    )
    protocol = dict(spec, schema_version="multimodal-live-campaign/1.16.0")
    base = SimpleNamespace(task=task)
    restore_execution_context(base, spec, protocol)
    assert (
        base.review_batch_size == 3 and base.root_direction_review == spec["root_direction_review"]
    )
    assert base.network_route == route
    with pytest.raises(ValueError, match="saved_report_network_route_binding"):
        restore_execution_context(base, {**spec, "provider": {}}, protocol)


@pytest.mark.parametrize("routed", [False, True])
def test_original_campaign_constructor_preserves_prepared_provider(tmp_path, monkeypatch, routed):
    """Exercise the actual constructor that originally lost the network selection."""
    from packages.multimodal_research.campaign import MultimodalCampaign, configuration

    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    base = SimpleNamespace(
        output=tmp_path,
        manifest=SimpleNamespace(sample="synthetic", adapter_version="multimodal-adapters/1.2.0"),
        inputs=object(),
        definition_expansion_policy={},
        resilient_vision=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )
    if routed:
        base.network_route = route.model_dump(mode="json")
    monkeypatch.setattr(
        "db.runtime.repository.postgres.validate_runtime_principal", lambda target: None
    )
    # No task exists yet; only the target enumeration has no input in this test.
    monkeypatch.setattr(MultimodalCampaign, "available_targets", staticmethod(lambda base: ()))
    campaign = MultimodalCampaign(base, object(), campaign_id="synthetic-constructor")
    expected = configuration(
        directional=True,
        resilient_vision=True,
        exact_primary_status=True,
        qualified_observations=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        network_route=route if routed else None,
    )
    assert campaign.config == expected
    assert campaign.config.schema_version == (
        "agent-provider-config/9.1.0" if routed else "agent-provider-config/9.0.0"
    )
