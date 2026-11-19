"""Uniform public model binding; original per-dataset owners remain authoritative."""

import json
from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as batch
from packages.grounding.models import canonical, digest
from packages.llm_provider.config import read_provider_config
from packages.llm_provider.model_selection import apply_selection, load_selection
from packages.multimodal_research.campaign import configuration

SELECTION = "configs/agents/relay_existing_model_selection_v1.json"


def manifest():
    common = dict(root="/synthetic/input", task="advanced_glaucoma", review_batch_size=3)
    return dict(
        schema_version="terminal-case-cohort/1.6.0",
        id="uniform-model-test",
        purpose="development_coverage_not_accuracy",
        expected_model="gpt-6-sol",
        fairvlmed_model_selection=SELECTION,
        network_route="configs/agents/environment_https_route_v1.json",
        vision_feedback="state-consistency-once/1.0.0",
        command_timeout_seconds=10800,
        max_cases_per_invocation=1,
        source_recovery=dict(
            mode="technical_backlog_no_source_replay",
            window_seconds=5400,
            authorization_code="test-bounded-recovery",
        ),
        cases=[
            dict(
                common,
                id="gamma",
                dataset="gamma",
                sample="0001",
                active_acquisition="configs/research/multimodal_structural_coverage_v3.json",
            ),
            dict(
                common,
                id="harvard",
                dataset="harvard_gdp",
                sample="data_0001.npz",
                input_semantics="harvard-gdp-v1",
                selection="development_preparation",
                active_acquisition="configs/research/multimodal_harvard_visual_priority_v5.json",
            ),
            dict(
                common,
                id="grape",
                dataset="grape",
                sample="101:OS",
                cutoff_visit=0,
                input_semantics="grape-v1",
                selection="development_preparation",
                active_acquisition="configs/research/multimodal_grape_active_requirements_v6.json",
            ),
            dict(
                id="fair",
                dataset="fairvlmed",
                root="/synthetic/input",
                sample="data_00001.npz",
                task="glaucoma",
            ),
        ],
    )


def test_original_commands_select_only_fair_and_preserve_multimodal_feedback(tmp_path):
    m = batch.read_manifest(manifest())
    for case in m.cases:
        command = batch.commands(m, case, tmp_path, "prepare")[0]
        assert "--model-selection" not in command
        if case.dataset == "fairvlmed":
            assert command[3] == "apps.fairvlmed_case"
            assert command[command.index("--provider-profile") + 1] == (
                "relay-grounded-shape-proxy-alternate-v1"
            )
        else:
            assert command[3] == "apps.multimodal_case"
            assert command[command.index("--vision-feedback") + 1] == m.vision_feedback
            successor = batch.commands(m, case, tmp_path, "source-prepare")[0]
            assert "--model-selection" not in successor
            assert successor[successor.index("--vision-feedback") + 1] == m.vision_feedback
            assert "--defer-failed-definition-expansion" in successor


def test_actual_owner_configs_same_model_not_same_prompts_or_replaced_limits():
    from packages.evaluation_preparation.generic_input import ROUTED_PROVIDER
    from packages.llm_provider.network_route import load_route

    m = batch.read_manifest(manifest())
    bindings = batch.provider_bindings(m)
    before = Path(ROUTED_PROVIDER).read_bytes()
    old = read_provider_config(json.loads(before))
    selected = apply_selection(old, load_selection(SELECTION))
    assert old.transport.model == "gpt-4o-2024-08-06"
    assert bindings["fair"]["provider_checksum"] == digest(selected)
    assert selected.roles == old.roles
    assert selected.max_requests == old.max_requests == 40
    assert selected.transport.max_retries == old.transport.max_retries
    for case in m.cases[:-1]:
        actual = configuration(
            resilient_vision=True,
            directional=True,
            exact_primary_status=True,
            qualified_observations=bool(case.input_semantics),
            execution_profile="relay-root-question-v1",
            review_batch_size=3,
            network_route=load_route(m.network_route),
            vision_feedback=m.vision_feedback,
        )
        assert bindings[case.id]["provider_checksum"] == digest(actual)
        assert actual.max_requests == 40
    assert {b["model"] for b in bindings.values()} == {"gpt-6-sol"}
    assert Path(ROUTED_PROVIDER).read_bytes() == before


def test_freeze_cold_read_and_binding_drift_rejected(tmp_path, monkeypatch):
    # No credentials, socket/API or database. The original public config owners run.
    monkeypatch.setattr("packages.llm_provider.network_route.route_opener", lambda *a: None)
    monkeypatch.setattr(batch, "implementation", lambda: {"test-freeze": "unchanged"})
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical(manifest()))
    out = tmp_path / "cohort"
    assert batch.initialize(path, out)["api_requests"] == 0
    _, m, frozen = batch.load(out, check_current=True)
    assert frozen["provider_bindings"] == batch.provider_bindings(m)
    before = (out / "freeze.json").read_bytes()
    actual = batch.provider_bindings
    monkeypatch.setattr(batch, "provider_bindings", lambda m: {**actual(m), "extra": {}})
    with pytest.raises(ValueError, match="cohort_provider_binding_changed"):
        batch.load(out, check_current=True)
    # Historical inspection is read-only and does not reauthorize execution.
    assert batch.load(out, check_current=False)[2] == frozen
    assert (out / "freeze.json").read_bytes() == before


def test_wrong_expected_model_refuses_before_partial_creation(tmp_path, monkeypatch):
    monkeypatch.setattr("packages.llm_provider.network_route.route_opener", lambda *a: None)
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical(dict(manifest(), expected_model="different-model")))
    out = tmp_path / "cohort"
    with pytest.raises(ValueError, match="cohort_actual_provider_model_mismatch"):
        batch.initialize(path, out)
    assert not out.exists()


@pytest.mark.parametrize("version", ["1.4.0", "1.5.0"])
def test_historical_command_versions_are_not_silently_upgraded(tmp_path, version):
    raw = manifest()
    raw["schema_version"] = "terminal-case-cohort/" + version
    del raw["expected_model"]
    del raw["fairvlmed_model_selection"]
    if version == "1.4.0":
        del raw["vision_feedback"]
        raw["model_selection"] = SELECTION
    m = batch.read_manifest(raw)
    fair = batch.commands(m, m.cases[-1], tmp_path, "prepare")[0]
    expected = "relay-grounded-shape-proxy" + ("-alternate" if version == "1.4.0" else "") + "-v1"
    assert fair[fair.index("--provider-profile") + 1] == expected
    multi = batch.commands(m, m.cases[0], tmp_path, "prepare")[0]
    assert ("--model-selection" in multi) == (version == "1.4.0")
    assert ("--vision-feedback" in multi) == (version == "1.5.0")


def test_version_does_not_accept_arbitrary_selector_or_hide_new_fields():
    raw = manifest()
    with pytest.raises(ValueError):
        batch.read_manifest(dict(raw, fairvlmed_model_selection=".local/private.json"))
    with pytest.raises(ValueError):
        batch.read_manifest(dict(raw, schema_version="terminal-case-cohort/1.5.0"))
