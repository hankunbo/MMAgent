"""Explicit proven public configurations reach original fresh case entry owners."""

import json
from pathlib import Path

import pytest

from packages.evaluation_preparation import generic_input as fair
from packages.evaluation_preparation import terminal_cohort as terminal
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config

PROFILE = "relay-local-gpt6-shape-v1"
PROVIDER = "configs/agents/fairvlmed_relay_local_gpt6_review_v1.json"
ROUTE = "configs/agents/server_local_https_route_v1.json"


def public(path):
    return read_provider_config(json.loads(Path(path).read_bytes()))


def manifest():
    from tests.evaluation_preparation.test_terminal_runtime_lease import manifest as original

    raw = original().model_dump(mode="json")
    raw.pop("fairvlmed_model_selection", None)
    raw.update(
        schema_version="terminal-case-cohort/1.11.0",
        fairvlmed_execution_lock="runtime-case-host-resource/1.0.0",
        multimodal_execution_lock="multimodal-runtime-entry/1.0.0",
        fairvlmed_provider_profile=PROFILE,
        review_contract="rating-presence/1.1.0",
        network_route=ROUTE,
    )
    return terminal.read_manifest(raw)


def local_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)


def prepared(tmp_path, monkeypatch):
    from tests.evaluation_preparation.test_generic_input import data

    data(tmp_path / "data", "synthetic-new.npz")
    monkeypatch.setattr(fair, "directory", lambda key: tmp_path / key)
    local_environment(monkeypatch)
    return fair.prepare(
        dataset="fairvlmed",
        root=tmp_path / "data",
        sample="synthetic-new.npz",
        key="local-case",
        evidence_profile="source-complete-view/1.0.0",
        upstream_profile="evidence-reuse/1.0.0",
        report_profile="case-report-production/1.0.0",
        provider_profile=PROFILE,
        execution_lock_profile="runtime-case-host-resource/1.0.0",
    )


def test_fresh_fair_prepare_load_provider_lease_and_owner_identity(tmp_path, monkeypatch):
    from apps import segman_resource_lock
    from packages.evaluation_preparation import generic_execution_lock
    from packages.evaluation_preparation.generic_campaign import GenericCampaign

    selected = prepared(tmp_path, monkeypatch)
    assert selected["schema_version"] == "fairvlmed-run-input/1.10.0"
    assert "model_selection" not in selected
    assert fair.load(selected["key"]) == selected
    assert fair.provider_config(selected) == public(PROVIDER)
    assert generic_execution_lock.validate_profile(selected)
    binding = dict(
        schema_version="generic-case-binding/1.2.0",
        execution_namespace=selected["key"],
        input_spec_checksum=digest(selected),
    )
    assert (
        segman_resource_lock.verify_input(
            generic_execution_lock.ROOT, tmp_path / selected["key"], binding
        )
        == selected
    )
    campaign = object.__new__(GenericCampaign)
    campaign.spec = selected
    with pytest.raises(ValueError, match="original_execution_lease_required"):
        campaign.execution_session()
    assert fair.owner_key(selected, "phase") != fair.owner_key(
        dict(selected, key="another"), "phase"
    )
    assert {p.name for p in (tmp_path / selected["key"]).iterdir()} == {"input.json"}


@pytest.mark.parametrize(
    "fault", ["model_selection", "old_version", "path", "checksum", "profile", "lock"]
)
def test_new_selection_cannot_reinterpret_old_or_changed_input(tmp_path, monkeypatch, fault):
    selected = prepared(tmp_path, monkeypatch)
    if fault == "model_selection":
        selected["model_selection"] = {}
    elif fault == "old_version":
        selected["schema_version"] = "fairvlmed-run-input/1.9.0"
    elif fault == "path":
        selected["provider_configuration"] = ".local/vectorengine/config.json"
    elif fault == "checksum":
        selected["provider_configuration_checksum"] = digest("changed")
    elif fault == "profile":
        selected["provider_profile"] = fair.ALTERNATE_PROFILE
    else:
        selected.pop("execution_lock_profile")
    (tmp_path / selected["key"] / "input.json").write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="generic_"):
        fair.load(selected["key"])


def test_terminal_actual_configs_equal_both_original_public_gates(tmp_path):
    selected = manifest()
    bindings = terminal.provider_bindings(selected)
    for case in selected.cases:
        expected = public(
            PROVIDER
            if case.dataset == "fairvlmed"
            else "configs/agents/multimodal_relay_local_gpt6_review_v4.json"
        )
        assert bindings[case.id]["provider_checksum"] == digest(expected)
        argv = terminal.commands(selected, case, tmp_path, "prepare")[0]
        assert "--model-selection" not in argv
        if case.dataset == "fairvlmed":
            assert argv[argv.index("--provider-profile") + 1] == PROFILE
            assert "--execution-lock-profile" in argv
        else:
            assert argv[argv.index("--review-contract") + 1] == "rating-presence/1.1.0"
            assert argv[argv.index("--network-route") + 1] == ROUTE
            assert "--entry-execution-lock" in argv


def test_initialize_freeze_and_cold_load_no_network_or_model_selector(tmp_path, monkeypatch):
    local_environment(monkeypatch)
    selected = manifest()
    path = tmp_path / "manifest.json"
    path.write_text(selected.model_dump_json())
    out = tmp_path / "cohort"
    assert terminal.initialize(path, out)["api_requests"] == 0
    _, restored, freeze = terminal.load(out, check_current=True)
    assert restored == selected
    assert freeze["provider_bindings"] == terminal.provider_bindings(selected)
    assert PROVIDER in freeze["configurations"]
    assert not any("model_selection" in p for p in freeze["configurations"])
    assert freeze["runtime_owner"] == terminal.runtime_owner_implementation()


def test_old_manifest_does_not_gain_local_route_or_contract(tmp_path):
    from tests.evaluation_preparation.test_terminal_runtime_lease import manifest as original

    old = original()
    for case in old.cases:
        argv = terminal.commands(old, case, tmp_path, "prepare")[0]
        assert PROFILE not in argv and "--review-contract" not in argv
    raw = manifest().model_dump(mode="json")
    for wrong in (
        dict(raw, schema_version="terminal-case-cohort/1.10.0"),
        dict(
            raw, fairvlmed_model_selection="configs/agents/relay_existing_model_selection_v1.json"
        ),
        dict(raw, network_route="configs/agents/environment_https_route_v1.json"),
    ):
        with pytest.raises(ValueError):
            terminal.read_manifest(wrong)
