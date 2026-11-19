"""New Fair route binds the original owners without reinterpreting old runs."""

import json
from pathlib import Path

import pytest

from packages.evaluation_preparation import generic_input as fair
from packages.evaluation_preparation import terminal_cohort as terminal
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config

PROFILE = "relay-environment-gpt6-shape-v1"
PROVIDER = "configs/agents/fairvlmed_relay_environment_gpt6_review_v1.json"
VERSION = "fairvlmed-run-input/1.11.0"
ROUTE = "configs/agents/environment_https_route_v1.json"


def prepared(tmp_path, monkeypatch):
    from packages.llm_provider import network_route
    from tests.evaluation_preparation.test_generic_input import data

    data(tmp_path / "data", "new-environment.npz")
    monkeypatch.setattr(fair, "directory", lambda key: tmp_path / key)
    # No HTTP dispatch in input preparation; routing is probed separately.
    monkeypatch.setattr(network_route, "route_opener", lambda *a: None)
    return fair.prepare(
        dataset="fairvlmed",
        root=tmp_path / "data",
        sample="new-environment.npz",
        key="environment-case",
        evidence_profile="source-complete-view/1.0.0",
        upstream_profile="evidence-reuse/1.0.0",
        report_profile="case-report-production/1.0.0",
        provider_profile=PROFILE,
        execution_lock_profile="runtime-case-host-resource/1.0.0",
    )


def test_fresh_input_uses_original_provider_and_resource_owner(tmp_path, monkeypatch):
    from apps import segman_resource_lock
    from packages.evaluation_preparation import generic_execution_lock
    from packages.evaluation_preparation.generic_campaign import GenericCampaign

    spec = prepared(tmp_path, monkeypatch)
    assert spec["schema_version"] == VERSION
    assert fair.load(spec["key"]) == spec
    config = fair.provider_config(spec)
    old = json.loads(Path(fair.LOCAL_PROVIDER).read_bytes())
    old["network_route"] = json.loads(Path(ROUTE).read_bytes())
    assert config == read_provider_config(old)
    assert config == read_provider_config(json.loads(Path(PROVIDER).read_bytes()))
    binding = dict(
        schema_version="generic-case-binding/1.2.0",
        execution_namespace=spec["key"],
        input_spec_checksum=digest(spec),
    )
    assert (
        segman_resource_lock.verify_input(
            generic_execution_lock.ROOT, tmp_path / spec["key"], binding
        )
        == spec
    )
    campaign = object.__new__(GenericCampaign)
    campaign.spec = spec
    with pytest.raises(ValueError, match="original_execution_lease_required"):
        campaign.execution_session()
    assert fair.owner_key(spec, "phase") != fair.owner_key(dict(spec, key="other"), "phase")
    assert {p.name for p in (tmp_path / spec["key"]).iterdir()} == {"input.json"}


@pytest.mark.parametrize(
    "fault", ["old_version", "path", "checksum", "profile", "lock", "selector"]
)
def test_changed_input_rejected(tmp_path, monkeypatch, fault):
    spec = prepared(tmp_path, monkeypatch)
    if fault == "old_version":
        spec["schema_version"] = "fairvlmed-run-input/1.10.0"
    elif fault == "path":
        spec["provider_configuration"] = fair.LOCAL_PROVIDER
    elif fault == "checksum":
        spec["provider_configuration_checksum"] = digest("wrong")
    elif fault == "profile":
        spec["provider_profile"] = fair.LOCAL_PROFILE
    elif fault == "lock":
        spec.pop("execution_lock_profile")
    else:
        spec["model_selection"] = {}
    (tmp_path / spec["key"] / "input.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="generic_"):
        fair.load(spec["key"])


def continuation(tmp_path, monkeypatch):
    from tests.evaluation_preparation.test_environment_cohort_continuation import fixture

    old, _, path, out = fixture(tmp_path, monkeypatch)
    _, original, freeze = terminal.load(old, check_current=False)
    raw = original.model_dump(mode="json")
    raw.update(
        schema_version="terminal-case-cohort/1.14.0",
        id="all-environment",
        network_route=ROUTE,
        fairvlmed_provider_profile=PROFILE,
        continuation=dict(
            previous_cohort=str(old),
            manifest_checksum=digest(original),
            freeze_checksum=digest(freeze),
            start_index=0,
        ),
    )
    path.write_text(json.dumps(raw))
    return old, terminal.read_manifest(raw), path, out


def test_all_dataset_continuation_freezes_exact_fair_config(tmp_path, monkeypatch):
    old, selected, path, out = continuation(tmp_path, monkeypatch)
    old_bytes = {p.name: p.read_bytes() for p in old.iterdir() if p.is_file()}
    assert terminal.initialize(path, out)["api_requests"] == 0
    _, current, freeze = terminal.load(out, check_current=True)
    assert current == selected
    assert PROVIDER in freeze["configurations"]
    for case in selected.cases:
        argv = terminal.commands(selected, case, out, "prepare")[0]
        if case.dataset == "fairvlmed":
            assert argv[argv.index("--provider-profile") + 1] == PROFILE
            expected = read_provider_config(json.loads(Path(PROVIDER).read_bytes()))
            assert freeze["provider_bindings"][case.id]["provider_checksum"] == digest(expected)
        else:
            assert argv[argv.index("--network-route") + 1] == ROUTE
    assert all((old / name).read_bytes() == raw for name, raw in old_bytes.items())


@pytest.mark.parametrize("fault", ["prepared_fair", "sent_fair", "budget", "profile", "old_schema"])
def test_continuation_never_replays_or_changes_budget(tmp_path, monkeypatch, fault):
    old, selected, path, out = continuation(tmp_path, monkeypatch)
    original = terminal.load(old, check_current=False)[1]
    case = next(c for c in selected.cases if c.dataset == "fairvlmed")
    raw = selected.model_dump(mode="json")
    monkeypatch.setattr(fair, "RUN_ROOT", tmp_path / "unused-runs")
    # directory() default is intentionally not monkeypatched implicitly.
    monkeypatch.setattr(fair, "directory", lambda key: tmp_path / "runs" / key)
    if fault == "prepared_fair":
        argv = terminal.commands(original, case, old, "prepare")[0]
        fair.directory(argv[argv.index("--key") + 1]).mkdir(parents=True)
    elif fault == "sent_fair":
        (old / "commands" / case.id).mkdir(parents=True)
    elif fault == "budget":
        raw["source_recovery"]["window_seconds"] += 1
    elif fault == "profile":
        raw["fairvlmed_provider_profile"] = fair.LOCAL_PROFILE
    else:
        raw["schema_version"] = "terminal-case-cohort/1.13.0"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        terminal.initialize(path, out)
    assert not out.exists()
