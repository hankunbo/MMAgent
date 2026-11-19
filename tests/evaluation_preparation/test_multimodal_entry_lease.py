"""Explicit fresh entry lease through the original command, not a replacement runner."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.multimodal_research import execution_lease as lease
from tests.evaluation_preparation.test_generic_execution_lock import Repository


def bound_entry(directory):
    directory.mkdir(exist_ok=True)
    spec = dict(
        schema_version="multimodal-live-entry/1.29.0",
        execution_profile="relay-root-question-v1",
        entry_execution_lock=dict(
            schema_version=lease.ENTRY_PROFILE, implementation=lease.entry_implementation()
        ),
    )
    (directory / "entry.json").write_text(json.dumps(spec))
    return spec


def fake_repository(monkeypatch):
    monkeypatch.setattr(lease, "entry_target", lambda: "test-owned")
    monkeypatch.setattr(lease, "CasePipelineRepository", Repository)


def test_command_holds_case_lease_during_owner_assembly_and_execute(tmp_path, monkeypatch):
    from packages.multimodal_research import entry, entry_preflight

    fake_repository(monkeypatch)
    spec = bound_entry(tmp_path)
    calls = []

    def execute():
        calls.append("execute")
        assert lease.check_current(tmp_path).spec == spec
        with lease.native_resource_session() as guard:
            guard()
            assert len(lease.check_current().guards) == 2
        assert len(lease.check_current().guards) == 1
        return dict(complete=True, api_requests=0)

    def assemble(output, *, preflight_only=False):
        assert lease.check_current(output).spec == spec
        calls.append("preflight" if preflight_only else "assembly")
        return dict(status="preflight") if preflight_only else SimpleNamespace(execute=execute)

    monkeypatch.setattr(entry, "assemble_entry", assemble)
    assert entry_preflight.command(tmp_path)["complete"]
    assert calls == ["preflight", "assembly", "execute"]
    assert lease.check_current() is None
    assert not (tmp_path / "assembly.lock").exists()


@pytest.mark.parametrize("change", ["version", "profile", "files", "parent"])
def test_extension_cannot_silently_change_legacy_or_successor(tmp_path, change):
    spec = bound_entry(tmp_path)
    if change == "version":
        spec["schema_version"] = "multimodal-live-entry/1.28.0"
    elif change == "profile":
        spec["entry_execution_lock"]["schema_version"] = "caller-authorized"
    elif change == "files":
        spec["entry_execution_lock"]["implementation"] = {}
    else:
        spec["definition_failure_of"] = {"parent": "old"}
    with pytest.raises(ValueError, match="entry_execution_lock_binding"):
        lease.validate_entry(spec)


def test_direct_assembly_without_lease_refused_before_owners(tmp_path, monkeypatch):
    from packages.multimodal_research import entry

    bound_entry(tmp_path)
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("owner assembly bypass"))
    with pytest.raises(ValueError, match="original_execution_lease_required"):
        entry.assemble_entry(tmp_path)


def test_change_and_exception_reset_context(tmp_path, monkeypatch):
    fake_repository(monkeypatch)
    bound_entry(tmp_path)
    with pytest.raises(ValueError, match="input_changed"):
        with lease.entry_session(tmp_path):
            with pytest.raises(ValueError, match="nested"):
                with lease.entry_session(tmp_path):
                    pytest.fail("nested root")
            (tmp_path / "entry.json").write_text("{}")
            lease.check_current()
    assert lease.check_current() is None


def test_campaign_reuses_exact_lease_not_second_flock(tmp_path, monkeypatch):
    from packages.multimodal_research.campaign import MultimodalCampaign

    fake_repository(monkeypatch)
    spec = bound_entry(tmp_path)
    campaign = object.__new__(MultimodalCampaign)
    campaign.output = tmp_path
    campaign.base = SimpleNamespace(entry_execution_lock=spec["entry_execution_lock"])
    with pytest.raises(ValueError, match="original_execution_lease_required"):
        with campaign.execution_session():
            pytest.fail("unguarded campaign")
    with lease.entry_session(tmp_path):
        with campaign.execution_session():
            assert lease.check_current() is not None
    assert not (tmp_path / "lock").exists()


def test_real_synthetic_prepare_preflight_roundtrip(tmp_path, monkeypatch):
    from packages.llm_provider.network_route import prepare_route
    from packages.multimodal_research import entry
    from tests.dataset_adapters.test_multimodal import gdp

    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8888")
    monkeypatch.setattr("packages.llm_provider.network_route.load_route", lambda _: route)
    out = tmp_path / "fresh"
    options = dict(
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
    )
    result = entry.prepare_entry(**options, output=out, entry_execution_lock=lease.ENTRY_PROFILE)
    spec = json.loads((out / "entry.json").read_bytes())
    assert result["api_requests"] == result["database_writes"] == 0
    assert spec["provider"]["transport"]["model"] == "gpt-6-sol"
    assert "model_selection" not in spec
    assert lease.validate_entry(spec)
    assert entry.assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    assert not (out / "protocol.json").exists()
    old = tmp_path / "legacy"
    entry.prepare_entry(**options, output=old)
    assert "entry_execution_lock" not in json.loads((old / "entry.json").read_bytes())


def test_terminal_uses_current_snapshot_without_old_model_transform(tmp_path, monkeypatch):
    from packages.evaluation_preparation import terminal_cohort as terminal

    raw = json.loads(Path("docs/acceptance/terminal-cohort-q-plan/manifest.json").read_bytes())
    raw.update(
        schema_version="terminal-case-cohort/1.10.0",
        supervision_lease="original-runtime-case-pipeline-session/1.0.0",
        fairvlmed_execution_lock="runtime-case-host-resource/1.0.0",
        multimodal_execution_lock=lease.ENTRY_PROFILE,
    )
    manifest = terminal.read_manifest(raw)
    for case in manifest.cases:
        argv = terminal.commands(manifest, case, tmp_path, "prepare")[0]
        if case.dataset != "fairvlmed":
            assert "--model-selection" not in argv
            assert argv[-2:] == ["--entry-execution-lock", lease.ENTRY_PROFILE]
    assert {v["model"] for v in terminal.provider_bindings(manifest).values()} == {"gpt-6-sol"}
    monkeypatch.setattr(terminal, "child_lock_capability", lambda _: dict(available=False))
    monkeypatch.setattr(lease, "entry_capability", lambda: dict(available=True))
    from packages.evaluation_preparation import generic_execution_lock

    monkeypatch.setattr(generic_execution_lock, "capability", lambda: dict(available=True))
    assert terminal.execution_lock_capability(tmp_path, manifest)["available"]
    with pytest.raises(ValueError, match="source_successor_not_supported"):
        terminal.commands(manifest, manifest.cases[1], tmp_path, "source-prepare")
