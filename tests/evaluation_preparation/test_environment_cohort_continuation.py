"""Versioned routing of an untouched queue slice, never old-case replay."""

from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as terminal
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config
from tests.evaluation_preparation.test_uniform_model_cohort import manifest as base_manifest


def fixture(tmp_path, monkeypatch):
    from packages.llm_provider import network_route

    # This seam does not send HTTP; actual configured-route probes are separate.
    monkeypatch.setattr(network_route, "route_opener", lambda *a: None)
    raw = base_manifest()
    raw.update(
        schema_version="terminal-case-cohort/1.12.0",
        network_route="configs/agents/server_local_https_route_v1.json",
        fairvlmed_model_selection=None,
        fairvlmed_provider_profile="relay-local-gpt6-shape-v1",
        review_contract="rating-presence/1.1.0",
        supervision_lease="original-runtime-case-pipeline-session/1.0.0",
        fairvlmed_execution_lock="runtime-case-host-resource/1.0.0",
        multimodal_execution_lock="multimodal-runtime-entry/1.0.0",
        multimodal_http_client="configs/agents/system_curl_client_v1.json",
    )
    original = terminal.read_manifest(raw)
    source = tmp_path / "old-manifest.json"
    source.write_text(original.model_dump_json())
    old = tmp_path / "old-cohort"
    terminal.initialize(source, old)
    freeze = terminal.read(old / "freeze.json")
    raw.update(
        schema_version="terminal-case-cohort/1.13.0",
        id="unstarted-environment",
        network_route="configs/agents/environment_https_route_v1.json",
        cases=raw["cases"][1:3],
        continuation=dict(
            previous_cohort=str(old),
            manifest_checksum=digest(original),
            freeze_checksum=digest(freeze),
            start_index=1,
        ),
    )
    selected = terminal.read_manifest(raw)
    new = tmp_path / "new-cohort"
    manifest_file = tmp_path / "new-manifest.json"
    manifest_file.write_text(selected.model_dump_json())
    return old, selected, manifest_file, new


def test_actual_owner_config_and_cli_binding(tmp_path, monkeypatch):
    old, selected, path, out = fixture(tmp_path, monkeypatch)
    originals = {p.name: p.read_bytes() for p in old.iterdir() if p.is_file()}
    assert terminal.initialize(path, out)["api_requests"] == 0
    _, restored, frozen = terminal.load(out, check_current=True)
    assert restored == selected
    expected = read_provider_config(
        terminal.read("configs/agents/multimodal_relay_environment_gpt6_review_curl_v1.json")
    )
    for case in selected.cases:
        assert frozen["provider_bindings"][case.id]["provider_checksum"] == digest(expected)
        argv = terminal.commands(selected, case, out, "prepare")[0]
        assert argv[3] == "apps.multimodal_case"
        assert argv[argv.index("--network-route") + 1] == selected.network_route
        assert argv[argv.index("--http-client") + 1] == selected.multimodal_http_client
        assert Path(argv[argv.index("--output") + 1]).name.startswith("cohort-")
        assert "--entry-execution-lock" in argv
    assert frozen["continuation"]["original_selected_requests"] == 0
    assert frozen["continuation"]["preceding_case_ids"] == ["gamma"]
    assert frozen["continuation"]["following_case_ids"] == ["fair"]
    assert all((old / p).read_bytes() == value for p, value in originals.items())


@pytest.mark.parametrize("fault", ["sent", "prepared", "slice", "freeze", "budget", "fair"])
def test_unsafe_transition_rejected_before_output(tmp_path, monkeypatch, fault):
    old, selected, path, out = fixture(tmp_path, monkeypatch)
    raw = selected.model_dump(mode="json")
    if fault == "sent":
        (old / "commands" / selected.cases[0].id).mkdir(parents=True)
    elif fault == "prepared":
        original = terminal.load(old, check_current=False)[1]
        argv = terminal.commands(original, selected.cases[0], old, "prepare")[0]
        Path(argv[argv.index("--output") + 1]).mkdir(parents=True)
    elif fault == "slice":
        raw["cases"].reverse()
    elif fault == "freeze":
        raw["continuation"]["freeze_checksum"] = digest("wrong")
    elif fault == "budget":
        raw["source_recovery"]["window_seconds"] += 1
    else:
        raw["cases"] = [base_manifest()["cases"][-1]]
    import json

    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        terminal.initialize(path, out)
    assert not out.exists()


def test_old_case_started_after_init_prevents_new_dispatch(tmp_path, monkeypatch):
    old, selected, path, out = fixture(tmp_path, monkeypatch)
    terminal.initialize(path, out)
    (old / "commands" / selected.cases[0].id).mkdir(parents=True)
    with pytest.raises(ValueError, match="already_started"):
        terminal.load(out, check_current=True)


def test_old_version_cannot_gain_new_route(tmp_path, monkeypatch):
    _, selected, _, _ = fixture(tmp_path, monkeypatch)
    raw = selected.model_dump(mode="json")
    raw.pop("continuation")
    raw["schema_version"] = "terminal-case-cohort/1.12.0"
    with pytest.raises(ValueError):
        terminal.read_manifest(raw)
