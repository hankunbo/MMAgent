"""The opt-in Rating contract reaches the existing CLI and cold recovery."""

import json
from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as cohort
from packages.llm_provider.review_rating import POLICY, apply, check_entry_binding
from tests.evaluation_preparation.test_vision_feedback_entry import (
    test_actual_prepare_preflight_provider_and_cold_binding as exercise,
)


@pytest.mark.parametrize("policy", [POLICY, "rating-presence/1.1.0"])
def test_actual_prepare_preflight_and_source_cold_recovery(tmp_path, monkeypatch, policy):
    exercise(tmp_path, monkeypatch, review_contract=policy)


@pytest.mark.parametrize("policy,version", [(POLICY, 3), ("rating-presence/1.1.0", 4)])
def test_manifest_cli_and_actual_provider_match_public_gate(tmp_path, monkeypatch, policy, version):
    raw = json.loads(Path("docs/acceptance/terminal-cohort-q-plan/manifest.json").read_bytes())
    raw.update(
        schema_version="terminal-case-cohort/1.7.0",
        review_contract=policy,
        network_route="configs/agents/server_local_https_route_v1.json",
    )
    raw["cases"] = [c for c in raw["cases"] if c["dataset"] in {"harvard_gdp", "grape"}]
    manifest = cohort.read_manifest(raw)
    from packages.grounding.models import digest
    from packages.llm_provider.config import read_provider_config

    public = read_provider_config(
        json.loads(
            Path(f"configs/agents/multimodal_relay_local_gpt6_review_v{version}.json").read_bytes()
        )
    )
    for case in manifest.cases:
        for action in ("prepare", "source-prepare"):
            argv = cohort.commands(manifest, case, tmp_path, action)[0]
            assert argv[argv.index("--review-contract") + 1] == policy
        assert cohort.provider_bindings(manifest)[case.id]["provider_checksum"] == digest(public)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(raw))
    output = tmp_path / "cohort"
    assert cohort.initialize(manifest_file, output)["api_requests"] == 0
    _, saved, frozen = cohort.load(output, check_current=True)
    assert saved == manifest
    assert frozen["provider_bindings"] == cohort.provider_bindings(manifest)
    with pytest.raises(ValueError):
        cohort.read_manifest(dict(raw, review_contract="unknown"))
    raw.pop("review_contract")
    raw["schema_version"] = "terminal-case-cohort/1.6.0"
    old = cohort.read_manifest(raw)
    assert "--review-contract" not in cohort.commands(old, old.cases[0], tmp_path, "prepare")[0]


def test_contract_is_not_silent_upgrade():
    from packages.llm_provider.config import read_provider_config

    value = json.loads(
        Path("configs/agents/multimodal_relay_local_gpt6_review_v2.json").read_bytes()
    )
    old = read_provider_config(value)
    assert apply(old, None) is old
    new = apply(old, POLICY)
    with pytest.raises(ValueError, match="missing_entry_policy"):
        check_entry_binding(dict(provider=new.model_dump(mode="json")))
    with pytest.raises(ValueError, match="configuration"):
        apply(old, "unbounded")


def test_new_manifest_retains_output_bound_case_identity(tmp_path):
    raw = json.loads(Path("docs/acceptance/terminal-cohort-q-plan/manifest.json").read_bytes())
    raw.update(schema_version="terminal-case-cohort/1.7.0", review_contract="rating-presence/1.1.0")
    manifest = cohort.read_manifest(raw)
    case = next(c for c in manifest.cases if c.dataset == "harvard_gdp")
    from packages.grounding.models import digest

    names = []
    for output in (tmp_path / "one", tmp_path / "two"):
        argv = cohort.commands(manifest, case, output, "prepare")[0]
        name = Path(argv[argv.index("--output") + 1]).name
        assert name == "cohort-" + digest((manifest.id, case.id, str(output.resolve())))[7:55]
        names.append(name)
    assert names[0] != names[1]


def recovery_scenario(tmp_path, version):
    from packages.grounding.models import canonical
    from packages.object_store import checksum

    raw = json.loads(Path("docs/acceptance/terminal-cohort-q-plan/manifest.json").read_bytes())
    raw["schema_version"] = "terminal-case-cohort/" + version
    if version == "1.7.0":
        raw["review_contract"] = "rating-presence/1.1.0"
    manifest = cohort.read_manifest(raw)
    case = next(c for c in manifest.cases if c.dataset == "harvard_gdp")
    output = tmp_path / "cohort"
    argv = cohort.commands(manifest, case, output, "prepare")[0]
    parent = Path(argv[argv.index("--output") + 1])
    parent.mkdir(parents=True)
    failure = dict(
        stage="definitions",
        error_code="transport",
        failure_kind="dispatched_unknown",
        transport=dict(remote_response_unknown=1, dispatch_unknown=0),
    )
    (parent / "stage-failure.json").write_bytes(canonical(failure))
    command = output / "commands" / case.id / "run"
    command.mkdir(parents=True)
    log = b"original technical failure\n"
    (command / "stdout.log").write_bytes(log)
    (command / "result.json").write_bytes(
        canonical(
            dict(
                status="command_exited",
                exit_code=2,
                command=cohort.commands(manifest, case, output, "run")[0],
                log_checksum=checksum(log),
            )
        )
    )
    return output, manifest, case, parent


@pytest.mark.parametrize("version", ["1.6.0", "1.7.0"])
def test_recovery_guard_and_status_support_current_manifest(tmp_path, version):
    output, manifest, case, parent = recovery_scenario(tmp_path, version)
    before = (parent / "stage-failure.json").read_bytes()
    assert cohort.source_recovery_ready(output, manifest, case)
    status = cohort.recovery_progress(manifest, case, output)
    assert status["status"] != "not_supported" and status["authorization"] is False
    assert (parent / "stage-failure.json").read_bytes() == before


@pytest.mark.parametrize("change", ["cancel", "result", "wrong_stage", "multiple_unknown"])
def test_current_manifest_does_not_authorize_unsafe_source_parent(tmp_path, change):
    output, manifest, case, parent = recovery_scenario(tmp_path, "1.7.0")
    if change == "cancel":
        (parent / "cancel").touch()
    elif change == "result":
        (parent / "result.json").write_text("{}")
    else:
        value = json.loads((parent / "stage-failure.json").read_bytes())
        if change == "wrong_stage":
            value["stage"] = "review"
        else:
            value["transport"]["remote_response_unknown"] = 2
        (parent / "stage-failure.json").write_text(json.dumps(value))
    assert not cohort.source_recovery_ready(output, manifest, case)
