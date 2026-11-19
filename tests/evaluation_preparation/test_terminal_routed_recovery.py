"""Explicit 1.3 supervision reaches the original routed backlog owner, never Source replay."""

from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as batch
from packages.grounding.models import canonical
from tests.evaluation_preparation.test_terminal_cohort import manifest
from tests.evaluation_preparation.test_terminal_source_recovery import failed_case


def routed_manifest():
    value = manifest()
    value.update(
        schema_version="terminal-case-cohort/1.3.0",
        network_route="configs/agents/environment_https_route_v1.json",
        command_timeout_seconds=10800,
        source_recovery=dict(
            mode="technical_backlog_no_source_replay",
            window_seconds=2700,
            authorization_code="bounded-development-recovery",
        ),
    )
    return batch.read_manifest(value)


def routed_failure(tmp_path, monkeypatch):
    out, _, case, parent, failure = failed_case(tmp_path, monkeypatch)
    m = routed_manifest()
    # The base key is stable; use the exact new argv in the synthetic receipt.
    receipt_path = out / "commands" / case.id / "run/result.json"
    receipt = batch.read(receipt_path)
    receipt["command"] = batch.commands(m, case, out, "run")[0]
    receipt_path.write_bytes(canonical(receipt))
    failure.update(
        error_code="transport",
        failure_kind="dispatched_unknown",
        transport=dict(remote_response_unknown=1, dispatch_unknown=0),
    )
    (parent / "stage-failure.json").write_bytes(canonical(failure))
    return out, m, case, parent, failure


def test_original_prepare_keeps_route_and_explicit_window_no_source_replay(tmp_path, monkeypatch):
    out, m, case, parent, failure = routed_failure(tmp_path, monkeypatch)
    original = (parent / "stage-failure.json").read_bytes()
    assert batch.source_recovery_ready(out, m, case)
    argv = batch.commands(m, case, out, "source-prepare")[0]
    assert argv[:4] == [batch.sys.executable, "-u", "-m", "apps.multimodal_case"]
    assert argv[argv.index("--definition-failure-of") + 1] == str(parent)
    assert argv[argv.index("--network-route") + 1] == m.network_route
    assert argv[argv.index("--source-fallback-seconds") + 1] == "2700"
    assert argv[argv.index("--source-fallback-authorization") + 1] == (
        "bounded-development-recovery:" + m.id + ":" + case.id
    )
    assert "--defer-failed-definition-expansion" in argv
    assert not any(x in argv for x in ("--max-requests", "--seconds", "--retry"))
    assert (parent / "stage-failure.json").read_bytes() == original
    child = argv[argv.index("--output") + 1]
    for action, verb in (("source-preflight", "preflight"), ("source-run", "run")):
        assert batch.commands(m, case, out, action)[0][-3:] == [verb, "--input", child]


@pytest.mark.parametrize("version", ["1.1.0", "1.2.0"])
def test_old_manifest_does_not_gain_unknown_recovery(tmp_path, monkeypatch, version):
    out, m, case, parent, failure = routed_failure(tmp_path, monkeypatch)
    value = m.model_dump(mode="json")
    value["schema_version"] = "terminal-case-cohort/" + version
    value.pop("source_recovery")
    if version == "1.1.0":
        value.pop("network_route")
    old = batch.read_manifest(value)
    assert not batch.source_recovery_ready(out, old, case)
    if version == "1.2.0":
        with pytest.raises(ValueError, match="cohort_source_recovery_not_supported"):
            batch.commands(old, case, out, "source-prepare")


@pytest.mark.parametrize(
    "change",
    [
        "wrong_stage",
        "multiple_unknown",
        "boolean_count",
        "no_unknown",
        "wrong_kind",
        "cancelled",
        "result",
    ],
)
def test_fast_rejection_does_not_turn_arbitrary_failure_into_successor(
    tmp_path, monkeypatch, change
):
    out, m, case, parent, failure = routed_failure(tmp_path, monkeypatch)
    if change == "wrong_stage":
        failure["stage"] = "review"
    elif change == "multiple_unknown":
        failure["transport"]["remote_response_unknown"] = 2
    elif change == "boolean_count":
        failure["transport"]["remote_response_unknown"] = True
    elif change == "no_unknown":
        failure["transport"]["remote_response_unknown"] = 0
    elif change == "wrong_kind":
        failure["failure_kind"] = "confirmed_response_or_domain_refusal"
    elif change == "cancelled":
        (parent / "cancel").write_bytes(b"")
    else:
        (parent / "result.json").write_bytes(b"{}")
    (parent / "stage-failure.json").write_bytes(canonical(failure))
    assert not batch.source_recovery_ready(out, m, case)


@pytest.mark.parametrize("mutation", ["no_policy", "long_window", "short_supervisor", "batch_one"])
def test_manifest_requires_explicit_bounded_supported_policy(mutation):
    value = routed_manifest().model_dump(mode="json")
    if mutation == "no_policy":
        value.pop("source_recovery")
    elif mutation == "long_window":
        value["source_recovery"]["window_seconds"] = 5401
    elif mutation == "short_supervisor":
        value["command_timeout_seconds"] = 60
    else:
        value["cases"][0]["review_batch_size"] = 1
    with pytest.raises(ValueError):
        batch.read_manifest(value)


def test_initialize_and_original_supervision_do_not_synthesize_owner_success(tmp_path, monkeypatch):
    value = routed_manifest()
    root = tmp_path / "repo"
    root.mkdir()
    config = root / "config.json"
    config.write_bytes(b"{}")
    from packages.llm_provider import network_route

    monkeypatch.setattr(network_route, "load_route", lambda _: object())
    monkeypatch.setattr(network_route, "route_opener", lambda *a: object())
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "public_config", lambda _: config)
    monkeypatch.setattr(batch, "implementation", lambda: {"owner.py": "unchanged"})
    source = root / "manifest.json"
    source.write_bytes(canonical(value))
    out = root / "cohort"
    assert batch.initialize(source, out)["api_requests"] == 0
    _, actual, frozen = batch.load(out, check_current=True)
    assert isinstance(actual, batch.RoutedRecoveryManifest)
    assert frozen["automatic_unknown_replay"] is False
    calls = []

    def action(output, manifest, case, name):
        calls.append(name)
        return [dict(exit_code=2 if name == "source-preflight" else 0)]

    monkeypatch.setattr(batch, "case_action", action)
    result = batch.run(out, "recover-source", [value.cases[0].id])
    assert result[-1]["exit_code"] == 2
    assert calls == ["source-prepare", "source-preflight"]
    assert not list(Path(out).glob("inputs/*/result.json"))


def test_routed_argv_parses_at_actual_case_cli_without_dispatch(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from apps import multimodal_case
    from packages.multimodal_research import entry

    m = routed_manifest()
    argv = batch.commands(m, m.cases[0], tmp_path, "source-prepare")[0]
    received = []

    def preparation_boundary(**kwargs):
        received.append(kwargs)
        return dict(api_requests=0, database_writes=0)

    monkeypatch.setattr(entry, "prepare_entry", preparation_boundary)
    monkeypatch.setattr(sys, "argv", ["apps.multimodal_case", *argv[4:]])
    assert multimodal_case.main() == 0
    assert json.loads(capsys.readouterr().out)["api_requests"] == 0
    assert len(received) == 1
    assert received[0]["source_fallback_seconds"] == 2700
    assert received[0]["defer_failed_definition_expansion"] is True
    assert received[0]["network_route"] == m.network_route
    assert received[0]["definition_failure_of"] != received[0]["output"]


def test_four_input_public_configuration_has_explicit_recovery_and_serial_dispatch():
    value = batch.read_manifest(batch.read("configs/research/terminal_cohort_coverage_v1.json"))
    assert isinstance(value, batch.RoutedRecoveryManifest)
    assert {c.dataset for c in value.cases} == {"fairvlmed", "harvard_gdp", "gamma", "grape"}
    assert len(value.cases) == 4 and value.max_cases_per_invocation == 1
    assert value.purpose == "development_coverage_not_accuracy"
    assert value.source_recovery.window_seconds == 5400
    assert value.source_recovery.mode == "technical_backlog_no_source_replay"


@pytest.mark.parametrize("broken", ["missing_log", "invalid_receipt"])
def test_missing_capture_evidence_refuses_recovery(tmp_path, monkeypatch, broken):
    out, m, case, _, _ = routed_failure(tmp_path, monkeypatch)
    command = out / "commands" / case.id / "run"
    if broken == "missing_log":
        (command / "stdout.log").unlink()
    else:
        (command / "result.json").write_bytes(b"[]")
    assert not batch.source_recovery_ready(out, m, case)
