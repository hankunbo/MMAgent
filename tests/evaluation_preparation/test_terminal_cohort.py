"""Batch supervision does not manufacture owner completion or replay requests."""

import os
import sys
from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as batch
from packages.grounding.models import canonical, validate


def manifest():
    return dict(
        schema_version="terminal-case-cohort/1.1.0",
        id="test-cohort",
        purpose="development_coverage_not_accuracy",
        command_timeout_seconds=60,
        max_cases_per_invocation=4,
        cases=[
            dict(
                id="gamma-one",
                dataset="gamma",
                root="/synthetic/input",
                sample="0001",
                task="advanced_glaucoma",
                active_acquisition="configs/research/test.json",
            )
        ],
    )


def fixture(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    monkeypatch.setattr(batch, "public_config", lambda _: config)
    # The real freeze is exercised separately, not on every isolated command test.
    monkeypatch.setattr(batch, "implementation", lambda: {"source.py": "frozen"})
    # initialize stores paths relative to ROOT.
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    file = tmp_path / "manifest.json"
    file.write_bytes(canonical(manifest()))
    output = tmp_path / "cohort"
    batch.initialize(file, output)
    return output


def test_fixed_argv_separates_data_from_commands_and_reuses_original_cli(tmp_path):
    value = manifest()
    value["cases"][0]["sample"] = "0001; touch /tmp/not-a-command"
    m = validate(batch.Manifest, value)
    argv = batch.commands(m, m.cases[0], tmp_path, "prepare")[0]
    assert argv[:4] == [sys.executable, "-u", "-m", "apps.multimodal_case"]
    assert argv[argv.index("--sample") + 1] == value["cases"][0]["sample"]
    assert "sh" not in argv and "bash" not in argv
    assert "--continuation-of" not in argv


def test_arbitrary_options_duplicate_inputs_and_unbound_modalities_rejected():
    value = manifest()
    value["cases"][0]["shell"] = "anything"
    with pytest.raises(ValueError):
        validate(batch.Manifest, value)
    value = manifest()
    value["cases"].append(dict(value["cases"][0], id="another-id"))
    with pytest.raises(ValueError, match="cohort_duplicate_case_input"):
        validate(batch.Manifest, value)
    value = manifest()
    value["cases"][0]["input_semantics"] = "harvard-gdp-v1"
    with pytest.raises(ValueError, match="cohort_input_semantics"):
        validate(batch.Manifest, value)


def test_real_local_child_exit_preserved_and_repeat_does_not_launch(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    case = validate(batch.Manifest, manifest()).cases[0]
    # A local CPU subprocess tests supervision only, not medical execution.
    argv = [sys.executable, "-c", "print('local child'); raise SystemExit(7)"]
    result = batch.operation(out, case, "run", argv, 10)
    assert result["exit_code"] == 7
    assert result["medical_completion"] == "not_inferred_from_process_exit"
    before = (out / "commands/gamma-one/run/stdout.log").read_bytes()
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("replayed"))
    assert batch.operation(out, case, "run", argv, 10) == result
    assert (out / "commands/gamma-one/run/stdout.log").read_bytes() == before


def test_interrupted_intent_never_dispatches_again(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    case = validate(batch.Manifest, manifest()).cases[0]
    directory = out / "commands/gamma-one/run"
    directory.mkdir(parents=True)
    (directory / "intent.json").write_bytes(b"{}")
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("replayed"))
    result = batch.operation(out, case, "run", ["not-executed"], 10)
    assert result["exit_code"] is None and result["replay_authorized"] is False
    assert batch.status(out)["cases"][0]["commands"]["run"]["status"].startswith("started_or")


def test_timeout_preserves_uncertainty_and_refuses_automatic_retry(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    case = validate(batch.Manifest, manifest()).cases[0]
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    result = batch.operation(out, case, "run", argv, 0.1)
    assert result["status"] == "timeout_remote_state_not_inferred"
    assert result["exit_code"] != 0 and result["replay_authorized"] is False
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("replayed"))
    assert batch.operation(out, case, "run", argv, 0.1) == result


def test_freeze_drift_rejects_execution_but_historical_status_stays_readable(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, "implementation", lambda: {"source.py": "changed"})
    with pytest.raises(ValueError, match="cohort_implementation_changed"):
        batch.run(out, "prepare", ["gamma-one"])
    assert batch.status(out)["cases"][0]["commands"]["prepare"]["status"] == "not_started"


def local_recovery_fixture(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cohort = fixture(tmp_path, monkeypatch)
    m = batch.load(cohort, check_current=False)[1]
    root_argv = batch.commands(m, m.cases[0], cohort, "prepare")[0]
    root = Path(root_argv[root_argv.index("--output") + 1])
    plan = tmp_path / "fold-plan.json"
    plan.write_bytes(
        canonical(
            dict(
                schema_version="accepted-review-aggregation-plan/1.3.0",
                facts=str(tmp_path / "failed"),
                origins=[dict(campaign=str(root))],
            )
        )
    )
    identity = dict(root=str(root), root_result="old", fact_result="failed")
    monkeypatch.setattr(batch, "local_recovery_identity", lambda *args: identity)
    calls = []

    def operation(out, case, action, argv, timeout):
        calls.append((out, case, action, argv, timeout))
        return dict(exit_code=0, status="command_exited")

    monkeypatch.setattr(batch, "operation", operation)
    return SimpleNamespace(
        cohort=cohort, plan=plan, output=tmp_path / "local-recovery", calls=calls, identity=identity
    )


def test_explicit_local_recovery_uses_only_original_zero_api_cli_and_separate_freeze(
    tmp_path, monkeypatch
):
    f = local_recovery_fixture(tmp_path, monkeypatch)
    old_freeze = (f.cohort / "freeze.json").read_bytes()
    # Repairs may differ from historical M; old normal dispatch remains refused.
    monkeypatch.setattr(batch, "implementation", lambda: {"source.py": "repair"})
    result = batch.recover_accepted(f.cohort, "gamma-one", f.plan, f.output)
    assert result["exit_code"] == 0 and result["current_consumption_authorized"] is False
    assert (f.cohort / "freeze.json").read_bytes() == old_freeze
    assert f.calls[0][3] == [
        sys.executable,
        "-u",
        "-m",
        "apps.multimodal_case",
        "aggregate-accepted",
        "--plan",
        str(f.output / "aggregation-plan.json"),
        "--output",
        result["aggregation_output"],
    ]
    assert f.calls[0][4] == 900
    record = batch.read(f.output / "recovery.json")
    assert record["maximum_new_api_requests"] == 0
    assert record["implementation"] == {"source.py": "repair"}
    assert Path(result["aggregation_output"]).name.startswith("local-fold-")
    with pytest.raises(ValueError, match="cohort_implementation_changed"):
        batch.run(f.cohort, "run", ["gamma-one"])


def test_distinct_recovery_paths_never_share_original_export_basename(tmp_path, monkeypatch):
    f = local_recovery_fixture(tmp_path, monkeypatch)
    first = batch.recover_accepted(f.cohort, "gamma-one", f.plan, f.output)
    second = batch.recover_accepted(
        f.cohort, "gamma-one", f.plan, tmp_path / "other" / f.output.name
    )
    assert Path(first["aggregation_output"]).name != Path(second["aggregation_output"]).name


@pytest.mark.parametrize("changed", ["implementation", "identity", "plan"])
def test_local_recovery_rejects_changed_frozen_operation_before_dispatch(
    tmp_path, monkeypatch, changed
):
    f = local_recovery_fixture(tmp_path, monkeypatch)
    batch.recover_accepted(f.cohort, "gamma-one", f.plan, f.output)
    f.calls.clear()
    if changed == "implementation":
        monkeypatch.setattr(batch, "implementation", lambda: {"source.py": "later"})
    elif changed == "identity":
        f.identity["fact_result"] = "different"
    else:
        raw = batch.read(f.plan)
        raw["origins"].append(dict(campaign=str(tmp_path / "another")))
        f.plan.write_bytes(canonical(raw))
    with pytest.raises(ValueError, match="freeze_or_plan_conflict"):
        batch.recover_accepted(f.cohort, "gamma-one", f.plan, f.output)
    assert not f.calls


def test_local_recovery_rejects_legacy_plan_unknown_case_and_output_overlap(tmp_path, monkeypatch):
    f = local_recovery_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="case_required"):
        batch.recover_accepted(f.cohort, "other", f.plan, f.output)
    with pytest.raises(ValueError, match="output_overlaps_history"):
        batch.recover_accepted(f.cohort, "gamma-one", f.plan, f.cohort / "overwrite")
    raw = batch.read(f.plan)
    raw["schema_version"] = "accepted-review-aggregation-plan/1.1.0"
    f.plan.write_bytes(canonical(raw))
    with pytest.raises(ValueError, match="accounted_plan_required"):
        batch.recover_accepted(f.cohort, "gamma-one", f.plan, f.output)
    assert not f.calls


def test_local_recovery_source_owner_rejection_precedes_output_creation(tmp_path, monkeypatch):
    f = local_recovery_fixture(tmp_path, monkeypatch)

    def revoked(*args):
        raise ValueError("original_owner_revoked")

    monkeypatch.setattr(batch, "local_recovery_identity", revoked)
    with pytest.raises(ValueError, match="original_owner_revoked"):
        batch.recover_accepted(f.cohort, "gamma-one", f.plan, f.output)
    assert not f.output.exists() and not f.calls


@pytest.mark.parametrize("change", [None, "case", "task", "input", "ancestor", "duplicate"])
def test_recovery_identity_uses_owner_case_input_and_exact_root_lineage(
    tmp_path, monkeypatch, change
):
    from types import SimpleNamespace

    from packages.multimodal_research import accepted_aggregation as a
    from packages.multimodal_research import campaign_renewal as renewal

    cohort = fixture(tmp_path, monkeypatch)
    m = batch.load(cohort, check_current=False)[1]
    argv = batch.commands(m, m.cases[0], cohort, "prepare")[0]
    root = str(Path(argv[argv.index("--output") + 1]))
    owner = SimpleNamespace(base=SimpleNamespace(manifest=SimpleNamespace(checksum="input")))
    root_result = dict(case_code="same-case", task={"target": "proposition"}, new_requests=16)
    leaf = dict(
        root_result,
        new_requests=3,
        previous_attempts=[
            dict(
                directory=root,
                requests=16,
                checksum=batch.digest(root_result),
            )
        ],
    )
    leaf_owner = owner
    if change == "case":
        leaf["case_code"] = "foreign-case"
    elif change == "task":
        leaf["task"] = {"target": "another-proposition"}
    elif change == "input":
        leaf_owner = SimpleNamespace(
            base=SimpleNamespace(manifest=SimpleNamespace(checksum="other"))
        )
    elif change == "ancestor":
        leaf["previous_attempts"][0]["checksum"] = batch.digest("another-result")
    elif change == "duplicate":
        leaf["previous_attempts"].append(dict(leaf["previous_attempts"][0]))
    monkeypatch.setattr(renewal, "completed_parent", lambda _: (owner, root_result))
    monkeypatch.setattr(a, "plan_campaign", lambda _: (leaf_owner, leaf, object()))
    plan = a.read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.3.0",
            facts=str(tmp_path / "leaf"),
            origins=[dict(campaign=root)],
        )
    )
    if change:
        with pytest.raises(ValueError, match="case_binding|root_lineage"):
            batch.local_recovery_identity(cohort, m, m.cases[0], plan)
    else:
        result = batch.local_recovery_identity(cohort, m, m.cases[0], plan)
        assert result["root_result"] == batch.digest(root_result)
        assert result["fact_result"] == batch.digest(leaf)


def test_manifest_change_cannot_change_existing_case_identity(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    value = batch.read(out / "manifest.json")
    value["cases"][0]["sample"] = "0002"
    (out / "manifest.json").write_bytes(canonical(value))
    with pytest.raises(ValueError, match="cohort_manifest_binding"):
        batch.status(out)


def test_prerequisites_and_html_escape_without_owner_dispatch(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    result = batch.run(out, "run", ["gamma-one"])
    assert result[0]["status"] == "prerequisite_not_completed"
    assert result[0]["exit_code"] is None
    result = batch.export_status(out)
    assert Path(result["html"]).is_file() and result["api_requests"] == 0
    assert "不是医学接纳" in Path(result["html"]).read_text()
    assert batch.export_status(out) == result


def test_freeze_includes_both_case_engines_and_public_provider_configuration():
    files = batch.implementation()
    for name in (
        "apps/case_cohort.py",
        "apps/fairvlmed_case.py",
        "apps/multimodal_case.py",
        "packages/evaluation_preparation/generic_run.py",
        "configs/agents/fairvlmed_relay_review_shape_v1.json",
    ):
        assert name in files
    assert not any(".local" in path for path in files)


def test_public_pilot_manifest_and_schema_cover_four_inputs_without_labels():
    m = validate(batch.Manifest, batch.read("configs/research/terminal_cohort_pilot_v2.json"))
    assert len(m.cases) == 4
    assert {case.dataset for case in m.cases} == {"fairvlmed", "gamma", "harvard_gdp", "grape"}
    assert batch.Manifest.model_json_schema()["additionalProperties"] is False
    from packages.multimodal_research.active_requirements import load

    for case in m.cases:
        if case.active_acquisition:
            profile = load(case.active_acquisition)
            assert profile["definition_expansion"]["prompt_version"] == "definition-draft-6.4.0"
        argv = batch.commands(m, case, Path("/synthetic/cohort"), "prepare")[0]
        assert "--labels" not in argv and "--ground-truth" not in argv


def test_execute_is_just_in_time_and_failed_case_does_not_hide_next_case(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    value = manifest()
    value["cases"].append(dict(value["cases"][0], id="gamma-two", sample="0002"))
    m = validate(batch.Manifest, value)
    original_load = batch.load
    monkeypatch.setattr(batch, "load", lambda path, **kw: (Path(path), m, {}))
    seen = []

    def action(output, manifest, case, current):
        seen.append((case.id, current))
        return [
            dict(case=case.id, exit_code=2 if (case.id, current) == ("gamma-one", "run") else 0)
        ]

    monkeypatch.setattr(batch, "case_action", action)
    result = batch.run(out, "execute", ["gamma-one", "gamma-two"])
    assert seen == [
        (case, action)
        for case in ("gamma-one", "gamma-two")
        for action in ("prepare", "preflight", "run")
    ]
    assert any(row["exit_code"] == 2 for row in result)
    assert result[-1]["exit_code"] == 0
    # This is a batch control test; original case clinical owners are not mocked
    # to claim medical success, and the original stored manifest stays intact.
    assert len(original_load(out, check_current=False)[1].cases) == 1


@pytest.mark.parametrize("dataset", ["gamma", "fairvlmed"])
def test_new_case_identity_is_stable_and_bound_to_batch_and_output(tmp_path, dataset):
    raw = manifest()
    if dataset == "fairvlmed":
        raw["cases"][0].update(dataset=dataset, task="glaucoma", active_acquisition=None)
    m = validate(batch.Manifest, raw)
    a = batch.commands(m, m.cases[0], tmp_path / "a", "prepare")[0]
    b = batch.commands(m, m.cases[0], tmp_path / "b", "prepare")[0]
    flag = "--key" if dataset == "fairvlmed" else "--output"
    first = a[a.index(flag) + 1]
    assert Path(first).name != Path(b[b.index(flag) + 1]).name
    assert len(Path(first).name) <= 64
    assert a == batch.commands(m, m.cases[0], tmp_path / "a", "prepare")[0]
    raw["id"] += "-another"
    other = validate(batch.Manifest, raw)
    c = batch.commands(other, other.cases[0], tmp_path / "a", "prepare")[0]
    assert c[c.index(flag) + 1] != first


def test_legacy_commands_remain_exact_and_new_legacy_creation_refused(tmp_path):
    raw = manifest()
    raw["schema_version"] = "terminal-case-cohort/1.0.0"
    m = validate(batch.Manifest, raw)
    for action, verb in (("preflight", "preflight"), ("run", "run"), ("export", "restore-report")):
        assert batch.commands(m, m.cases[0], tmp_path, action) == [
            [
                sys.executable,
                "-u",
                "-m",
                "apps.multimodal_case",
                verb,
                "--input",
                str(tmp_path / "inputs" / "gamma-one"),
            ]
        ]
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical(raw))
    with pytest.raises(ValueError, match="cohort_legacy_creation_disabled"):
        batch.initialize(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_stale_or_reused_pid_is_not_a_running_process(tmp_path):
    saved = batch.process_identity(os.getpid())
    assert saved["start_ticks"]
    path = tmp_path / "process.json"
    path.write_bytes(canonical(saved))
    assert batch.process_status(tmp_path) == "live_same_process"
    path.write_bytes(canonical(dict(saved, start_ticks="0")))
    assert (
        batch.process_status(tmp_path)
        == "original_process_not_present_owner_reconciliation_required"
    )
    path.write_bytes(canonical(dict(pid=os.getpid())))
    assert batch.process_status(tmp_path) == "identity_not_recorded_no_liveness_inference"
