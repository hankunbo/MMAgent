"""Explicit saved history is explanatory, never a new observation or grant."""

import json

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.demand_history import inspect_history
from tests.evaluation_preparation.test_demand_coverage import report


def fixture(tmp_path):
    root = tmp_path / "history"
    old = root / "old"
    old.mkdir(parents=True)
    phase = dict(
        target="three",
        run_id="RUN:OLD",
        accepted=False,
        error="timeout",
        requests=dict(run_id="RUN:OLD", api_requests_observed=1),
        transport=dict(remote_response_unknown=1),
    )
    historical = dict(case_code="CASE:TEST", task=dict(target_code="GOAL:TEST"), phases=[phase])
    data = report()
    data["result"]["previous_attempts"] = [dict(directory=str(old), checksum=digest(historical))]
    path = tmp_path / "report.json"
    (old / "result.json").write_text(json.dumps(historical))
    path.write_text(json.dumps(data))
    return path, root, data, historical


def test_unknown_ancestor_is_not_unattempted_or_a_new_score(tmp_path):
    path, root, data, _ = fixture(tmp_path)
    before = path.read_bytes()
    result = inspect_history(path, root)
    rows = {r["id"]: r for r in result["rows"]}
    assert rows["three"]["reconciled_state"] == "historical_remote_or_dispatch_unknown"
    assert rows["three"]["historical_attempts"][0]["run_id"] == "RUN:OLD"
    assert rows["three"]["e"] is None
    assert rows["one"]["e"] == data["scores"]["reviews"][0]["judgment"]["value"] == 0
    assert not result["current_execution_authorization"]
    assert not result["historical_unknown_replay_authorized"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", ["hash", "case", "target", "outside", "duplicate", "request_run"])
def test_changed_or_unbound_history_rejected(tmp_path, fault):
    path, root, data, old = fixture(tmp_path)
    if fault in {"case", "target", "request_run"}:
        if fault == "case":
            old["case_code"] = "CASE:OTHER"
        elif fault == "target":
            old["task"]["target_code"] = "GOAL:OTHER"
        else:
            old["phases"][0]["requests"]["run_id"] = "RUN:OTHER"
        data["result"]["previous_attempts"][0]["checksum"] = digest(old)
    elif fault == "hash":
        old["phases"][0]["error"] = "modified"
    elif fault == "outside":
        root = tmp_path / "elsewhere"
        root.mkdir()
    elif fault == "duplicate":
        data["result"]["previous_attempts"] *= 2
    (tmp_path / "history/old/result.json").write_text(json.dumps(old))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="demand_history"):
        inspect_history(path, root)


def test_duplicate_same_phase_counts_once_but_conflict_rejected(tmp_path):
    path, root, data, old = fixture(tmp_path)
    data["result"]["phases"] = old["phases"]
    path.write_text(json.dumps(data))
    assert len(inspect_history(path, root)["rows"][2]["historical_attempts"]) == 1
    data["result"]["phases"][0]["error"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="demand_history_phase_conflict"):
        inspect_history(path, root)


def test_source_failure_is_not_patient_observation(tmp_path):
    path, root, data, _ = fixture(tmp_path)
    source = root / "source"
    source.mkdir()
    failure = dict(stage="definitions", error_code="timeout")
    (source / "stage-failure.json").write_text(json.dumps(failure))
    data["result"]["previous_attempts"].append(
        dict(
            directory=str(source), checksum=digest(failure), kind="terminal_source_unknown_deferred"
        )
    )
    path.write_text(json.dumps(data))
    result = inspect_history(path, root)
    assert len(result["history_files"]) == 2
    assert sum(len(row["historical_attempts"]) for row in result["rows"]) == 1


def test_cli_explicit_history_does_not_change_default(tmp_path, monkeypatch, capsys):
    from apps.multimodal_case import main

    path, root, _, _ = fixture(tmp_path)
    monkeypatch.setattr("sys.argv", ["multimodal_case", "coverage", "--report", str(path)])
    assert main() == 0
    assert (
        json.loads(capsys.readouterr().out)["coverage"]["schema_version"]
        == "demand-execution-coverage/1.1.0"
    )
    monkeypatch.setattr(
        "sys.argv",
        ["multimodal_case", "coverage", "--report", str(path), "--history-root", str(root)],
    )
    assert main() == 0
    assert (
        json.loads(capsys.readouterr().out)["schema_version"] == "saved-demand-history-audit/1.0.0"
    )


def test_current_observation_not_erased_by_historical_unknown(tmp_path):
    path, root, data, _ = fixture(tmp_path)
    data["requirements"][2]["observation"] = dict(status="obtained")
    path.write_text(json.dumps(data))
    row = inspect_history(path, root)["rows"][2]
    assert row["reconciled_state"] == "obtained_pending_review"
    assert row["historical_unknown_present"] and row["obtained"]
    assert row["e"] is None


def test_file_symlink_outside_declared_directory_rejected(tmp_path):
    path, root, _, _ = fixture(tmp_path)
    original = root / "old/result.json"
    outside = tmp_path / "outside.json"
    original.rename(outside)
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="demand_history_path_scope"):
        inspect_history(path, root)


def test_unknown_history_kind_cannot_be_ignored(tmp_path):
    path, root, data, _ = fixture(tmp_path)
    data["result"]["previous_attempts"][0]["kind"] = "new_unrecognized_kind"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="demand_history_kind"):
        inspect_history(path, root)


@pytest.mark.parametrize(
    "failure_kind", ["dispatched_unknown", "confirmed_response_or_domain_refusal"]
)
def test_routed_source_failure_is_bound_but_not_a_visual_attempt(tmp_path, failure_kind):
    path, root, data, _ = fixture(tmp_path)
    source = root / "source"
    source.mkdir()
    from db.runtime.repository.case_pipeline import identity

    data["task"]["checksum"] = digest(data["task"])
    protocol = dict(input=digest("input"), task=data["task"]["checksum"])
    failure = dict(
        stage="definitions",
        campaign_id=source.name,
        run_id=identity(
            "RUN", "task-research:mm-live:" + digest((source.name, "definition-expansion"))[7:55]
        ),
        protocol_checksum=digest(protocol),
        input_checksum=protocol["input"],
        new_requests=1,
        failure_kind=failure_kind,
    )
    (source / "protocol.json").write_text(json.dumps(protocol))
    (source / "stage-failure.json").write_text(json.dumps(failure))
    data["result"]["previous_attempts"].append(
        dict(
            directory=str(source),
            checksum=digest(failure),
            requests=1,
            kind="routed_source_failure_deferred",
        )
    )
    path.write_text(json.dumps(data))
    result = inspect_history(path, root)
    assert len(result["history_files"]) == 2
    assert sum(len(row["historical_attempts"]) for row in result["rows"]) == 1
    assert result["source_failures"][0]["failure_kind"] == failure_kind
    assert result["source_failures"][0]["current_execution_authorization"] is False
    assert result["database_writes"] == result["new_api_requests"] == 0
    for field, changed in [("task", digest("other-target")), ("input", digest("other-input"))]:
        bad = {**protocol, field: changed}
        (source / "protocol.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="demand_history_source_binding"):
            inspect_history(path, root)
    (source / "protocol.json").write_text(json.dumps(protocol))
    failure["run_id"] = "RUN:OTHER"
    (source / "stage-failure.json").write_text(json.dumps(failure))
    data["result"]["previous_attempts"][-1]["checksum"] = digest(failure)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="demand_history_source_binding"):
        inspect_history(path, root)
