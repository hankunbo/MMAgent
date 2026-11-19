"""Thin recovery argv/state supervision; the original CLI retains authorization."""

from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as batch
from packages.grounding.models import canonical, validate
from packages.object_store import checksum
from tests.evaluation_preparation.test_terminal_cohort import fixture, manifest


def failed_case(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    m = validate(batch.Manifest, manifest())
    case = m.cases[0]
    for action, code in (("prepare", 0), ("preflight", 0), ("run", 2)):
        path = out / "commands" / case.id / action
        path.mkdir(parents=True)
        raw = b"synthetic command receipt, not a real case"
        (path / "stdout.log").write_bytes(raw)
        (path / "result.json").write_bytes(
            canonical(
                dict(
                    status="command_exited",
                    exit_code=code,
                    command=batch.commands(m, case, out, action)[0],
                    log_checksum=checksum(raw),
                )
            )
        )
    argv = batch.commands(m, case, out, "prepare")[0]
    parent = Path(argv[argv.index("--output") + 1])
    parent.mkdir()
    failure = dict(
        stage="definitions",
        error_code="schema",
        failure_kind="confirmed_response_or_domain_refusal",
        transport=dict(remote_response_unknown=0, dispatch_unknown=0),
    )
    (parent / "stage-failure.json").write_bytes(canonical(failure))
    return out, m, case, parent, failure


def test_recovery_argv_reuses_exact_config_and_parent_not_new_budget(tmp_path, monkeypatch):
    out, m, case, parent, _ = failed_case(tmp_path, monkeypatch)
    assert batch.source_recovery_ready(out, m, case)
    argv = batch.commands(m, case, out, "source-prepare")[0]
    assert argv[argv.index("--definition-failure-of") + 1] == str(parent)
    assert "--defer-failed-definition-expansion" in argv
    assert argv[argv.index("--active-acquisition") + 1] == case.active_acquisition
    child = argv[argv.index("--output") + 1]
    assert child != str(parent) and len(Path(child).name) <= 64
    assert "--seconds" not in argv and "--max-requests" not in argv
    for action, verb in (
        ("source-preflight", "preflight"),
        ("source-run", "run"),
        ("source-export", "restore-report"),
    ):
        assert batch.commands(m, case, out, action)[0][-3:] == [verb, "--input", child]


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "timeout", "successful", "missing_exit", "changed_log", "cancelled", "wrong_stage"],
)
def test_nonconfirmed_or_unbound_failure_cannot_request_successor(tmp_path, monkeypatch, mutation):
    out, m, case, parent, failure = failed_case(tmp_path, monkeypatch)
    path = out / "commands" / case.id / "run" / "result.json"
    receipt = batch.read(path)
    if mutation == "unknown":
        failure["transport"]["remote_response_unknown"] = 1
    elif mutation == "wrong_stage":
        failure["stage"] = "acquisition"
    elif mutation == "cancelled":
        (parent / "cancel").write_bytes(b"")
    elif mutation == "changed_log":
        (path.parent / "stdout.log").write_bytes(b"different")
    elif mutation == "timeout":
        receipt["status"] = "timeout_remote_state_not_inferred"
    elif mutation == "successful":
        receipt["exit_code"] = 0
    else:
        receipt["exit_code"] = None
    path.write_bytes(canonical(receipt))
    (parent / "stage-failure.json").write_bytes(canonical(failure))
    assert not batch.source_recovery_ready(out, m, case)
    monkeypatch.setattr(batch, "operation", lambda *a, **k: pytest.fail("dispatch"))
    result = batch.run(out, "recover-source", [case.id])
    assert result[0]["missing"] == ["original_confirmed_source_failure"]


def test_recovery_sequence_stops_at_failed_owner_and_never_changes_parent(tmp_path, monkeypatch):
    out, _, case, parent, _ = failed_case(tmp_path, monkeypatch)
    before = (parent / "stage-failure.json").read_bytes()
    calls = []

    def action(output, manifest, current, name):
        calls.append(name)
        return [dict(exit_code=2 if name == "source-preflight" else 0)]

    monkeypatch.setattr(batch, "case_action", action)
    result = batch.run(out, "recover-source", [case.id])
    assert calls == ["source-prepare", "source-preflight"] and result[-1]["exit_code"] == 2
    assert (parent / "stage-failure.json").read_bytes() == before


def test_recovery_progress_is_local_and_separate_from_original_failure(tmp_path, monkeypatch):
    out, _, case, _, _ = failed_case(tmp_path, monkeypatch)
    result = batch.status(out)
    row = result["cases"][0]
    assert row["commands"]["run"]["exit_code"] == 2
    assert row["recovery_progress"]["status"] == "not_started"
    assert row["commands"]["source-run"]["status"] == "not_started"
    assert "Source后继镜像" in Path(batch.export_status(out)["html"]).read_text()


def test_new_source_feedback_profiles_are_explicit_and_keep_limits():
    from packages.multimodal_research.active_requirements import load

    for old, new in (
        ("multimodal_structural_coverage_v2.json", "multimodal_structural_coverage_v3.json"),
        (
            "multimodal_harvard_visual_priority_v4.json",
            "multimodal_harvard_visual_priority_v5.json",
        ),
        (
            "multimodal_grape_active_requirements_v5.json",
            "multimodal_grape_active_requirements_v6.json",
        ),
    ):
        a, b = [load("configs/research/" + p) for p in (old, new)]
        assert b.pop("id") != a.pop("id")
        assert b["definition_expansion"]["prompt_version"] == "definition-draft-6.5.0"
        for key in ("prompt_version", "schema_version"):
            b["definition_expansion"][key] = a["definition_expansion"][key]
        assert a == b
