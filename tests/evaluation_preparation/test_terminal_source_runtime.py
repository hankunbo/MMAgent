"""Explicit repair supervision preserves old cohort and calls original CLI owners."""

import json
from contextlib import contextmanager

import pytest

from packages.evaluation_preparation import terminal_cohort as terminal
from packages.evaluation_preparation import terminal_source_recovery as recovery
from packages.grounding.models import canonical
from packages.object_store import checksum
from tests.evaluation_preparation.test_terminal_local_provider import manifest


def fixture(tmp_path, monkeypatch):
    cohort = tmp_path / "original"
    cohort.mkdir()
    m = manifest()
    case = next(c for c in m.cases if c.dataset != "fairvlmed")
    current = {"unchanged.py": "sha256:" + "1" * 64}
    runtime = {"runtime.py": "sha256:" + "2" * 64}
    freeze = dict(implementation=dict(current), configurations={}, runtime_owner=runtime)
    monkeypatch.setattr(terminal, "load", lambda *a, **k: (cohort, m, freeze))
    monkeypatch.setattr(terminal, "implementation", lambda: dict(current))
    monkeypatch.setattr(terminal, "runtime_owner_implementation", lambda: dict(runtime))
    argv = terminal.commands(m, case, cohort, "prepare")[0]
    from pathlib import Path

    parent = Path(argv[argv.index("--output") + 1])
    parent.mkdir(parents=True)
    failure = dict(
        stage="definitions",
        error_code="transport",
        failure_kind="dispatched_unknown",
        transport=dict(remote_response_unknown=1, dispatch_unknown=0),
    )
    for name, value in (("entry", {}), ("protocol", {}), ("stage-failure", failure)):
        (parent / (name + ".json")).write_bytes(canonical(value))
    for action, exit_code in (("prepare", 0), ("preflight", 0), ("run", 2)):
        folder = cohort / "commands" / case.id / action
        folder.mkdir(parents=True)
        raw = b"synthetic original command, no API"
        (folder / "stdout.log").write_bytes(raw)
        (folder / "result.json").write_bytes(
            canonical(
                dict(
                    status="command_exited",
                    exit_code=exit_code,
                    command=terminal.commands(m, case, cohort, action)[0],
                    log_checksum=checksum(raw),
                )
            )
        )
    held, calls = [], []

    @contextmanager
    def session(path):
        held.append(path)
        try:
            yield lambda: None
        finally:
            held.remove(path)

    monkeypatch.setattr(terminal, "runtime_supervision_session", session)

    def operation(out, selected, action, command, timeout, *, lease_guard):
        assert cohort in held and out in held
        lease_guard()
        calls.append(command)
        result = dict(status="command_exited", exit_code=0, command=command)
        p = out / "commands" / selected.id / action
        p.mkdir(parents=True)
        (p / "result.json").write_bytes(canonical(result))
        return result

    monkeypatch.setattr(terminal, "operation", operation)
    return cohort, m, case, parent, current, runtime, calls


def test_supervisor_uses_original_cli_and_absolute_window(tmp_path, monkeypatch):
    cohort, m, case, parent, _, _, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "recovery"
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    for phase in ("prepare", "preflight", "run"):
        assert recovery.run(cohort, case.id, out, phase)["exit_code"] == 0
    argv = calls[0]
    assert argv[argv.index("--definition-failure-of") + 1] == str(parent)
    assert (
        argv[argv.index("--entry-execution-lock") + 1] == "multimodal-runtime-source-backlog/1.0.0"
    )
    assert "--defer-failed-definition-expansion" in argv
    assert "--retain-closed-public-wire-failures" in argv
    saved = json.loads((out / "source-recovery.json").read_bytes())
    assert saved["schema_version"] == "terminal-source-runtime-recovery/1.1.0"
    assert not {"--source-fallback-seconds", "--seconds", "--max-requests", "--retry"} & set(argv)
    assert all(a[2:4] == ["-m", "apps.multimodal_case"] for a in calls)
    assert before == {p.name: p.read_bytes() for p in parent.iterdir()}
    assert not terminal.source_recovery_ready(cohort, m, case)
    with pytest.raises(ValueError, match="source_successor_not_supported"):
        terminal.commands(m, case, cohort, "source-prepare")


@pytest.mark.parametrize(
    "change", ["unrelated", "runtime", "unknown", "cancel", "log", "prerequisite"]
)
def test_invalid_parent_or_delta_never_dispatches(tmp_path, monkeypatch, change):
    cohort, _, case, parent, current, runtime, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "recovery"
    if change == "unrelated":
        current["unrelated.py"] = "changed"
    elif change == "runtime":
        monkeypatch.setattr(terminal, "runtime_owner_implementation", lambda: {})
    elif change == "unknown":
        p = parent / "stage-failure.json"
        value = json.loads(p.read_bytes())
        value["transport"]["remote_response_unknown"] = 2
        p.write_bytes(canonical(value))
    elif change == "cancel":
        (parent / "cancel").touch()
    elif change == "log":
        (cohort / "commands" / case.id / "prepare/stdout.log").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source_runtime"):
        recovery.run(cohort, case.id, out, "run" if change == "prerequisite" else "prepare")
    assert not calls


def test_recovery_freeze_cannot_be_replaced(tmp_path, monkeypatch):
    cohort, _, case, _, current, _, calls = fixture(tmp_path, monkeypatch)
    out = tmp_path / "recovery"
    recovery.run(cohort, case.id, out, "prepare")
    current["apps/case_cohort.py"] = "changed_after_prepare"
    with pytest.raises(ValueError, match="freeze_conflict"):
        recovery.run(cohort, case.id, out, "preflight")
    assert len(calls) == 1
