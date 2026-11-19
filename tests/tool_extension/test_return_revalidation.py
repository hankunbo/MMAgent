"""Late invalidation after genuine admission returns at the controlled commit boundary."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from packages.tool_extension import ExtensionAuthority
from tests.tool_adapter.test_postgres import adapter, host, scenario, snapshots
from tests.tool_adapter.test_postgres import consume as adapter_consume
from tests.tool_extension.support import assemble, consume
from tests.tool_extension.test_contracts import setup
from tests.tool_extension.test_postgres import (
    extension,
    install_future,
    test_audit_rollback_reserved_and_terminal_cancel,
    test_lost_reply_and_revocation_same_key_read_only,
)

__all__ = [
    "adapter",
    "host",
    "scenario",
    "extension",
    "test_audit_rollback_reserved_and_terminal_cancel",
    "test_lost_reply_and_revocation_same_key_read_only",
]


def export(name, value):
    folder = os.environ.get("MMAGENT_11D_REPAIR_EVIDENCE")
    if folder:
        (Path(folder) / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "grant",
        "catalog",
        "source",
        "binding",
        "technical",
        "version",
        "expiry",
        "tail_grant",
        "tail_catalog",
    ],
)
def test_return_boundary(host, adapter, tmp_path, monkeypatch, change):
    adapter_consume(adapter)
    owner, future = install_future(host, adapter, tmp_path)
    e = assemble(adapter, catalog=owner, future=future)
    commit, admit, recheck = (
        e.repository._commit,
        future.admission.admit_research,
        adapter.g.recheck,
    )
    in_commit = False
    calls = []
    tail_calls = []
    origin_calls = []

    def committing(*args):
        nonlocal in_commit
        in_commit = True
        try:
            return commit(*args)
        finally:
            in_commit = False

    def revoke():
        e.authority.revoke(adapter.a.run_id, adapter.grant.task_id)

    def admission(request):
        result = admit(request)
        calls.append(dict(commit=in_commit, status=result.status))
        if in_commit:
            if change == "grant":
                revoke()
            elif change == "catalog":
                owner.reader.disable(owner.fixture.implementation_ref)
            elif change == "source":
                adapter.d.sources[0]._identities = ()
            elif change == "binding":
                future.run_id = adapter.a.run_id
            elif change == "technical":
                adapter.d.sources[0].technical_failure = True
            elif change == "version":
                adapter.c.manifest = adapter.c.manifest.model_copy(update=dict(version="2.0.0"))
            elif change == "expiry":
                g = e.authority.current(
                    adapter.a.run_id, adapter.grant.task_id, "extension-1", "offline-d"
                )
                e.authority = ExtensionAuthority(
                    [g.model_copy(update=dict(expires_at=datetime.now(UTC) - timedelta(seconds=1)))]
                )
        return result

    def original_owner_return(application):
        result = recheck(application)
        origin_calls.append(in_commit)
        # The final origin recheck itself calls 06. Invalidation here must also be
        # caught; appending a prepare that checks authority only FIRST is insufficient.
        if in_commit and calls and calls[-1]["commit"]:
            tail_calls.append("after_genuine_origin_recheck")
            if change == "tail_grant":
                revoke()
            elif change == "tail_catalog":
                owner.reader.disable(owner.fixture.implementation_ref)
        return result

    monkeypatch.setattr(e.repository, "_commit", committing)
    monkeypatch.setattr(future.admission, "admit_research", admission)
    monkeypatch.setattr(adapter.g, "recheck", original_owner_return)
    before = snapshots(host)
    error = None
    try:
        # Exercise both installed public entries, with identical ownership rules.
        entry = e.repository if change in {"catalog", "binding", "version"} else e
        consume(entry, adapter)
    except (ValueError, OSError) as exc:
        error = exc
    saved = e.read(adapter.a.run_id, adapter.a.key)
    after = snapshots(host)
    export(
        "observed-" + change,
        dict(
            synthetic=True,
            history_only=True,
            change=change,
            calls=calls,
            tail_calls=tail_calls,
            error=None if error is None else dict(type=type(error).__name__, message=str(error)),
            saved=saved,
            audit_delta=after["audit"] - before["audit"],
            original_run_unchanged=after["run"] == before["run"],
            tool_attempts=after["attempts"],
        ),
    )
    assert calls == [
        dict(commit=False, status="preparation_only"),
        dict(commit=True, status="preparation_only"),
    ]
    assert {k: v for k, v in before.items() if k != "audit"} == {
        k: v for k, v in after.items() if k != "audit"
    }
    assert after["attempts"] == 0 and saved["budget_charged"] == 1
    if change == "none":
        assert error is None and saved["status"] == "saved"
        assert after["audit"] == before["audit"] + 2
        assert saved["result"]["validation_status"] == "unvalidated"
        assert not saved["result"]["available_next_run"]
        assert origin_calls == [False, False, True, True]
        revoke()
        owner.reader.disable(owner.fixture.implementation_ref)
    else:
        assert saved["status"] == "reserved" and saved["result"] is None
        assert after["audit"] == before["audit"] + 1
        expected = {
            "grant": "extension_current_grant_required",
            "catalog": "extension_catalog_unavailable_or_terminal",
            "source": "extension_candidate_changed",
            "binding": "extension_explicit_new_run_required",
            "technical": "synthetic_candidate_source_unavailable",
            "version": "extension_current_contract_changed",
            "expiry": "extension_grant_expired",
            "tail_grant": "extension_current_grant_required",
            "tail_catalog": "extension_catalog_unavailable_or_terminal",
        }[change]
        assert type(error) is (OSError if change == "technical" else ValueError)
        assert str(error) == expected
        if change in {"tail_grant", "tail_catalog"}:
            assert tail_calls == ["after_genuine_origin_recheck"]
            assert origin_calls == [False, False, True, True]
        revoke()
    # Unknown commit / terminal history never starts new work or renews its lease.
    stable = snapshots(host)
    calls_before = list(calls)
    assert consume(e, adapter) == saved and snapshots(host) == stable and calls == calls_before
    with pytest.raises(ValueError, match="extension_stage_identity_conflict"):
        e.repository.consume(
            adapter.a.run_id, adapter.a.key, stage_key="escape", worker="offline-d"
        )
    if change != "none":
        cancelled = e.cancel(
            adapter.a.run_id, adapter.a.key, stage_key="extension-1", worker="offline-d"
        )
        assert cancelled["status"] == "cancelled" and cancelled["budget_charged"] == 1
        stable = snapshots(host)
        assert consume(e, adapter) == cancelled and snapshots(host) == stable
        assert calls == calls_before


@pytest.mark.parametrize("change", ["none", "catalog", "binding", "selection", "technical"])
def test_formal_return_boundary(tmp_path, monkeypatch, change):
    s = setup(tmp_path)
    original = s.future.admission.admit
    calls = []

    def admission(*args):
        result = original(*args)
        calls.append(result.status)
        if change == "catalog":
            s.owner.reader.disable(s.owner.fixture.implementation_ref)
        elif change == "binding":
            from tests.grounding.query_support import updated
            from tests.tool_admission.support import reseal

            s.f.q = updated(s.f.q, lock=reseal(s.f.q.lock, graph_version=2))
        elif change == "selection":
            s.f.selection = s.f.selection.model_copy(update=dict(candidate_refs=()))
        elif change == "technical":

            def unavailable(run_id):
                raise OSError("synthetic_future_owner_unavailable")

            s.future.selection_reader = unavailable
        return result

    monkeypatch.setattr(s.future.admission, "admit", admission)
    if change == "none":
        check = s.future.evaluate(s.plan)
        assert check["admission"]["status"] == "preparation_only"
        assert not check["available_next_run"] and check["execution"] == "not_executed"
    elif change == "technical":
        with pytest.raises(OSError, match="^synthetic_future_owner_unavailable$"):
            s.future.evaluate(s.plan)
    else:
        expected = (
            "extension_catalog_unavailable_or_terminal"
            if change == "catalog"
            else "extension_future_binding_changed"
        )
        with pytest.raises(ValueError, match="^" + expected + "$"):
            s.future.evaluate(s.plan)
    assert calls == ["preparation_only"]


def test_control_return_boundary_last(host, adapter, tmp_path, monkeypatch):
    from agents.planner.service import Planner
    from db import manage
    from db.runtime.repository.planner import PlannerRepository
    from packages.planner_control.models import Command
    from tests.database.knowledge_support_04a import sql
    from tests.planner.support import FakeOwner, binding

    adapter_consume(adapter)
    owner, future = install_future(host, adapter, tmp_path)
    e = assemble(adapter, catalog=owner, future=future)
    commit, admit = e.repository._commit, future.admission.admit_research
    in_commit = False
    calls = []

    def committing(*args):
        nonlocal in_commit
        in_commit = True
        try:
            return commit(*args)
        finally:
            in_commit = False

    def admission(request):
        result = admit(request)
        calls.append((in_commit, result.status))
        if in_commit:
            runtime = PlannerRepository(manage.test_target("runtime"))
            start, checksum = runtime.origin(adapter.a.run_id)
            b = binding(
                case_code=adapter.a.case_code,
                run_id=adapter.a.run_id,
                run_context_checksum=checksum,
                started_at=start,
                deadline=start + timedelta(seconds=300),
            )
            planner = Planner(runtime=runtime, owner=FakeOwner(b))
            state = planner.begin(b)
            planner.step(
                adapter.a.run_id,
                Command(key="cancel-11d-return", expected_revision=state.revision, action="cancel"),
            )
        return result

    monkeypatch.setattr(e.repository, "_commit", committing)
    monkeypatch.setattr(future.admission, "admit_research", admission)
    before = snapshots(host)
    with pytest.raises(ValueError, match="^extension_current_plan_changed$"):
        consume(e.repository, adapter)
    saved = e.read(adapter.a.run_id, adapter.a.key)
    assert saved["status"] == "reserved" and saved["result"] is None
    assert saved["budget_charged"] == 1
    events = sql(
        host.db.admin,
        "SELECT payload->>'action' FROM audit.event WHERE task_id=%s "
        "AND correlation_code='SOURCE:SYNTHETIC_TOOL_EXTENSION'",
        (adapter.grant.task_id,),
    )
    assert events == [("reserve",)]
    assert calls == [(False, "preparation_only"), (True, "preparation_only")]
    assert snapshots(host)["run"] == before["run"] and snapshots(host)["attempts"] == 0
    assert consume(e, adapter) == saved
    assert (
        e.cancel(adapter.a.run_id, adapter.a.key, stage_key="extension-1", worker="offline-d")[
            "status"
        ]
        == "cancelled"
    )
    export(
        "observed-control",
        dict(
            synthetic=True,
            history_only=True,
            saved=saved,
            calls=calls,
            d_audit_actions=events,
            original_run_unchanged=True,
            tool_attempts=0,
        ),
    )
