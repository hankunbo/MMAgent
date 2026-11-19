"""09A joins real isolated Runtime/Audit to harmless synthetic CPU observations."""

import pytest

from tests.deterministic_pipeline.test_budget_attribution import request_state
from tests.deterministic_pipeline.test_pipeline import invoke


def test_withdrawal_after_execution_rechecks_current_tool_source(assembled_pipeline, monkeypatch):
    p = assembled_pipeline
    original = p.service._trace

    def withdraw(key, state, kind, **data):
        value = original(key, state, kind, **data)
        if kind == "execution":

            def revoked(_):
                raise ValueError("tool_safety_revoked")

            monkeypatch.setattr(p.tools.ready.compiler, "validate_plan", revoked)
        return value

    monkeypatch.setattr(p.service, "_trace", withdraw)
    with pytest.raises(ValueError, match="tool_safety_revoked"):
        invoke(p, "09a-current-source")
    state = request_state(p, "09a-current-source")
    events = [p.service._read(r) for r in state.trace]
    assert len([e for e in events if e["kind"] == "execution"]) == 1
    assert not [e for e in events if e["kind"] == "validation"]
    assert_confirmed_once(p, state)


def test_known_source_rejection_stops_without_erasing_confirmed_facts(
    assembled_pipeline, monkeypatch
):
    from packages.tool_workflow import CompileError

    p = assembled_pipeline
    original = p.service._trace

    def withdraw(key, state, kind, **data):
        value = original(key, state, kind, **data)
        if kind == "execution":

            def revoked(_):
                raise CompileError("stale_or_modified_plan")

            monkeypatch.setattr(p.tools.ready.compiler, "validate_plan", revoked)
        return value

    monkeypatch.setattr(p.service, "_trace", withdraw)
    report = invoke(p, "09a-known-withdrawal")
    assert report["stop_reason"] == "tool_source_not_current"
    assert report["observations"] and all(
        o["clinical_effect"] == "NONE" for o in report["observations"]
    )
    state = request_state(p, "09a-known-withdrawal")
    assert not [
        p.service._read(r) for r in state.trace if p.service._read(r)["kind"] == "validation"
    ]
    assert invoke(p, "09a-known-withdrawal") == report
    assert_confirmed_once(p, state)


def test_tool_check_commit_loss_recovers_without_new_execution(assembled_pipeline, monkeypatch):
    p = assembled_pipeline
    original = p.service._trace
    lost = False

    def fail_once(key, state, kind, **data):
        nonlocal lost
        value = original(key, state, kind, **data)
        if kind == "tool_validation" and not lost:
            lost = True
            raise OSError("09a_check_ack_lost")
        return value

    monkeypatch.setattr(p.service, "_trace", fail_once)
    with pytest.raises(OSError, match="09a_check_ack_lost"):
        invoke(p, "09a-check-recovery")
    state = request_state(p, "09a-check-recovery")
    events = [p.service._read(r) for r in state.trace]
    checks = [e for e in events if e["kind"] == "tool_validation"]
    assert len(checks) == 1 and checks[0]["check"]["clinical_effect"] == "NONE"
    assert not [e for e in events if e["kind"] == "validation"]
    report = invoke(p, "09a-check-recovery")
    assert report["stop_reason"] == "graph_supported" and invoke(p, "09a-check-recovery") == report
    state = request_state(p, "09a-check-recovery")
    events = [p.service._read(r) for r in state.trace]
    assert [e for e in events if e["kind"] == "tool_validation"] == checks
    assert len([e for e in events if e["kind"] == "validation"]) == 1
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]
    assert_confirmed_once(p, state)


def assert_confirmed_once(p, state):
    import json

    from packages.tool_workflow import ExecutionPlan

    assert len(state.actions) == 1
    events = [p.service._read(r) for r in state.trace]
    execution = [e for e in events if e["kind"] == "execution"]
    assert len(execution) == 1
    plan = ExecutionPlan.from_bundle_json(
        json.dumps(next(e["bundle"] for e in events if e["kind"] == "plan"))
    )
    attempt = p.tools.executor_for(plan).recover(state.run_id, state.actions[0])
    assert attempt["status"] == "CONFIRMED" and attempt["result"] == execution[0]["workflow"]
    assert len(list((p.objects.root.parent / "attempts").glob("*/*/process.json"))) == 1
    audit = [
        e
        for e in p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
        if e.run_id == state.run_id
    ]
    assert audit and len({e.event_id for e in audit}) == len(audit)
