"""P3 live rejection closure; all Knowledge objects are synthetic on the owned PG lease."""

import json

import pytest

from tests.deterministic_pipeline.test_demand_resolution import counts, unmapped
from tests.deterministic_pipeline.test_pipeline import invoke


@pytest.fixture
def fresh_pipeline(database, tmp_path_factory, tmp_path, closure, request):
    # Independent Published release per destructive test; never withdraw a shared fixture.
    from tests.deterministic_pipeline.conftest import assembled_pipeline, shared_mapped

    mapped = shared_mapped.__wrapped__(database, tmp_path_factory)
    return unmapped(assembled_pipeline.__wrapped__(database, mapped, tmp_path, closure, request))


def test_unresolved_release_withdrawn_after_run_creation_returns_safe_report(
    monkeypatch, fresh_pipeline
):
    p = fresh_pipeline
    original = p.service._planning
    withdrawn = False
    original_facts = {}

    def revoke_after_planning(state):
        nonlocal withdrawn
        original(state)
        if not withdrawn:
            original_facts.update(
                state=state,
                lock=p.runtime.get_knowledge_lock(state.run_id),
                policies=p.runtime.get_run_policy_references(state.run_id),
            )
            p.database.publisher.withdraw(
                p.data["pack"].bindings.release_ref, "RULE:SYNTHETIC_WITHDRAWAL"
            )
            withdrawn = True

    monkeypatch.setattr(p.service, "_planning", revoke_after_planning)
    try:
        report = invoke(p)
    except Exception as error:
        state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
        print(
            "OBSERVED",
            type(error).__name__,
            "report_saved",
            state.report is not None,
            "run_status",
            p.runtime.run_status(state.run_id),
            "actions",
            len(state.actions),
        )
        raise
    assert withdrawn
    assert report["stop_reason"] == "knowledge_not_admitted"
    assert report["run_id"] and report["actions"] == 0
    assert report["observations"] == []
    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    assert state.report is not None

    assert_closed(p, report, "unresolved_admission", "knowledge_unavailable")
    assert state.demand_resolution == original_facts["state"].demand_resolution
    assert state.started_at == original_facts["state"].started_at
    assert p.runtime.get_knowledge_lock(state.run_id) == original_facts["lock"]
    assert p.runtime.get_run_policy_references(state.run_id) == original_facts["policies"]
    print(
        "REJECTION_ACCEPTANCE_JSON:"
        + json.dumps(
            {
                "report": report,
                "audit": [e.model_dump(mode="json") for e in case_audit(p)],
                "state": state.model_dump(mode="json"),
                "knowledge_lock": original_facts["lock"],
                "counts": counts(p),
                "replay_matches": True,
            },
            sort_keys=True,
        )
    )


def case_audit(p):
    return tuple(
        e
        for e in p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
        if e.case_code == p.case.case_code
    )


def assert_closed(p, report, phase, reason, *, has_run=True):
    from tests.database.knowledge_support_04a import sql

    assert report["stop_reason"] == "knowledge_not_admitted"
    assert report["conclusion_status"] == "insufficient"
    assert report["actions"] == 0 and report["observations"] == []
    assert report["components"] == dict.fromkeys(
        ("support", "refutation", "completeness", "conflict")
    )
    rejection = report["knowledge_rejections"][-1]
    assert rejection["phase"] == phase and rejection["reason"] == reason
    assert rejection["pack_ref"] == p.data["pack"].reference().model_dump(mode="json")
    assert rejection["release_ref"] == p.data["pack"].bindings.release_ref.model_dump(mode="json")
    assert rejection["requirement_refs"]
    assert reason in report["gaps"]
    key = p.runtime.key(p.case.case_code, "complete")
    state = p.runtime.read_request(key)
    assert p.service._read(state.report) == report
    trace = p.service.trace(p.case.case_code, "complete")
    assert rejection in trace
    assert not any(e["kind"] in ("plan", "execution", "validation") for e in trace)
    audit = case_audit(p)
    decision = [e for e in audit if e.event_type.value == "RULE_FINAL_DECISION"]
    assert len(decision) == 1
    assert decision[0].payload["report"] == state.report.model_dump(mode="json")
    assert not any(e.event_type.value == "GRAPH_VERSION_APPENDED" for e in audit)
    if has_run:
        assert counts(p)[:4] == (1, 1, 1, 0)
        assert report["graph"]["graph_version"] == 1
        assert p.runtime.run_status(state.run_id) == "COMPLETED_INSUFFICIENT_EVIDENCE"
        terminal = [e for e in audit if e.event_type.value == "RUN_STATUS_CHANGED"][-1]
        assert terminal.payload["to_status"] == "COMPLETED_INSUFFICIENT_EVIDENCE"
        assert p.service.replay(state.run_id).model_dump(mode="json") == report["graph"]
        assert (
            sql(
                p.database.admin,
                "SELECT count(*) FROM runtime.tool_attempt WHERE run_id=%s",
                (state.run_id,),
            )[0][0]
            == 0
        )
    else:
        assert counts(p)[:4] == (0, 0, 0, 0)
        assert report["run_id"] is report["graph"] is state.demand_resolution is None
    before = counts(p)
    assert invoke(p) == report
    assert p.runtime.read_request(key) == state
    assert counts(p) == before and case_audit(p) == audit


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    from packages.deterministic_pipeline.service import DeterministicCasePipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("unresolved branch must not validate or execute")

    monkeypatch.setattr(DeterministicCasePipeline, "_recover_actions", forbidden)
    monkeypatch.setattr(DeterministicCasePipeline, "_validate_results", forbidden)


def test_real_release_withdrawal_at_rule_validation_window(monkeypatch, fresh_pipeline):
    p = fresh_pipeline
    original = p.service.disease._published.validate_rule_package_for_run
    calls = 0

    def withdraw(*args, **kwargs):
        nonlocal calls
        calls += 1
        p.database.publisher.withdraw(
            p.data["pack"].bindings.release_ref, "RULE:SYNTHETIC_WITHDRAWAL"
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(p.service.disease._published, "validate_rule_package_for_run", withdraw)
    report = invoke(p)
    assert calls == 1
    assert_closed(p, report, "unresolved_propagation", "knowledge_unavailable")
    assert report["demand_propagations"] == []


def test_real_requirement_withdrawal_during_resolver_reread(monkeypatch, fresh_pipeline):
    p = fresh_pipeline
    reader = p.service.demand_resolver.knowledge

    class WithdrawingReader:
        def get_published_evidence(self, ref, **kwargs):
            p.database.publisher.withdraw(ref, "RULE:SYNTHETIC_WITHDRAWAL")
            return reader.get_published_evidence(ref, **kwargs)

    p.service.demand_resolver.knowledge = WithdrawingReader()
    report = invoke(p)
    assert_closed(p, report, "demand_resolution", "knowledge_unavailable", has_run=False)


@pytest.mark.parametrize(
    "phase,mode",
    [
        ("unresolved_admission", "gap"),
        ("unresolved_rule_package", "binding"),
        ("unresolved_propagation", "conflict"),
        ("unresolved_propagation", "gap"),
        ("demand_resolution", "binding"),
        ("final_admission", "unavailable"),
        ("final_rule_package", "binding"),
        ("final_propagation", "unavailable"),
        ("final_propagation", "gap"),
    ],
)
def test_known_refusal_windows(monkeypatch, assembled_pipeline, phase, mode):
    from dataclasses import replace

    import packages.deterministic_pipeline.service as module
    from db.knowledge.repository import KnowledgeConflictError, KnowledgeNotFoundError
    from packages.rule_engine import KnowledgeGap

    p = unmapped(assembled_pipeline)
    admit = p.service._admit
    admitted = None
    calls = 0

    def capture(i, c, state):
        nonlocal admitted
        admitted = admit(i, c, state)
        return admitted

    monkeypatch.setattr(p.service, "_admit", capture)

    def refuse(*args, **kwargs):
        nonlocal calls
        calls += 1
        if mode == "gap":
            gap = replace(
                admitted,
                template=None,
                knowledge_gap=KnowledgeGap("KNO04C_NOT_ADMITTED", "synthetic current refusal", ()),
            )
            return (gap, None) if phase.endswith("propagation") else gap
        if mode == "unavailable":
            raise KnowledgeNotFoundError("private database detail")
        if mode == "conflict":
            raise KnowledgeConflictError("private database detail")
        raise ValueError(
            "demand_requirement_version_conflict"
            if phase == "demand_resolution"
            else "graph does not originate from the exact reviewed template"
        )

    def install():
        if phase.endswith("admission"):
            monkeypatch.setattr(p.service, "_admit", refuse)
        elif phase.endswith("rule_package"):
            monkeypatch.setattr(module, "build_rule_package", refuse)
        elif phase.endswith("propagation"):
            monkeypatch.setattr(p.service.disease, "propagate_for_run", refuse)
        else:
            monkeypatch.setattr(p.service.demand_resolver, "resolve", refuse)

    if phase == "demand_resolution":
        install()
    elif phase.startswith("unresolved"):
        planning = p.service._planning

        def start(state):
            planning(state)
            install()

        monkeypatch.setattr(p.service, "_planning", start)
    else:
        finish = p.service._finish

        def end(*args, **kwargs):
            install()
            return finish(*args, **kwargs)

        monkeypatch.setattr(p.service, "_finish", end)
    report = invoke(p)
    assert calls == 1  # No admission retry loop after a recognized refusal.
    reason = {"gap": "KNO04C_NOT_ADMITTED", "unavailable": "knowledge_unavailable"}.get(
        mode, "knowledge_version_binding_rejected"
    )
    assert_closed(p, report, phase, reason, has_run=phase != "demand_resolution")
    assert "private database detail" not in str(report)
    if mode == "gap":
        assert report["knowledge_gap_details"][0]["code"] == "KNO04C_NOT_ADMITTED"


@pytest.mark.parametrize("phase", ["initial", "demand_resolution", "unresolved", "final"])
def test_unknown_rule_or_source_failure_is_not_a_knowledge_gap(
    monkeypatch, assembled_pipeline, phase
):
    from packages.retrieval.knowledge.models import RetrievalInfrastructureError

    p = unmapped(assembled_pipeline)
    error = (
        RetrievalInfrastructureError("source unavailable")
        if phase == "demand_resolution"
        else ValueError("rule implementation defect")
    )

    def fail(*args, **kwargs):
        raise error

    if phase == "initial":
        monkeypatch.setattr(p.service, "_admit", fail)
    elif phase == "demand_resolution":
        monkeypatch.setattr(p.service.demand_resolver, "resolve", fail)
    elif phase == "unresolved":
        monkeypatch.setattr(p.service.disease, "propagate_for_run", fail)
    else:
        finish = p.service._finish

        def end(*args, **kwargs):
            monkeypatch.setattr(p.service.disease, "propagate_for_run", fail)
            return finish(*args, **kwargs)

        monkeypatch.setattr(p.service, "_finish", end)
    with pytest.raises(type(error)) as raised:
        invoke(p)
    assert raised.value is error
    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    assert state.report is None
    if phase in ("initial", "demand_resolution"):
        assert counts(p)[:4] == (0, 0, 0, 0)


def test_permission_integrity_and_unknown_errors_keep_their_identity():
    from db.knowledge.repository import KnowledgePermissionError, KnowledgeRepositoryError
    from packages.deterministic_pipeline.knowledge_rejection import knowledge_read
    from packages.retrieval.knowledge.models import (
        RetrievalInfrastructureError,
        RetrievalIntegrityError,
    )

    for error in (
        KnowledgePermissionError("principal unavailable"),
        KnowledgeRepositoryError("database failure"),
        RetrievalInfrastructureError("source unavailable"),
        RetrievalIntegrityError("manifest checksum mismatch"),
        ValueError("unknown rule defect"),
        RuntimeError("unknown runtime defect"),
        OSError("object acknowledgement lost"),
    ):

        def fail(error=error):
            raise error

        with pytest.raises(type(error)) as raised:
            knowledge_read("final_propagation", fail)
        assert raised.value is error


@pytest.mark.parametrize("fault", ["report_ack", "terminal_conflict", "terminal_ack"])
def test_rejection_commit_failures_preserve_recovery(monkeypatch, assembled_pipeline, fault):
    from psycopg.errors import UniqueViolation

    from db.knowledge.repository import KnowledgeNotFoundError

    p = unmapped(assembled_pipeline)
    admit = p.service._admit

    def reject_running(i, c, state):
        if state.run_id:
            raise KnowledgeNotFoundError("synthetic live rejection")
        return admit(i, c, state)

    monkeypatch.setattr(p.service, "_admit", reject_running)
    with monkeypatch.context() as failing:
        if fault == "report_ack":
            put = p.service._put

            def lost_report(value):
                ref = put(value)
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == "deterministic-case-report/2.0.0"
                ):
                    raise OSError("report promotion acknowledgement lost")
                return ref

            failing.setattr(p.service, "_put", lost_report)
        elif fault == "terminal_conflict":
            audit = p.runtime._audit

            def conflict(connection, key, state, event_type, payload):
                if event_type == "RULE_FINAL_DECISION":
                    state = state.model_copy(update={"revision": 0})
                return audit(connection, key, state, event_type, payload)

            failing.setattr(p.runtime, "_audit", conflict)
        else:
            finish = p.runtime.finish

            def lost_commit(*args, **kwargs):
                finish(*args, **kwargs)
                raise OSError("terminal transaction acknowledgement lost")

            failing.setattr(p.runtime, "finish", lost_commit)
        with pytest.raises(UniqueViolation if fault == "terminal_conflict" else OSError):
            invoke(p)
    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    before = counts(p)
    if fault == "terminal_ack":
        assert state.report is not None
        assert p.runtime.run_status(state.run_id) == "COMPLETED_INSUFFICIENT_EVIDENCE"
    else:
        assert state.report is None
        assert p.runtime.run_status(state.run_id) in ("PLANNING", "VERIFYING")
        assert not any(e.event_type.value == "RULE_FINAL_DECISION" for e in case_audit(p))
    report = invoke(p)
    assert counts(p)[:4] == before[:4]
    after = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    assert after.demand_resolution == state.demand_resolution
    assert after.started_at == state.started_at
    assert_closed(
        p,
        report,
        "unresolved_admission" if fault == "terminal_ack" else "initial_admission",
        "knowledge_unavailable",
    )
