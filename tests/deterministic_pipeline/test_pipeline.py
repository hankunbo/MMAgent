import pytest

from db.runtime.repository.ports import IdempotencyConflictError
from packages.deterministic_pipeline.models import PipelineError


def invoke(p, key="complete"):
    return p.service.run_synthetic(
        p.case, pack_ref=p.data["pack"].reference(), config_checksum=p.config.checksum, key=key
    )


def test_complete_synthetic_actual_services_and_recovery(assembled_pipeline):
    p = assembled_pipeline
    report = invoke(p)
    assert report["stop_reason"] == "graph_supported", (
        report["gaps"],
        report["observations"],
        [
            p.service._read(r)
            for r in p.runtime.read_request(p.runtime.key(p.case.case_code, "complete")).trace
            if p.service._read(r)["kind"] == "tool_rejected"
        ],
    )
    assert report["synthetic"] and not report["real_medical_graph_complete"]
    assert report["observations"][0]["qc_status"] == "PASS"
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]
    assert invoke(p) == report
    assert len(list((p.objects.root.parent / "attempts").glob("*/*/process.json"))) == 1


def test_input_conflict_not_patient_hash_dedup(assembled_pipeline):
    p = assembled_pipeline
    invoke(p)
    changed = p.case.model_copy(update={"patient_hash": "sha256:" + "c" * 64})
    with pytest.raises(IdempotencyConflictError):
        p.service.run_synthetic(
            changed,
            pack_ref=p.data["pack"].reference(),
            config_checksum=p.config.checksum,
            key="complete",
        )


def test_real_entry_cannot_use_synthetic_service(assembled_pipeline):
    p = assembled_pipeline
    with pytest.raises(PipelineError, match="real_input_forbidden"):
        p.service.run(
            p.case.case_code, input_checksum="x", config_checksum=p.config.checksum, key="x"
        )


def test_partial_database_audit_and_confirmed_execution_recovery(assembled_pipeline, monkeypatch):
    """Fail the run transaction, then lose an execution receipt; neither repeats work."""
    p = assembled_pipeline
    original = p.runtime._audit

    def fail_run(c, key, state, event_type, payload):
        if event_type == "CONFIG_VERSION_LOCKED":
            raise RuntimeError("injected Audit failure")
        return original(c, key, state, event_type, payload)

    monkeypatch.setattr(p.runtime, "_audit", fail_run)
    with pytest.raises(RuntimeError, match="Audit failure"):
        invoke(p)
    with p.runtime._connect() as c:
        assert (
            c.execute(
                "SELECT count(*) AS n FROM runtime.run WHERE case_code=%s", (p.case.case_code,)
            ).fetchone()["n"]
            == 0
        )
        assert (
            c.execute(
                "SELECT count(*) AS n FROM runtime.graph_snapshot WHERE case_code=%s",
                (p.case.case_code,),
            ).fetchone()["n"]
            == 0
        )
    monkeypatch.setattr(p.runtime, "_audit", original)
    trace = p.service._trace

    def lost(key, state, kind, **data):
        if kind == "execution":
            raise RuntimeError("injected receipt loss after confirmation")
        return trace(key, state, kind, **data)

    monkeypatch.setattr(p.service, "_trace", lost)
    with pytest.raises(RuntimeError, match="receipt loss"):
        invoke(p)
    monkeypatch.setattr(p.service, "_trace", trace)
    report = invoke(p)
    assert report["stop_reason"] == "graph_supported"
    assert len(list((p.objects.root.parent / "attempts").glob("*/*/process.json"))) == 1
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]


def test_report_object_failure_is_recoverable_without_new_execution(
    assembled_pipeline, monkeypatch
):
    p = assembled_pipeline
    put = p.service._put

    def fail(value):
        if (
            isinstance(value, dict)
            and value.get("schema_version") == "deterministic-case-report/2.0.0"
        ):
            put(value)
            raise OSError("injected report object promotion failure")
        return put(value)

    monkeypatch.setattr(p.service, "_put", fail)
    with pytest.raises(OSError, match="promotion failure"):
        invoke(p)
    monkeypatch.setattr(p.service, "_put", put)
    report = invoke(p)
    assert report["stop_reason"] == "graph_supported"
    assert len(list((p.objects.root.parent / "attempts").glob("*/*/process.json"))) == 1


@pytest.mark.parametrize("assembled_pipeline", ["fallback", "shared"], indirect=True)
def test_alternative_workflows_and_shared_and_product(assembled_pipeline, request):
    p = assembled_pipeline
    report = invoke(p)
    assert report["stop_reason"] == "graph_supported", report["gaps"]
    processes = list((p.objects.root.parent / "attempts").glob("*/*/process.json"))
    if request.node.callspec.params["assembled_pipeline"] == "fallback":
        assert len(processes) == 2
        assert [o["execution_status"] for o in report["observations"]] == ["FAILED", "SUCCEEDED"]
    else:
        assert len(processes) == 1
        assert len({o["target"] for o in report["observations"]}) == 2
        assert len({a["asset_code"] for o in report["observations"] for a in o["artifacts"]}) == 1
        assert all(o["clinical_effect"] == "SUPPORTS" for o in report["observations"])
        assert (
            len(
                {
                    o["evidence_validations"][0]["validations"][0]["validation_code"]
                    for o in report["observations"]
                }
            )
            == 2
        )
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]


@pytest.mark.parametrize("assembled_pipeline", ["nonzero", "badlabels"], indirect=True)
def test_technical_failure_is_not_medical_negative(assembled_pipeline):
    p = assembled_pipeline
    report = invoke(p)
    assert report["conclusion_status"] == "insufficient"
    assert all(o["clinical_effect"] == "NONE" for o in report["observations"])
    assert not any(n["status"] == "REFUTED" for n in report["graph"]["nodes"])
    assert report["observations"]


def test_unknown_reservation_is_never_automatically_reexecuted(assembled_pipeline, monkeypatch):
    p = assembled_pipeline
    get = p.tools.executor_for

    def uncertain(plan):
        executor = get(plan)
        reserve = executor._repo.reserve

        def lost(*args, **kwargs):
            reserve(*args, **kwargs)
            raise RuntimeError("injected reservation acknowledgement loss")

        monkeypatch.setattr(executor._repo, "reserve", lost)
        return executor

    monkeypatch.setattr(p.tools, "executor_for", uncertain)
    with pytest.raises(RuntimeError, match="acknowledgement loss"):
        invoke(p)
    monkeypatch.setattr(p.tools, "executor_for", get)
    from datetime import timedelta

    from tests.deterministic_pipeline.test_budget_attribution import freeze_case_clock

    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "complete"))
    freeze_case_clock(
        monkeypatch, state.started_at + timedelta(seconds=p.config.policy.max_seconds)
    )
    report = invoke(p)
    assert report["stop_reason"] == "execution_unknown_no_automatic_rerun"
    assert not list((p.objects.root.parent / "attempts").glob("*/*/process.json"))
    assert report["conclusion_status"] == "insufficient"


@pytest.mark.parametrize("assembled_pipeline", ["shared"], indirect=True)
def test_graph_conflict_recomputes_validation_on_actual_new_snapshot(
    assembled_pipeline, monkeypatch
):
    from db.runtime.repository.case_pipeline import identity
    from packages.graph_engine import bind_artifacts, canonical_checksum

    p = assembled_pipeline
    append = p.audit.append_graph_version
    injected = False

    def concurrent(command):
        nonlocal injected
        payload = command.audit_event.payload
        if not injected and payload.get("validation_checksum"):
            injected = True
            graph = p.runtime.current_graph(command.run_id)
            # Concurrent owner binds the same committed, reviewed product to the other
            # legitimate AND slot. No medical status is fabricated to force a conflict.
            artifact = next(
                a
                for a in p.runtime.artifacts(p.case.case_code, command.run_id)
                if a.scope.run_id == command.run_id
            )
            target = p.data["pack"].definition.evidence_slots[1].node_code
            updated, event = bind_artifacts(
                graph,
                ((target, artifact),),
                expected_checksum=canonical_checksum(graph),
                rule_ref=graph.provenance.rule_versions[0],
            )
            audit = command.audit_event.model_copy(
                update={
                    "event_id": identity("MESSAGE", event["event_checksum"]),
                    "idempotency_key": identity("TASK", event["event_checksum"]),
                    "payload": event,
                }
            )
            append(command.model_copy(update={"graph": updated, "audit_event": audit}))
        return append(command)

    monkeypatch.setattr(p.audit, "append_graph_version", concurrent)
    report = invoke(p)
    assert injected and report["stop_reason"] == "graph_supported"
    events = p.service.trace(p.case.case_code, "complete")
    snapshots = [e["validation"]["source_snapshot"] for e in events if e["kind"] == "validation"]
    assert len({s["graph_checksum"] for s in snapshots}) >= 2
    assert any(e["kind"] == "graph_conflict" for e in events)
    assert p.service.replay(report["run_id"]).model_dump(mode="json") == report["graph"]


def test_0010_terminal_status_and_report_are_atomic(assembled_pipeline, monkeypatch):
    """Initialize through the real owner, then test terminal transaction rollback."""
    from datetime import UTC, datetime

    import psycopg

    from db.audit.repository import AuditEvent, RunStatusChange
    from db.runtime.repository.case_pipeline import identity
    from packages.dataset_adapters.models import ObjectReference
    from packages.object_store import checksum

    p = assembled_pipeline
    runtime = p.runtime
    start_run = runtime.start_run

    def initialized(*args, **kwargs):
        start_run(*args, **kwargs)
        raise RuntimeError("synthetic stop after controlled run creation")

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "start_run", initialized)
        with pytest.raises(RuntimeError, match="stop after controlled run creation"):
            invoke(p, key="P2-finish")
    key = runtime.key(p.case.case_code, "P2-finish")
    state = runtime.read_request(key)
    assert state.run_id is not None and runtime.run_status(state.run_id) == "CREATED"
    assert p.tools.delegate is None
    data = b'{"synthetic_persistence_fixture":true}'
    uri = p.objects.promote(p.objects.stage(data), checksum(data))
    ref = ObjectReference(uri=uri, checksum=checksum(data), byte_size=len(data))

    def change(expected):
        event_key = "P2-transition-" + expected
        return RunStatusChange(
            run_id=state.run_id,
            expected_status=expected,
            new_status="FAILED_TECHNICAL",
            audit_event=AuditEvent(
                event_id=identity("MESSAGE", event_key),
                event_type="RUN_STATUS_CHANGED",
                actor="PLANNER",
                case_code=p.case.case_code,
                run_id=state.run_id,
                correlation_code="SOURCE:DETERMINISTIC_PIPELINE",
                idempotency_key=identity("TASK", event_key),
                occurred_at=datetime.now(UTC),
                payload={"from_status": expected, "to_status": "FAILED_TECHNICAL"},
            ),
        )

    report_event = identity("MESSAGE", key + ":" + str(state.revision + 1))
    with pytest.raises(psycopg.errors.SerializationFailure):
        runtime.finish(key, state, ref, change("PLANNING"))
    assert runtime.read_request(key) == state
    assert runtime.run_status(state.run_id) == "CREATED"
    assert p.audit_reader.get_event(report_event) is None
    assert p.audit_reader.get_event(identity("MESSAGE", "P2-transition-PLANNING")) is None
    completed = runtime.finish(key, state, ref, change("CREATED"))
    assert completed.report == ref and runtime.run_status(state.run_id) == "FAILED_TECHNICAL"
    assert p.audit_reader.get_event(report_event) is not None
    assert p.audit_reader.get_event(identity("MESSAGE", "P2-transition-CREATED")) is not None
