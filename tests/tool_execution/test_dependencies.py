"""DAG/QC process tests with a test ledger; PG atomicity is exercised separately."""

from dataclasses import replace

import pytest

from packages.contracts import QCStatus, require_transition
from packages.grounding.models import validate
from packages.tool_execution.models import WorkflowResult
from tests.tool_execution.support import prepared, runtime_files


class TestLedger:
    __test__ = False

    def __init__(self):
        self.attempts = {}
        self.transitions = []

    def reserve(self, document, steps):
        key = (document["run_id"], document["idempotency_key"])
        if key in self.attempts:
            previous = self.attempts[key]
            if previous["request_hash"] != document["request_hash"]:
                raise ValueError("idempotency_conflict")
            return dict(created=False, attempt=previous)
        record = {
            **document,
            "result": None,
            "status": "ACTIVE",
            "steps": [dict(step_id=s.step_id, status="REQUESTED", result=None) for s in steps],
        }
        self.attempts[key] = record
        return dict(created=True, attempt=record)

    def advance(self, attempt, step_id, expected, target, **kwargs):
        require_transition(expected, target)
        row = next(s for s in attempt["steps"] if s["step_id"] == step_id)
        assert row["status"] == expected.value
        row.update(status=target.value)
        if kwargs.get("record") is not None:
            row["result"] = kwargs["record"].model_dump(mode="json")
        row.update({k: v for k, v in kwargs.items() if k == "process_identity"})
        self.transitions.append((step_id, target.value))

    def confirm(self, workflow):
        workflow = validate(WorkflowResult, workflow)
        attempt = next(a for a in self.attempts.values() if a["attempt_id"] == workflow.attempt_id)
        attempt.update(result=workflow.model_dump(mode="json"), status="CONFIRMED")
        return workflow

    def recover(self, run_id, key):
        return self.attempts.get((run_id, key))


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    return runtime_files(tmp_path_factory.mktemp("06d-dag-runtime"))


@pytest.mark.parametrize("qc", ["PASS", "PENDING", "FAIL", "UNASSESSED"])
def test_dependency_only_consumes_actual_pass(tmp_path, closure, qc):
    ledger = TestLedger()
    p = prepared(tmp_path, chain=True, repository=ledger, files=closure)
    assert len(p.plan.document.steps) == 2
    original = p.qc.evaluate

    def evaluate(*args, **kwargs):
        return replace(original(*args, **kwargs), status=QCStatus(qc))

    if qc != "PASS":
        p.qc.evaluate = evaluate
    w = p.executor.execute(p.plan, "dag")
    if qc == "PASS":
        assert w.status == "SUCCEEDED", w.model_dump_json(indent=2)
        upstream = w.steps[0].result.artifacts[0]
        assert upstream.asset_code in w.steps[1].result.artifacts[0].derived_from
        assert upstream.checksum in w.steps[1].copied_checksums.values()
        assert upstream.checksum != w.steps[1].result.artifacts[0].checksum
        assert upstream.checksum not in {r.artifact.checksum for r in p.f.owners.records.values()}
        assert len({s.result.implementation_code for s in w.steps}) == 2
    else:
        assert w.status == "PARTIAL_FAILED"
        assert w.steps[1].error_code == "dependency_qc_not_satisfied"
        assert w.steps[1].result.artifacts == ()
    starts = [s for s, status in ledger.transitions if status == "RUNNING"]
    assert len(starts) == (2 if qc == "PASS" else 1)


def test_unconsumed_secondary_output_must_exist_and_be_qcd(tmp_path, closure):
    from tests.tool_workflow.test_extra_outputs import with_extra_output

    old = with_extra_output(tmp_path, "compatible")
    base = old.f
    # The helper's catalog reference is on the actual controlled Grounding reader.
    base.catalogs = base.grounding.mappings.catalogs
    # Drop old synthetic deployment snapshots from this *new* ephemeral test owner.
    base.owners.manifests = {
        k: v for k, v in base.owners.manifests.items() if v.catalog_ref == base.q.lock.catalog_ref
    }
    p = prepared(tmp_path, base=base, repository=TestLedger(), files=closure)
    assert len(p.plan.document.steps[0].outputs) == 2
    w = p.executor.execute(p.plan, "all-outputs")
    assert w.status == "FAILED"
    assert not w.steps[0].result.artifacts


def test_caller_cannot_supply_approval_or_readers(tmp_path, closure):
    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    with pytest.raises(TypeError):
        p.executor.execute(p.plan, "bad", approved=True)
    with pytest.raises(TypeError):
        p.executor.execute(p.plan, "bad", reader=p.f.owners)


def test_committed_cancellation_survives_cleared_transport_event(tmp_path, closure):
    from threading import Event

    cancel = Event()

    class ClearingLedger(TestLedger):
        def advance(self, attempt, step_id, expected, target, **kwargs):
            super().advance(attempt, step_id, expected, target, **kwargs)
            if target.value == "CANCELLED":
                cancel.clear()

    p = prepared(tmp_path, repository=ClearingLedger(), files=closure)
    cancel.set()
    result = p.executor.execute(p.plan, "cancel-record", cancel=cancel)
    assert not cancel.is_set()
    assert result.status == "CANCELLED"


def test_unknown_qc_state_fails_before_artifact_registration(tmp_path, closure):
    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    evaluate = p.qc.evaluate
    p.qc.evaluate = lambda *args, **kwargs: replace(evaluate(*args, **kwargs), status="UNKNOWN")
    w = p.executor.execute(p.plan, "qc-unknown")
    assert w.status == "FAILED"
    assert w.steps[0].error_code == "qc_observation_invalid"
    assert not w.steps[0].result.artifacts


def test_shared_object_uri_preserves_each_selected_artifact_identity(tmp_path, closure):
    from tests.tool_admission.support import fixture

    base = fixture(tmp_path)
    first = next(iter(base.owners.records.values())).artifact.uri
    for code, record in list(base.owners.records.items()):
        data = record.model_dump(mode="json")
        data["artifact"]["uri"] = first
        base.owners.records[code] = validate(type(record), data)
    p = prepared(tmp_path, base=base, repository=TestLedger(), files=closure)
    expected = {
        s.asset_code
        for b in p.plan.document.steps[0].inputs
        for s in b.sources
        if s.kind == "artifact"
    }
    assert len(expected) > 1
    result = p.executor.execute(p.plan, "shared-uri")
    assert result.status == "SUCCEEDED"
    assert set(result.steps[0].result.artifacts[0].derived_from) == expected
