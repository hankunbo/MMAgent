"""Failed successor identity/accounting checks, before any new dispatch."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_acquisition_successor as owner
from packages.grounding.models import digest
from tests.evaluation_preparation.test_directional_review import fixture as state_fixture


def fixture(tmp_path, monkeypatch):
    state = state_fixture().owner.current()
    proof = dict(parent_key="root", target="check", request_id="root-request", previous_requests=13)
    binding = {"case_code": "CASE:SYNTHETIC"}
    docs = SimpleNamespace(run_id=state.material.scope.run_id)
    config = state_fixture().config
    c = SimpleNamespace(spec={"key": "root"}, config=config, base=SimpleNamespace(binding=binding))
    checked = SimpleNamespace(campaign=c, binding=binding, proof=proof)
    monkeypatch.setattr(owner, "directory", lambda key: tmp_path / key)
    selection = owner.selection_for(checked, "failed")
    result = dict(
        complete=False,
        phases=[dict(target="check", accepted=False, acceptance_key=None, run_id=docs.run_id)],
        new_requests=1,
        prior_graph_requests=13,
    )
    request = dict(lock={"run_id": docs.run_id})
    protocol = dict(
        input=digest(c.spec), provider_checksum=digest(config), observations={"proof": proof}
    )
    report = {"result": result}
    saved = {
        "dynamic-case-result": (result,),
        "dynamic-phase-result": (result["phases"][0],),
        "dynamic-campaign": (protocol,),
        "dynamic-phase-input": (state.model_dump(mode="json"),),
        "dynamic-case-report": (report, {"report_checksum": digest(report)}),
    }
    docs.read = saved.get
    c.documents = lambda _: docs
    c.sessions = SimpleNamespace()
    c.contexts = SimpleNamespace(
        read_research_run=lambda _: SimpleNamespace(
            request=SimpleNamespace(model_dump=lambda **_: request)
        )
    )
    for key, value in {
        "root/acquisition-successor.json": selection,
        "failed/input.json": selection,
        "failed/binding.json": binding,
        "failed/evidence/result.json": result,
        "failed/evidence/vision/request.json": request,
        "failed/evidence/protocol.json": protocol,
    }.items():
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    journal = [{"phase": "reserved", "request_id": "child-request"}]
    monkeypatch.setattr(
        owner, "RuntimeAgentJournal", lambda _: SimpleNamespace(read=lambda: journal)
    )
    projected = SimpleNamespace(project=lambda **_: (state, {}), verify=lambda _: None)
    monkeypatch.setattr(owner, "GenericPhase", lambda *_, **__: projected)
    calls = []

    def original_attempt(adapter, phase, request_id):
        calls.append(request_id)
        if request_id != "child-request":
            raise ValueError("unknown_acquisition_request_identity")
        return {"request_id": request_id, "old_response_state": "unknown"}

    monkeypatch.setattr(
        "packages.multimodal_research.unknown_acquisition.original_attempt", original_attempt
    )
    return checked, saved, journal, calls, state


def test_failed_child_exact_record_not_an_implicit_retry(tmp_path, monkeypatch):
    checked, saved, journal, calls, _ = fixture(tmp_path, monkeypatch)
    before = (
        digest(saved),
        digest(journal),
        (tmp_path / "root/acquisition-successor.json").read_bytes(),
    )
    proof = owner.failed_successor(checked, "failed", "child-request")
    assert proof["requests"] == 1 and proof["attempt"]["old_response_state"] == "unknown"
    assert owner.failed_successor(checked, "failed", "child-request") == proof
    assert calls == ["child-request", "child-request"]
    assert before == (
        digest(saved),
        digest(journal),
        (tmp_path / "root/acquisition-successor.json").read_bytes(),
    )
    assert not (tmp_path / "failed/acquisition-successor.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "claim",
        "binding",
        "cancelled",
        "accepted",
        "complete",
        "missing_report",
        "wrong_request",
        "extra_request",
        "prior_count",
        "protocol",
        "material",
    ],
)
def test_failed_chain_rejects_unqualified_history(tmp_path, monkeypatch, fault):
    checked, saved, journal, calls, state = fixture(tmp_path, monkeypatch)
    if fault in {"claim", "binding"}:
        path = tmp_path / (
            "root/acquisition-successor.json" if fault == "claim" else "failed/binding.json"
        )
        path.write_text("{}")
    elif fault == "cancelled":
        (tmp_path / "failed/cancel-request.json").write_text("{}")
    elif fault in {"accepted", "complete", "prior_count"}:
        result = saved["dynamic-case-result"][0]
        if fault == "accepted":
            result["phases"][0]["accepted"] = True
        elif fault == "complete":
            result["complete"] = True
        else:
            result["prior_graph_requests"] = 0
        (tmp_path / "failed/evidence/result.json").write_text(json.dumps(result))
    elif fault == "missing_report":
        saved.pop("dynamic-case-report")
    elif fault == "extra_request":
        journal.append({"phase": "reserved", "request_id": "hidden-request"})
    elif fault == "protocol":
        saved["dynamic-campaign"] = ({"foreign": True},)
    elif fault == "material":
        saved["dynamic-phase-input"] = (
            state.model_copy(update={"config_checksum": digest("other")}).model_dump(mode="json"),
        )
    request = "foreign-request" if fault == "wrong_request" else "child-request"
    with pytest.raises(
        ValueError, match="generic_acquisition_failed_child_|unknown_acquisition_request_identity"
    ):
        owner.failed_successor(checked, "failed", request)
    assert not (tmp_path / "failed/acquisition-successor.json").exists()


@pytest.mark.parametrize("key", ["root", "failed"])
def test_new_attempt_cannot_reuse_either_predecessor_key(key):
    checked = SimpleNamespace(proof={"parent_key": "root", "failed_successor": {"key": "failed"}})
    with pytest.raises(ValueError, match="generic_acquisition_distinct_successor"):
        owner.selection_for(checked, key)


def test_partial_cli_chain_identity_rejected_before_original_owners():
    with pytest.raises(ValueError, match="generic_acquisition_failed_chain_all_bindings_required"):
        owner.run(
            "root", "new", "check", "old-request", previous_successor="failed", preflight=True
        )


def test_cancelled_failed_parent_is_observed_during_request(tmp_path, monkeypatch):
    from threading import Event

    monkeypatch.setattr(owner, "directory", lambda key: tmp_path / key)
    inherited = Event()
    monkeypatch.setattr(owner.GenericReview, "cancellation_signal", lambda _: inherited)
    c = owner.FailedAcquisitionSuccessor.__new__(owner.FailedAcquisitionSuccessor)
    c.checked = SimpleNamespace(proof={"failed_successor": {"key": "failed"}})
    signal = c.cancellation_signal()
    assert not signal.is_set()
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed/cancel-request.json").write_text("{}")
    assert signal.is_set()
    (tmp_path / "failed/cancel-request.json").unlink()
    inherited.set()
    assert signal.is_set()


@pytest.mark.parametrize("changed", [False, True])
def test_historical_chain_restore_preserves_freeze_but_rechecks_current_proof(
    tmp_path, monkeypatch, changed
):
    monkeypatch.setattr(owner, "directory", lambda key: tmp_path / key)
    proof = dict(
        schema_version=owner.CHAIN_VERSION,
        parent_key="root",
        target="check",
        request_id="root-request",
        failed_successor={"key": "failed", "request_id": "child-request"},
        client_binding={"path": "configs/agents/system_curl_client_v1.json"},
    )
    checked = SimpleNamespace(proof=proof, binding={"case": "same"})
    value = owner.selection_for(checked, "new")
    value["implementation"] = {"historical.py": "sha256:" + "a" * 64}
    for key, content in {
        "new/input.json": value,
        "new/binding.json": checked.binding,
        "failed/acquisition-successor.json": value,
        "new/evidence/result.json": {"complete": True, "phases": [{"target": None}]},
    }.items():
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    if changed:
        checked = SimpleNamespace(proof=dict(proof, changed=True), binding=checked.binding)
    monkeypatch.setattr(owner, "chained_origin", lambda *args: checked)
    calls = []
    monkeypatch.setattr(owner, "FailedAcquisitionSuccessor", lambda checked, key: (checked, key))
    monkeypatch.setattr(
        owner.pending, "confirmed_campaign", lambda c: calls.append(c) or "confirmed"
    )
    if changed:
        with pytest.raises(ValueError, match="generic_acquisition_restore_binding"):
            owner.restore(tmp_path / "new")
        assert calls == []
    else:
        assert owner.restore(tmp_path / "new") == "confirmed"
        assert calls == [(checked, "new")]
        assert json.loads((tmp_path / "new/input.json").read_bytes()) == value
