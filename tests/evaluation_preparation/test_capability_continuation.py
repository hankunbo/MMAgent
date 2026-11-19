"""Typed continuation selects only unfinished work and counts every prior request."""

import copy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_capability_execution as execution
from packages.evaluation_preparation import generic_capability_recovery as recovery
from packages.evaluation_preparation.generic_capability_successor import (
    CONTINUATION_VERSION,
    CapabilitySuccessor,
    selection_for,
)
from packages.grounding.models import digest
from tests.evaluation_preparation.test_generic_capability_successor import config


def setup(monkeypatch):
    event = dict(code="review:original", checksum="same")
    stage = dict(
        event=event,
        run_id="RUN:FAILED",
        reason="deadline",
        domain_status="not_accepted",
        acceptance_key=None,
    )
    submitted = dict(
        status="rejected",
        submission_checksum="same",
        receipt=dict(status="rejected", submission_checksum="same"),
    )
    documents = {
        "live-stage-intent:vision": (dict(role="vision", event=event),),
        "live-stage:vision": (stage,),
        "review:review:original:result": (stage,),
        "submission:review:original:domain": (submitted,),
    }
    selection = dict(schema_version="generic-capability-successor/1.1.0", key="expired")
    rows = [
        dict(target="accepted", status="accepted", accepted=True, requests=[], reservations=1),
        dict(
            target="rejected",
            run_id="RUN:FAILED",
            status="confirmed_response_requires_local_acceptance_check",
            accepted=False,
            error="deadline",
            reservations=1,
            requests=[dict(request_id="request:rejected", status="confirmed_response")],
        ),
        dict(
            target="unsent",
            status="registered_not_dispatched",
            accepted=False,
            requests=[],
            reservations=0,
        ),
        dict(target=None, status="not_registered", reservations=0, requests=[]),
    ]
    inventory = dict(phases=rows, acquired_targets=["accepted"], cumulative_requests=12)
    original = SimpleNamespace(spec={"key": "root"}, config={"fixture_provider": "offline"})
    checked = SimpleNamespace(
        campaign=original,
        history={"reservations": 10},
        projection={"dispatch_targets": ["accepted", "rejected", "unsent"]},
        owner=SimpleNamespace(snapshot={}, policy={}),
        execution_config=original.config,
        route_transition=None,
    )
    previous = SimpleNamespace(
        selection=selection,
        config=config(),
        checked=checked,
        documents=lambda run: SimpleNamespace(read=lambda key: documents.get(key)),
    )
    monkeypatch.setattr(execution, "load_selection", lambda *a, **k: previous)
    monkeypatch.setattr(recovery, "snapshot", lambda *a, **k: copy.deepcopy(inventory))
    return previous, inventory, documents


def test_reuse_accepted_and_explicitly_distinguish_rejected_and_unsent(monkeypatch):
    previous, inventory, docs = setup(monkeypatch)
    checked = recovery.continuation_origin(
        "expired", {"path": "proof", "checksum": "same"}, ["request:rejected"]
    )
    assert checked.continuation["dispatch_targets"] == ["rejected", "unsent"]
    assert checked.continuation["reuse_targets"] == ["accepted"]
    assert checked.continuation["inventory"]["cumulative_requests"] == 12
    assert checked.limits["expected_vision_requests"] == 2
    assert (
        checked.limits["minimum_review_requests"] == 1
    )  # includes the previously accepted observation
    assert checked.predecessor is previous
    value = selection_for(checked, "new-synthetic")
    assert value["schema_version"] == CONTINUATION_VERSION
    assert value["continuation"]["root_claim"] == previous.selection
    assert value["continuation"]["rejected_attempts"][0]["previous_answer_relabelled"] is False
    assert docs["live-stage:vision"][0]["domain_status"] == "not_accepted"


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "not_terminal",
        "review_registered",
        "accepted_commit",
        "wrong_event",
        "no_rejection",
        "wrong_receipt",
        "budget",
        "no_ack",
        "wrong_ack",
        "duplicate_ack",
    ],
)
def test_no_silent_resend_or_budget_reset(monkeypatch, change):
    previous, inventory, docs = setup(monkeypatch)
    ack = ["request:rejected"]
    if change == "unknown":
        inventory["phases"][2]["status"] = "unknown_requires_explicit_resolution"
    elif change == "not_terminal":
        inventory["phases"][2]["status"] = "running"
    elif change == "review_registered":
        inventory["phases"][-1]["status"] = "registered_not_dispatched"
    elif change == "accepted_commit":
        docs["accepted:review:original:domain"] = ({"accepted": True},)
    elif change == "wrong_event":
        docs["live-stage-intent:vision"][0]["event"] = {"code": "different"}
    elif change == "no_rejection":
        docs.pop("submission:review:original:domain")
    elif change == "wrong_receipt":
        docs["submission:review:original:domain"][0]["receipt"]["submission_checksum"] = "different"
    elif change == "budget":
        inventory["cumulative_requests"] = 39
    elif change == "no_ack":
        ack = []
    elif change == "wrong_ack":
        ack = ["request:foreign"]
    elif change == "duplicate_ack":
        ack *= 2
    with pytest.raises(ValueError, match="generic_capability_"):
        recovery.continuation_origin("expired", {"path": "proof", "checksum": "same"}, ack)


def test_review_only_continuation_counts_reused_observations(monkeypatch):
    previous, inventory, _ = setup(monkeypatch)
    inventory["phases"] = [inventory["phases"][0], inventory["phases"][-1]]
    checked = recovery.continuation_origin("expired", {"path": "proof", "checksum": "same"})
    assert checked.continuation["dispatch_targets"] == []
    assert checked.limits["expected_vision_requests"] == 0
    assert checked.limits["minimum_review_requests"] == 1


def test_previous_accounting_change_rejected_before_new_budget(monkeypatch):
    previous, inventory, _ = setup(monkeypatch)
    checked = recovery.continuation_origin(
        "expired", {"path": "proof", "checksum": "same"}, ["request:rejected"]
    )
    from packages.evaluation_preparation import generic_capability_successor as successor

    value = object.__new__(CapabilitySuccessor)
    value.checked = checked
    value.acquisition = checked.campaign
    value.selection = selection_for(checked, "new-synthetic")
    monkeypatch.setattr(successor, "completed_history", lambda *a, **k: checked.history)
    prior = value.prior()
    assert prior["reservations"] == 12
    assert prior["predecessor_inventory_checksum"] == digest(checked.continuation["inventory"])
    inventory["cumulative_requests"] = 13
    with pytest.raises(ValueError, match="continuation_original_changed"):
        value.prior()


@pytest.mark.parametrize("check_only", [True, False])
def test_cli_composition_claim_and_start_are_exact_and_not_repeatable(
    tmp_path, monkeypatch, check_only
):
    import json
    from contextlib import contextmanager

    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.evaluation_preparation import generic_control, generic_execution_lock, generic_run
    from packages.evaluation_preparation.generic_capability_successor import CONTINUATION_CLAIM
    from packages.evaluation_preparation.generic_definition_successor import CLAIM
    from packages.multimodal_research import unknown_acquisition

    previous, _, _ = setup(monkeypatch)
    previous.selection.update(parent_key="root")
    checked = recovery.continuation_origin(
        "expired", {"path": "proof", "checksum": "same"}, ["request:rejected"]
    )
    monkeypatch.setattr(recovery, "continuation_origin", lambda *a, **kw: checked)
    for module in (execution, generic_control):
        monkeypatch.setattr(module, "directory", lambda key: tmp_path / key)
    (tmp_path / "expired").mkdir()
    (tmp_path / "root").mkdir()
    (tmp_path / "expired/input.json").write_text(json.dumps(previous.selection))
    (tmp_path / "root" / CLAIM).write_text(json.dumps(previous.selection))
    (tmp_path / "root/binding.json").write_text('{"original":"case"}')
    protected = (tmp_path / "root" / CLAIM).read_bytes()
    proof = tmp_path / "public-proof.json"
    proof.write_text(
        "{}"
    )  # The original proof reader is tested separately, not bypassed in production.
    calls = []

    @contextmanager
    def lease(*args):
        yield lambda: None

    monkeypatch.setattr(generic_execution_lock, "target", lambda: object())
    monkeypatch.setattr(CasePipelineRepository, "__init__", lambda *a: None)
    monkeypatch.setattr(CasePipelineRepository, "guarded_session", lease)
    monkeypatch.setattr(unknown_acquisition, "check_network", lambda *a: None)
    monkeypatch.setattr(
        execution,
        "CapabilitySuccessor",
        lambda *a: SimpleNamespace(execute=lambda: calls.append("execute") or {"complete": True}),
    )
    monkeypatch.setattr(
        generic_run, "persist_export", lambda key, campaign, result: dict(result=result)
    )
    args = ("expired", "next", str(proof), ["request:rejected"])
    result = execution.continue_run(*args, check_only=check_only)
    assert (tmp_path / "root" / CLAIM).read_bytes() == protected
    if check_only:
        assert result["new_window_created"] is False and calls == []
        assert not (tmp_path / "next").exists()
        assert not (tmp_path / "expired" / CONTINUATION_CLAIM).exists()
    else:
        claim = json.loads((tmp_path / "expired" / CONTINUATION_CLAIM).read_bytes())
        assert claim == json.loads((tmp_path / "next/input.json").read_bytes())
        started = (tmp_path / "next/started.json").read_bytes()
        with pytest.raises(ValueError, match="started_use_export_not_resend"):
            execution.continue_run(*args)
        assert calls == ["execute"]
        assert (tmp_path / "next/started.json").read_bytes() == started


def test_terminal_cli_exposes_typed_continuation_without_hidden_dispatch(monkeypatch, capsys):
    import sys

    from apps import fairvlmed_case

    calls = []
    monkeypatch.setattr(
        execution,
        "continue_run",
        lambda *args, **kw: calls.append((args, kw)) or {"api_requests": 0},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fairvlmed_case",
            "capability-continue-preflight",
            "--key",
            "expired",
            "--successor-key",
            "next",
            "--history-software",
            "public-proof.json",
            "--acknowledge-rejected-request",
            "request:rejected",
        ],
    )
    assert fairvlmed_case.main() == 0
    assert calls == [
        (("expired", "next", "public-proof.json", ["request:rejected"]), {"check_only": True})
    ]
    assert '"api_requests": 0' in capsys.readouterr().out
