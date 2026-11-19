"""Explicit attempt plans, cold claim qualification, and fail-closed failure scope."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import capability_window as window
from packages.multimodal_research import scope_unknown_successor as owner
from tests.evaluation_preparation.test_capability_window import plan_fixture


def planned(tmp_path, monkeypatch):
    parent, tip, found, args = plan_fixture(tmp_path)
    parent.base.visible_value = SimpleNamespace(checksum=digest("visible"))
    proof = dict(
        previous=str(tmp_path / "failure"),
        protocol_checksum=digest("old"),
        failure_checksum=digest("failure"),
        requests=1,
        acknowledged_request_id="request:old",
        original_attempt=dict(run_id="RUN:OLD", target="input-scope:one"),
        duplicate_billing_possible=True,
        new_attempt_not_old_replay=True,
    )
    predecessor = SimpleNamespace(protocol=dict(deadline=parent.protocol["deadline"]))
    monkeypatch.setattr(window, "route_configuration", lambda p, b: p.config)
    args["route_transition"] = {"explicit": True}
    plan = owner.plan_for(parent, predecessor, tip, found, proof, **args)
    return parent, predecessor, tip, found, proof, args, plan


def test_original_plan_counts_unknown_retains_facts_and_bounds(tmp_path, monkeypatch):
    parent, prior, tip, found, proof, args, plan = planned(tmp_path, monkeypatch)
    assert plan["schema_version"] == owner.VERSION
    assert sum(r["requests"] for r in plan["previous_attempts"]) == 34
    assert plan["max_new_requests"] == 6 and plan["max_cumulative_requests"] == 40
    assert tip["previous_attempts"] == [dict(requests=33)]
    assert plan["reuse_assessment_ids"] == ["one"]
    assert plan["facts"] == str(parent.output.absolute())
    assert plan["tip"] == tip and plan["prior_deadline"] == prior.protocol["deadline"]
    assert plan["original_unknown_not_replayed"] is True
    assert plan["unknown_scope_attempt"]["duplicate_billing_possible"] is True
    changed = deepcopy(tip)
    changed["previous_attempts"][0]["requests"] = 39
    with pytest.raises(ValueError, match="capability_window_bounds"):
        owner.plan_for(parent, prior, changed, found, proof, **args)


def test_cold_original_claim_and_changed_plan_reject(tmp_path, monkeypatch):
    parent, prior, tip, found, proof, args, plan = planned(tmp_path, monkeypatch)
    reader = object()
    saved = [window.claim(plan, plan["output"])]
    docs = SimpleNamespace(read=lambda key: tuple(saved))
    monkeypatch.setattr(owner, "source", lambda *a: prior)
    monkeypatch.setattr(owner, "inventory", lambda *a, **kw: (tip, found, reader, proof))
    monkeypatch.setattr(owner, "claim_documents", lambda *a: docs)
    assert owner.check(parent, plan) is reader
    changed = deepcopy(plan)
    changed["max_new_requests"] += 1
    with pytest.raises(ValueError, match="binding_changed"):
        owner.check(parent, changed)
    saved.clear()
    with pytest.raises(ValueError, match="claim_binding"):
        owner.check(parent, plan)


@pytest.mark.parametrize(
    "fault", ["cancel", "result", "protocol", "input", "history", "stage", "count", "phases", "ack"]
)
def test_original_failure_rejects_unqualified_scope(tmp_path, fault):
    old = dict(input=digest("input"), previous_attempts=[])
    failure = dict(
        protocol_checksum=digest(old),
        input_checksum=old["input"],
        campaign_id=tmp_path.name,
        previous_attempts=[],
        cancel_marker=False,
        complete=False,
        failure_stage="acquisition",
        new_requests=1,
        confirmed_phases=[{}],
    )
    ack = "request:old"
    if fault == "cancel":
        (tmp_path / "cancel").touch()
    elif fault == "result":
        (tmp_path / "result.json").write_text("{}")
    elif fault == "protocol":
        failure["protocol_checksum"] = digest("other")
    elif fault == "input":
        failure["input_checksum"] = digest("other")
    elif fault == "history":
        failure["previous_attempts"] = [dict(requests=0)]
    elif fault == "stage":
        failure["failure_stage"] = "review"
    elif fault == "count":
        failure["new_requests"] = 0
    elif fault == "phases":
        failure["confirmed_phases"] = []
    else:
        ack = ""
    (tmp_path / "stage-failure.json").write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="original_failure_binding"):
        owner.inspect(SimpleNamespace(output=tmp_path, protocol=old), ack)


def test_public_cli_is_explicit_original_prepare(monkeypatch, capsys):
    import sys

    from apps import multimodal_case

    seen = []
    monkeypatch.setattr(owner, "prepare", lambda **kw: seen.append(kw) or dict(api_requests=0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "capability-scope-attempt-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--authorization",
            "test",
            "--acknowledge-request",
            "request:old",
            "--network-route",
            "route.json",
            "--check-only",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen == [
        dict(
            previous="old",
            output="new",
            authorization="test",
            acknowledge_request="request:old",
            network_route="route.json",
            check_only=True,
        )
    ]
    assert json.loads(capsys.readouterr().out)["api_requests"] == 0


def test_history_keeps_unknown_and_rejects_old_file_change(tmp_path, monkeypatch):
    parent, prior, tip, found, proof, args, _ = planned(tmp_path, monkeypatch)
    path = tmp_path / "failure"
    path.mkdir()
    protocol, failure = {"original": "protocol"}, {"original": "failure"}
    (path / "protocol.json").write_text(json.dumps(protocol))
    (path / "stage-failure.json").write_text(json.dumps(failure))
    proof.update(protocol_checksum=digest(protocol), failure_checksum=digest(failure))
    plan = owner.plan_for(parent, prior, tip, found, proof, **args)
    owner.check_history(plan)
    altered = deepcopy(plan)
    altered["previous_attempts"][-1]["requests"] = 0
    with pytest.raises(ValueError, match="scope_unknown_history_changed"):
        owner.check_history(altered)
    (path / "stage-failure.json").write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="scope_unknown_history_changed"):
        owner.check_history(plan)


@pytest.mark.parametrize(
    "field",
    [
        "transition",
        "scope_plan",
        "review_target_ids",
        "acquisition_targets",
        "native_checksum",
        "bound_observations_checksum",
        "confirmed_observations_checksum",
    ],
)
def test_retry_cannot_alter_original_medical_inventory(tmp_path, monkeypatch, field):
    parent, prior, tip, found, proof, _, _ = planned(tmp_path, monkeypatch)
    values = {
        name: {"original": name}
        for name in [
            "transition",
            "scope_plan",
            "review_target_ids",
            "acquisition_targets",
            "native_checksum",
            "bound_observations_checksum",
            "confirmed_observations_checksum",
        ]
    }
    from packages.multimodal_research.active_requirements import load

    values["profile"] = dict(
        path=window.CURRENT_PROFILE, checksum=digest(load(window.CURRENT_PROFILE))
    )
    prior.protocol["capability_transition"] = deepcopy(values)
    values[field] = {"changed": True}
    monkeypatch.setattr(owner, "inspect", lambda *a: proof)
    monkeypatch.setattr(window, "inventory", lambda *a, **kw: (tip, values, object()))
    with pytest.raises(ValueError, match="scope_unknown_current_inventory_changed:" + field):
        owner.inventory(parent, prior, tmp_path / "next", "request:old", {"route": True})
