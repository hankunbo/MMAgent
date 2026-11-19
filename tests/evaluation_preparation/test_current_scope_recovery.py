"""Current profile handoff preserves route, scope ownership and cumulative budget."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import capability_window as window
from packages.multimodal_research import scope_successor as owner
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from tests.evaluation_preparation.test_capability_window import plan_fixture


def planned(tmp_path, monkeypatch):
    parent, tip, found, args = plan_fixture(tmp_path)
    parent.base.visible_value = SimpleNamespace(checksum=digest("visible"))
    old = SimpleNamespace(
        protocol=dict(
            schema_version=window.SCOPE_UNKNOWN_VERSION,
            previous_attempts=[
                dict(requests=32),
                dict(requests=1, kind="unknown-scope-new-attempt/1.0.0"),
            ],
            route_transition={"bound": True},
            deadline=parent.protocol["deadline"],
        )
    )
    found["scope_reuse"] = dict(
        previous="/old",
        protocol_checksum=digest("protocol"),
        failure_checksum=digest("failure"),
        requests=1,
        reused=[dict(target="scope:one")],
    )
    found["input_scope"]["dispatch_targets"] = ["scope:two"]
    found["current_scope_reuse"] = None
    monkeypatch.setattr(window, "route_configuration", lambda p, b: p.config)
    return parent, old, tip, found, args, owner.plan_for(parent, old, tip, found, **args)


def test_current_plan_keeps_original_unknown_and_review_and_only_pending_survey(
    tmp_path, monkeypatch
):
    parent, old, tip, found, args, plan = planned(tmp_path, monkeypatch)
    assert plan["schema_version"] == owner.CURRENT_VERSION
    assert plan["route_transition"] == old.protocol["route_transition"]
    assert plan["previous_attempts"][:-1] == old.protocol["previous_attempts"]
    assert sum(r["requests"] for r in plan["previous_attempts"]) == 34
    assert plan["max_new_requests"] == 6 and plan["max_cumulative_requests"] == 40
    assert plan["input_scope"]["targets"] == ["scope:one", "scope:two"]
    assert plan["input_scope"]["dispatch_targets"] == ["scope:two"]
    assert plan["reuse_assessment_ids"] == ["one"]
    with pytest.raises(ValueError, match="reused_target_dispatched"):
        RenewedCampaign.execute_phase(SimpleNamespace(protocol=plan), "scope:one")
    old.protocol["previous_attempts"] = [dict(requests=39)]
    with pytest.raises(ValueError, match="bounds_or_review_reserve"):
        owner.plan_for(parent, old, tip, found, **args)


def test_current_check_requires_exact_new_plan_and_claim(tmp_path, monkeypatch):
    parent, old, tip, found, args, plan = planned(tmp_path, monkeypatch)
    reader = object()
    saved = [window.claim(plan, plan["output"])]
    monkeypatch.setattr(owner, "source", lambda *a: old)
    monkeypatch.setattr(owner, "inventory", lambda *a, **kw: (tip, found, reader))
    monkeypatch.setattr(
        owner, "claim_documents", lambda *a: SimpleNamespace(read=lambda k: tuple(saved))
    )
    assert window.check(parent, plan) is reader
    bad = deepcopy(plan)
    bad["route_transition"] = {"other": True}
    with pytest.raises(ValueError, match="current_binding_changed"):
        window.check(parent, bad)
    saved.clear()
    with pytest.raises(ValueError, match="claim_binding"):
        window.check(parent, plan)


def test_v9_changes_only_versioned_scope_text_capacity():
    from packages.multimodal_research.active_requirements import load
    from packages.multimodal_research.input_scope import read_policy

    before = load(window.CURRENT_PROFILE)
    after = load(owner.CURRENT_PROFILE)
    assert {k for k in before if before[k] != after[k]} == {"id", "input_scope_policy"}
    old = read_policy(before["input_scope_policy"])
    new = read_policy(after["input_scope_policy"])
    assert new.questions == old.questions and new.procedure == old.procedure
    assert new.region_text_contract == "bounded_full_text_512"


@pytest.mark.parametrize("fault", [None, "claim", "input", "protocol"])
def test_budget_branch_requires_exact_original_predecessor_and_claim(tmp_path, monkeypatch, fault):
    p = tmp_path / "old"
    p.mkdir()
    prior = SimpleNamespace(
        output=p, protocol=dict(schema_version=window.SCOPE_UNKNOWN_VERSION, input="i", task="t")
    )
    failure = dict(reason="confirmed_format")
    (p / "protocol.json").write_text(json.dumps(prior.protocol))
    (p / "stage-failure.json").write_text(json.dumps(failure))
    proof = dict(
        previous=str(p),
        protocol_checksum=digest(prior.protocol),
        failure_checksum=digest(failure),
        reused=[dict(fact=dict(phase=dict(run_id="RUN:X")))],
    )
    plan = dict(schema_version=owner.CURRENT_VERSION, input="i", task="t", scope_reuse=proof)
    current = SimpleNamespace(
        protocol=plan, base=SimpleNamespace(scope_reuse_predecessor=prior), output=tmp_path / "new"
    )
    saved = window.claim(plan, current.output)
    monkeypatch.setattr(
        owner,
        "claim_documents",
        lambda *a: SimpleNamespace(read=lambda k: () if fault == "claim" else (saved,)),
    )
    if fault == "input":
        prior.protocol["input"] = "wrong"
    if fault == "protocol":
        proof["protocol_checksum"] = digest("wrong")
    if fault:
        with pytest.raises(
            ValueError, match="scope_budget_predecessor_binding|original_history_changed"
        ):
            owner.budget_predecessor(current)
    else:
        assert owner.budget_predecessor(current) is prior


def test_unknown_budget_follows_claimed_recovery_branch_without_replaying(monkeypatch):
    from packages.multimodal_research import scope_unknown_successor as unknown

    proof = dict(
        previous="/unknown",
        protocol_checksum=digest("prior"),
        failure_checksum=digest("failure"),
        acknowledged_request_id="request:old",
    )
    prior = SimpleNamespace(
        protocol=dict(schema_version=unknown.VERSION, unknown_scope_attempt=proof),
        output="/confirmed",
        parent=object(),
    )
    current = SimpleNamespace(protocol=dict(schema_version=owner.CURRENT_VERSION), parent=None)
    original = object()
    calls = []
    monkeypatch.setattr(
        owner, "budget_predecessor", lambda c: calls.append("bound-branch") or prior
    )
    monkeypatch.setattr(unknown, "check_history", lambda p: calls.append("old-history"))
    monkeypatch.setattr(unknown, "source", lambda *a: original)
    monkeypatch.setattr(unknown, "inspect", lambda p, a: calls.append("old-request-owner") or proof)
    monkeypatch.setattr(
        unknown,
        "claim_documents",
        lambda *a: SimpleNamespace(read=lambda k: (window.claim(prior.protocol, prior.output),)),
    )
    unknown.verify_budget_history(current, unknown.history_row(proof))
    assert calls == ["bound-branch", "old-history", "old-request-owner"]


@pytest.mark.parametrize("fault", [None, "source", "policy", "ids", "facts", "task", "execution"])
def test_current_inventory_only_rebinds_original_preview_run_identity(monkeypatch, fault):
    before = dict(
        profile=dict(path="old", checksum=digest("old-profile")),
        scope_plan=dict(entries=[dict(target="scope:one")]),
        input_scope=dict(
            targets=["scope:one", "scope:two"], dispatch_targets=["scope:one", "scope:two"]
        ),
        transition=dict(
            profile_checksum=digest("old-profile"),
            catalog_plan=dict(profile_checksum=digest("old-profile")),
        ),
        confirmed_observations_checksum=digest("facts"),
        confirmed_review=dict(
            assessment_ids=["one"],
            compatibility_preview=dict(
                schema_version="review-compatibility-preview/1.0.0",
                state_checksum=digest("old-run"),
                source_checksum=digest("sources"),
                policy_checksum=digest("policy"),
                assessment_ids=["one"],
                execution_authorized=False,
                current_scope_required_before_dispatch=True,
            ),
        ),
    )
    found = deepcopy(before)
    found["profile"] = dict(path=owner.CURRENT_PROFILE, checksum=digest("new-profile"))
    for row in [found["transition"], found["transition"]["catalog_plan"]]:
        row["profile_checksum"] = digest("new-profile")
    preview = found["confirmed_review"]["compatibility_preview"]
    preview["state_checksum"] = digest("new-run")
    if fault == "source":
        preview["source_checksum"] = digest("changed")
    if fault == "policy":
        preview["policy_checksum"] = digest("changed")
    if fault == "ids":
        preview["assessment_ids"] = ["different"]
    if fault == "facts":
        found["confirmed_observations_checksum"] = digest("changed")
    if fault == "task":
        found["scope_plan"]["entries"] = [dict(target="scope:other")]
    if fault == "execution":
        preview["execution_authorized"] = True
    old = SimpleNamespace(
        protocol=dict(schema_version=window.SCOPE_UNKNOWN_VERSION, capability_transition=before)
    )
    original = deepcopy(before)
    monkeypatch.setattr(window, "inventory", lambda *a, **kw: ("tip", found, "reader"))
    monkeypatch.setattr(owner, "inspect", lambda *a: dict(reused=[dict(target="scope:one")]))
    if fault:
        with pytest.raises(
            ValueError, match="medical_inventory_changed|task_or_view_changed|preview_contract"
        ):
            owner.inventory(object(), old, "new")
    else:
        _, new, _ = owner.inventory(object(), old, "new")
        assert new["input_scope"]["dispatch_targets"] == ["scope:two"]
        assert new["confirmed_review"]["compatibility_preview"]["state_checksum"] == digest(
            "new-run"
        )
    assert before == original
