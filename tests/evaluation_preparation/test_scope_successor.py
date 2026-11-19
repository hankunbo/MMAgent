"""Scope successor accounting and exact reuse, without medical-result assertions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import input_scope
from packages.multimodal_research import scope_successor as successor
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from tests.evaluation_preparation.test_capability_window import plan_fixture


def test_bound_plan_counts_confirmed_old_request_and_only_pending_scope(tmp_path):
    parent, tip, inv, kwargs = plan_fixture(tmp_path)
    parent.base.visible_value = SimpleNamespace(checksum=digest("visible"))
    old = SimpleNamespace(
        protocol=dict(
            previous_attempts=deepcopy(tip["previous_attempts"]),
            deadline=parent.protocol["deadline"],
        )
    )
    inv["scope_reuse"] = dict(
        previous="/old",
        protocol_checksum=digest("protocol"),
        failure_checksum=digest("failure"),
        requests=1,
    )
    inv["input_scope"]["dispatch_targets"] = ["scope:two"]
    before = deepcopy((inv, tip, old.protocol))
    plan = successor.plan_for(parent, old, tip, inv, **kwargs)
    assert plan["schema_version"] == successor.VERSION
    assert sum(r["requests"] for r in plan["previous_attempts"]) == 34
    assert plan["max_new_requests"] == 6 and plan["max_model_dispatches"] == 5
    assert plan["input_scope"]["targets"] == ["scope:one", "scope:two"]
    assert plan["input_scope"]["dispatch_targets"] == ["scope:two"]
    assert plan["reuse_assessment_ids"] == ["one"]
    assert plan["tip"] == tip  # Original fact-tip accounting remains immutable.
    assert (inv, tip, old.protocol) == before
    old.protocol["previous_attempts"] = [dict(requests=38)]
    with pytest.raises(ValueError, match="bounds_or_review_reserve"):
        successor.plan_for(parent, old, tip, inv, **kwargs)


def test_execute_skips_reused_scope_but_dispatches_unattempted_scope(monkeypatch):
    calls = []
    campaign = SimpleNamespace(scope_targets=("input-scope:one", "input-scope:two"))
    campaign.execute_phase = lambda t: calls.append(t) or dict(accepted=True, error=None)
    monkeypatch.setattr(successor, "inherited", lambda c: {"input-scope:one": {"accepted": True}})
    monkeypatch.setattr(input_scope, "read_results", lambda c, require_all: ["verified"])
    assert input_scope.execute(campaign) == ["verified"]
    assert calls == ["input-scope:two"]


def test_direct_phase_cannot_redispatch_inherited_target():
    campaign = SimpleNamespace(
        protocol=dict(
            schema_version=successor.VERSION,
            scope_reuse=dict(reused=[dict(target="input-scope:one")]),
        )
    )
    with pytest.raises(ValueError, match="confirmed_target_no_dispatch"):
        RenewedCampaign.execute_phase(campaign, "input-scope:one")


def test_public_profile_changes_only_scope_representation():
    from packages.multimodal_research.active_requirements import load

    old = load("configs/research/multimodal_catalog_acquisition_v6.json")
    new = load(successor.PROFILE)
    assert {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)} == {
        "input_scope_policy"
    }
    policy = input_scope.read_policy(new["input_scope_policy"])
    assert policy.schema_version == "input-scope-phase-policy/1.1.0"
    assert policy.questions == input_scope.read_policy(old["input_scope_policy"]).questions


def test_old_versions_do_not_inherit_recovery(monkeypatch):
    monkeypatch.setattr(successor, "source", lambda *a: pytest.fail("unexpected ancestry read"))
    assert successor.inherited(SimpleNamespace()) == {}
    for version in ("multimodal-campaign-renewal/1.81.0", "multimodal-campaign-renewal/1.82.0"):
        assert successor.inherited(SimpleNamespace(protocol={"schema_version": version})) == {}
