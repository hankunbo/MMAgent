"""Known rejected responses may qualify a different method, never UNKNOWN replay."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.multimodal_research import pending_review
from packages.multimodal_research import review_method_window as window
from packages.multimodal_research import root_review_window as root
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from tests.evaluation_preparation.test_review_method_window import fixture_plan


def owner_inventory():
    return dict(
        accepted_ids=["accepted"],
        selected_ids=["failed", "unstarted"],
        quarantined_batches=[
            dict(
                request_ids=["first", "correction"],
                new_reassessment_authorized=True,
                known_format_failure=dict(
                    last_request_id="correction",
                    original_answer_not_accepted=True,
                    attempts=[dict(request_id="first"), dict(request_id="correction")],
                ),
            )
        ],
    )


def request_owner(extra=()):
    records = [
        SimpleNamespace(request_id="good", error=None),
        SimpleNamespace(request_id="first", error="schema"),
        SimpleNamespace(request_id="correction", error="schema"),
        *extra,
    ]
    return SimpleNamespace(records=lambda: records)


def test_format_bridge_calls_original_pending_owner_with_exact_ack(monkeypatch):
    calls = []
    value = owner_inventory()

    def inventory(parent, tip, **kw):
        calls.append((parent, tip, kw))
        return deepcopy(value)

    monkeypatch.setattr(pending_review, "inventory", inventory)
    proof = window.confirmed_format_origin("parent", "tip", request_owner(), ["correction"])
    assert calls == [("parent", "tip", {"format_requests": ("correction",)})]
    assert proof["confirmed_failed_request_ids"] == ["correction", "first"]
    assert proof["old_responses_reused"] is False
    assert value == owner_inventory()
    with pytest.raises(ValueError, match="ack_required"):
        window.confirmed_format_origin("p", "t", request_owner(), [])
    with pytest.raises(ValueError, match="history_mismatch"):
        window.confirmed_format_origin(
            "p",
            "t",
            request_owner([SimpleNamespace(request_id="unknown", error="timeout")]),
            ["correction"],
        )


@pytest.mark.parametrize("fault", ["unknown", "not_authorized", "wrong_ack", "attempt_mismatch"])
def test_format_bridge_never_accepts_unresolved_quarantines(monkeypatch, fault):
    value = owner_inventory()
    batch = value["quarantined_batches"][0]
    if fault == "unknown":
        batch.pop("known_format_failure")
    elif fault == "not_authorized":
        batch["new_reassessment_authorized"] = False
    elif fault == "wrong_ack":
        batch["known_format_failure"]["last_request_id"] = "other"
    else:
        batch["request_ids"].append("unknown")
    monkeypatch.setattr(pending_review, "inventory", lambda *a, **kw: value)
    with pytest.raises(ValueError, match="unresolved_quarantine"):
        window.confirmed_format_origin("p", "t", request_owner(), ["correction"])


def format_plan():
    parent, result, proof, tip, args, _ = fixture_plan()
    parent.cumulative_request_limit = lambda: 40
    tip["previous_attempts"] = [dict(requests=18)]
    proof["confirmed_format_origin"] = dict(acknowledged_format_requests=["correction"])
    return parent, result, proof, tip, args, window.plan_for(parent, result, tip, proof, **args)


def test_new_version_keeps_case_cap_and_old_version_unchanged(monkeypatch):
    parent, result, proof, tip, args, plan = format_plan()
    assert plan["schema_version"] == window.FORMAT_VERSION
    assert window.cumulative_limit(plan) == 25
    campaign = object.__new__(RenewedCampaign)
    campaign.protocol = plan
    assert campaign.cumulative_request_limit() == 25
    seen = []

    def origin(*a, **kw):
        seen.append(kw)
        return proof

    monkeypatch.setattr(root, "origin_proof", origin)
    window.check(plan, parent, result)
    assert seen[0]["format_requests"] == ["correction"]
    assert root.configuration_for(parent.config, plan) == window.configuration(
        parent.config, window.PROFILE
    )
    tip["previous_attempts"] = [dict(requests=34)]
    with pytest.raises(ValueError, match="cumulative_limit"):
        window.plan_for(parent, result, tip, proof, **args)
    old = fixture_plan()[-1]
    assert old["schema_version"] == window.VERSION and window.cumulative_limit(old) == 47


def test_version_downgrade_and_repeated_method_rejected(monkeypatch):
    parent, result, proof, tip, args, plan = format_plan()
    with pytest.raises(ValueError, match="format_version_binding"):
        window.check({**plan, "schema_version": window.VERSION}, parent, result)
    changed = deepcopy(plan)
    changed["text_review_method"]["diagnostic_requests"] = 40
    with pytest.raises(ValueError, match="cumulative_limit"):
        window.cumulative_limit(changed)
    parent.protocol["schema_version"] = window.FORMAT_VERSION
    with pytest.raises(ValueError, match="already_used"):
        window.plan_for(parent, result, tip, proof, **args)


def test_public_cli_preserves_format_ack_for_method(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main

    calls = []
    monkeypatch.setattr(window, "prepare", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case",
            "renewal-prepare",
            "--facts",
            "old",
            "--previous",
            "old",
            "--output",
            "new",
            "--mode",
            "review",
            "--authorization",
            "test",
            "--review-method-profile",
            window.PROFILE,
            "--acknowledge-format-request",
            "correction",
        ],
    )
    assert main() == 0
    assert calls[0]["acknowledge_format_request"] == ["correction"]
    assert calls[0]["format_closure_seconds"] is None
    capsys.readouterr()


def test_routed_config_requires_new_version_and_preserves_original_route():
    from packages.llm_provider.config import with_network_route
    from packages.llm_provider.network_route import load_route
    from tests.evaluation_preparation.test_review_method_window import old_config

    old = with_network_route(
        old_config(), load_route("configs/agents/environment_https_route_v1.json")
    )
    new = window.configuration(old, window.PROFILE, format_recovery=True)
    assert new.schema_version == old.schema_version == "agent-provider-config/9.1.0"
    assert new.network_route == old.network_route
    before, after = old.model_dump(mode="json"), new.model_dump(mode="json")
    after["transport"].update(
        model=before["transport"]["model"],
        timeout_seconds=before["transport"]["timeout_seconds"],
    )
    assert after == before
    with pytest.raises(ValueError, match="original_configuration"):
        window.configuration(old, window.PROFILE)


def test_base_campaign_budget_comes_from_owner_not_renewal_only_field():
    parent, result, proof, tip, args, _ = format_plan()
    del parent.protocol["max_cumulative_requests"]
    parent.cumulative_request_limit = lambda: 26
    plan = window.plan_for(parent, result, tip, proof, **args)
    assert plan["text_review_method"]["old_window_cap"] == 26
    parent.cumulative_request_limit = lambda: 24
    with pytest.raises(ValueError, match="cumulative_limit"):
        window.plan_for(parent, result, tip, proof, **args)
