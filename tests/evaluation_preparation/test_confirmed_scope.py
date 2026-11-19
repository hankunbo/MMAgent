import json
from copy import deepcopy
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import capability_window as window
from packages.multimodal_research import scope_unknown_successor as owner
from packages.multimodal_research.active_requirements import load
from tests.evaluation_preparation.test_scope_unknown_successor import planned


def confirmed_plan(tmp_path, monkeypatch):
    parent, prior, tip, found, proof, args, _ = planned(tmp_path, monkeypatch)
    proof["schema_version"] = "original-confirmed-invalid-scope-proof/1.0.0"
    proof["original_attempt"].update(
        old_response_state="confirmed_invalid", saved_response_present=True
    )
    return (
        parent,
        prior,
        tip,
        found,
        proof,
        args,
        owner.plan_for(parent, prior, tip, found, proof, **args),
    )


def test_explicit_confirmed_plan_preserves_budget_and_old_identity(tmp_path, monkeypatch):
    parent, prior, tip, found, proof, args, plan = confirmed_plan(tmp_path, monkeypatch)
    assert plan["schema_version"] == window.SCOPE_CONFIRMED_VERSION
    assert "unknown_scope_attempt" not in plan
    assert owner.saved_attempt(plan) == proof
    assert plan["previous_attempts"][-1]["kind"] == owner.CONFIRMED_KIND
    assert sum(r["requests"] for r in plan["previous_attempts"]) == 34
    assert plan["max_new_requests"] == 6 and plan["reuse_assessment_ids"] == ["one"]
    assert owner.claim_key(proof) == owner.claim_key(
        dict(proof, schema_version="original-unknown-scope-proof/1.0.0")
    )
    assert tip["previous_attempts"] == [dict(requests=33)]
    for altered in [
        dict(plan, unknown_scope_attempt=proof),
        dict(plan, schema_version=owner.VERSION),
    ]:
        with pytest.raises(ValueError, match="explicit_mode_binding"):
            owner.saved_attempt(altered)


def profiles():
    old, new = load(window.NEW_CURRENT_PROFILE), load(window.CORRECTED_SCOPE_PROFILE)
    before = dict(
        profile=dict(path=window.NEW_CURRENT_PROFILE, checksum=digest(old)),
        transition=dict(
            profile_checksum=digest(old),
            catalog_plan=dict(profile_checksum=digest(old)),
            requirements=["exact"],
        ),
        scope_plan=dict(policy=old["input_scope_policy"], entries=["exact"]),
    )
    found = dict(
        profile=dict(path=window.CORRECTED_SCOPE_PROFILE, checksum=digest(new)),
        scope_plan=dict(policy=new["input_scope_policy"], entries=["exact"]),
    )
    return before, found


def test_only_instruction_projection_is_versioned():
    before, found = profiles()
    original = deepcopy(before)
    adjusted = owner.corrected_inventory_baseline(before, found)
    assert before == original
    assert adjusted["transition"]["requirements"] == ["exact"]
    assert adjusted["scope_plan"]["entries"] == ["exact"]
    assert adjusted["transition"]["profile_checksum"] == found["profile"]["checksum"]
    old = json.loads(Path("configs/research/input_scope_policy_v3.json").read_bytes())
    new = json.loads(Path("configs/research/input_scope_policy_v4.json").read_bytes())
    assert {k: v for k, v in old.items() if k != "procedure"} == {
        k: v for k, v in new.items() if k != "procedure"
    }
    assert new["procedure"].startswith(old["procedure"])
    assert "1-240" in new["procedure"] and "not_identified requires coverage=0" in new["procedure"]


@pytest.mark.parametrize("field", ["profile", "transition", "scope_plan"])
def test_bad_previous_projection_rejected(field):
    before, found = profiles()
    if field == "profile":
        before["profile"]["checksum"] = digest("wrong")
    elif field == "transition":
        before["transition"]["profile_checksum"] = digest("wrong")
    else:
        before["scope_plan"]["policy"] = dict(path="bad", checksum=digest("bad"))
    with pytest.raises(ValueError, match="scope_confirmed"):
        owner.corrected_inventory_baseline(before, found)


def test_explicit_cli_for_confirmed_failure(monkeypatch, capsys):
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
            "--confirmed-invalid",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen[0]["confirmed_invalid"] is True and seen[0]["check_only"] is True
    assert json.loads(capsys.readouterr().out)["api_requests"] == 0
