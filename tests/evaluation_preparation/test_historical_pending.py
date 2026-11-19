"""An acknowledged ancestor request is not permission to replay current work."""

from copy import deepcopy

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import pending_review
from packages.multimodal_research.historical_pending import VERSION, selected_material
from tests.agent_provider.test_accepted_review_reuse import accepted_fixture, state_from
from tests.evaluation_preparation.test_pending_reassessment_window import fixture, install_proof


def test_old_material_can_match_new_run_without_copying_judgments():
    f = accepted_fixture(new_observation=True)
    old = f.old.owner.current()
    current = f.owner.current()
    ids = [i.id for i in old.policy.items]
    proof = selected_material(old, current, ids)
    assert proof["exact_item_observation_sources"]
    assert proof["old_state_checksum"] == digest(old)
    assert proof["current_state_checksum"] == digest(current)
    assert not f.budget.records()


@pytest.mark.parametrize("change", ["case", "goal", "definition", "observation", "source"])
def test_changed_meaning_or_fact_cannot_use_old_unknown_permission(change):
    f = accepted_fixture(new_observation=True)
    old = f.old.owner.current()
    raw = f.owner.current().model_dump(mode="json")
    item = raw["policy"]["items"][0]
    if change == "case":
        raw["material"]["scope"]["case_code"] = "CASE:OTHER"
    elif change == "goal":
        raw["material"]["goal_summary"] += " changed"
        raw["policy"]["target_summary"] = raw["material"]["goal_summary"]
    elif change == "definition":
        item["definition"]["interpretation"] += " changed"
        item["definition"]["reference"]["checksum"] = digest(
            {k: v for k, v in item["definition"].items() if k != "reference"}
        )
        next(r for r in raw["policy"]["requirements"] if r["id"] == item["id"])["definition"] = (
            deepcopy(item["definition"])
        )
    elif change == "observation":
        raw["material"]["observations"][0]["value"] = "changed"
    else:
        raw["material"]["sources"][0]["text"] += " changed"
    with pytest.raises(ValueError, match="historical_pending"):
        selected_material(old, state_from(raw), [old.policy.items[0].id])
    assert not f.budget.records()


def historical_window(monkeypatch):
    install_proof(monkeypatch)
    parent, plan = fixture()
    plan["pending_review"]["historical_unknown_origin"] = dict(
        current_acceptance={"owner": "verified"},
        material={"exact_item_observation_sources": True},
        current_config_checksum=digest(parent.config),
    )
    plan["pending_review"]["accepted_ids"] = ["already:a"]
    return parent, plan


def test_new_version_keeps_original_budget_and_one_new_text_dispatch(monkeypatch):
    parent, plan = historical_window(monkeypatch)
    before = deepcopy(parent.protocol), deepcopy(plan["previous_attempts"])
    pending_review.declare_reassessment_window(plan, parent)
    assert plan["schema_version"] == VERSION
    assert plan["max_model_dispatches"] == 1
    assert (parent.protocol, plan["previous_attempts"]) == before
    pending_review.validate_reassessment_window(plan, parent)


@pytest.mark.parametrize("change", ["legacy", "config", "proof", "overlap", "budget"])
def test_historical_scope_cannot_be_smuggled_into_old_version(monkeypatch, change):
    parent, plan = historical_window(monkeypatch)
    pending_review.declare_reassessment_window(plan, parent)
    if change == "legacy":
        plan["schema_version"] = pending_review.REASSESSMENT_WINDOW_VERSION
    elif change == "config":
        plan["pending_review"]["historical_unknown_origin"]["current_config_checksum"] = "changed"
    elif change == "proof":
        plan["pending_review"]["historical_unknown_origin"].pop("material")
    elif change == "overlap":
        plan["pending_review"]["accepted_ids"] += plan["review_target_ids"]
    else:
        plan["max_model_dispatches"] = 2
    with pytest.raises(ValueError):
        pending_review.validate_reassessment_window(plan, parent)
