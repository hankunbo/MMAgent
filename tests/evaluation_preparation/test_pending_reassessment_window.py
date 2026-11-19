"""Explicit new UNKNOWN batch assessment, not replay or implicit expiry bypass."""

from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from packages.multimodal_research import pending_review as pending
from tests.evaluation_preparation.test_pending_runtime_lease import window


def fixture():
    parent, plan = window()
    now = datetime.fromisoformat(plan["started_at"])
    parent.protocol["deadline"] = (now - timedelta(seconds=1)).isoformat()
    parent.protocol["schema_version"] = "multimodal-campaign-renewal/1.54.0"
    plan.update(
        schema_version="multimodal-campaign-renewal/1.55.0",
        pending_reassessment_window="original_owner_unknown_batch_new_window/1.0.0",
        execution_lease_profile="original-runtime-case-pipeline-session/1.0.0",
        previous_window_deadline=parent.protocol["deadline"],
    )
    plan.pop("original_window_deadline")
    p = plan["pending_review"]
    p.update(
        run_id="original-run",
        journal_checksum="original-journal",
        selected_ids=["unknown:a"],
        acknowledge_unknown_request="old-request",
    )
    p["quarantined_batches"][0].update(
        request_ids=["old-request"],
        new_reassessment_authorized=True,
        old_response_state_unchanged=True,
    )
    plan["review_target_ids"] = ["unknown:a"]
    plan["unknown_response_proof"] = {"proof": "original-owner"}
    return parent, plan


def install_proof(monkeypatch):
    monkeypatch.setattr(pending, "current_unknown_proof", lambda *args: {"proof": "original-owner"})


def test_fresh_window_keeps_original_deadline_budget_and_unknown(monkeypatch):
    install_proof(monkeypatch)
    parent, plan = fixture()
    before = deepcopy(plan), deepcopy(parent.protocol)
    pending.validate_reassessment_window(plan, parent)
    assert (plan, parent.protocol) == before


@pytest.mark.parametrize(
    "change",
    [
        "accepted",
        "unstarted",
        "other_unknown",
        "ack",
        "provider",
        "future_parent",
        "deadline",
        "seconds",
        "retries",
        "dispatch",
        "budget",
        "cumulative",
        "history",
        "vision",
        "search",
        "lease",
        "scope",
        "version",
        "repeat",
        "proof",
        "empty",
    ],
)
def test_reassessment_window_rejects_scope_or_authority_change(monkeypatch, change):
    install_proof(monkeypatch)
    parent, plan = fixture()
    p = plan["pending_review"]
    if change == "accepted":
        p["accepted_ids"].append("unknown:a")
    elif change == "unstarted":
        p["selected_ids"].append("unstarted:b")
        plan["review_target_ids"] = p["selected_ids"]
    elif change == "other_unknown":
        p["quarantined_batches"][0]["request_ids"] = ["different-request"]
    elif change == "ack":
        p["acknowledge_unknown_request"] = None
    elif change == "provider":
        plan["provider"] = "changed"
    elif change == "future_parent":
        parent.protocol["deadline"] = plan["deadline"]
        plan["previous_window_deadline"] = plan["deadline"]
    elif change == "deadline":
        plan["deadline"] = "2026-10-28T00:00:00+00:00"
    elif change == "seconds":
        plan["seconds"] += 1
    elif change == "retries":
        parent.config = parent.config.model_copy(
            update={"transport": parent.config.transport.model_copy(update={"max_retries": 1})}
        )
    elif change == "dispatch":
        plan["max_model_dispatches"] = 2
    elif change == "budget":
        plan["max_new_requests"] = 8
    elif change == "cumulative":
        plan["previous_attempts"] = [dict(requests=39)]
    elif change == "history":
        plan["previous_attempts"] = []
    elif change == "vision":
        plan["new_vision_requests"] = 1
    elif change == "search":
        plan["new_public_search_allowed"] = True
    elif change == "lease":
        plan["execution_lease_profile"] = "unchecked"
    elif change == "scope":
        plan["mode"] = "acquisition"
    elif change == "version":
        plan["schema_version"] = "multimodal-campaign-renewal/1.54.0"
    elif change == "repeat":
        parent.protocol["pending_review"] = deepcopy(p)
    elif change == "proof":
        plan["unknown_response_proof"] = {"proof": "different"}
    elif change == "empty":
        p["selected_ids"] = []
        plan["review_target_ids"] = []
    with pytest.raises(ValueError):
        pending.validate_reassessment_window(plan, parent)


def test_readonly_proof_changed_after_preparation_rejected(monkeypatch):
    install_proof(monkeypatch)
    parent, plan = fixture()
    pending.validate_reassessment_window(plan, parent)
    monkeypatch.setattr(pending, "current_unknown_proof", lambda *args: {"proof": "changed"})
    with pytest.raises(ValueError, match="reassessment_window"):
        pending.validate_reassessment_window(plan, parent)


def test_cli_reassessment_scope_rejected_before_owner_access():
    from packages.multimodal_research.campaign_renewal import prepare

    with pytest.raises(ValueError, match="pending_reassessment_explicit_scope"):
        prepare(
            facts="unused",
            previous="unused",
            output="unused",
            authorization="test",
            mode="acquisition",
            pending_reassessment_window=True,
        )


def test_original_request_proof_no_rewrite_and_confirmed_response_rejected(monkeypatch, tmp_path):
    from packages.evidence_details.pending_review import unknown_response_proof
    from packages.grounding.models import digest
    from tests.expert_agent.test_pending_failed_attempt import failed

    fixture, _ = failed(monkeypatch, tmp_path)
    journal = fixture.ports.budget.journal
    before = deepcopy(journal.read())
    request = next(e["record"]["request_id"] for e in before if e.get("phase") == "completed")
    proof = unknown_response_proof(fixture.documents, request, digest(before))
    assert proof["duplicate_remote_cost_possible"] and journal.read() == before
    changed = list(deepcopy(before))
    changed.append(dict(phase="response_confirmation", request_id=request, confirmed=True))
    from types import SimpleNamespace

    monkeypatch.setattr(
        "packages.evidence_details.pending_review.RuntimeAgentJournal",
        lambda _: SimpleNamespace(read=lambda: changed),
    )
    with pytest.raises(ValueError, match="requires_different_recovery"):
        unknown_response_proof(fixture.documents, request, digest(changed))
