from datetime import timedelta

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.finalization import GRANT_KEY
from packages.grounding.models import digest
from tests.agent_provider.finalization_support import fixture, permission


def test_original_deadline_stays_closed_but_explicit_fact_finalization_is_idempotent():
    f = fixture()
    with pytest.raises(ValueError, match="deadline"):
        f.old.check(f.state, role="review", record=f.requests.records()[0])
    original_control = f.planner.read(f.docs.run_id)
    journal = tuple(f.budget.journal.read())
    p, sub = permission(f)
    assert sub == f.submission and not hasattr(p.budget, "reserve")
    acceptance = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p
    )
    assert acceptance.submit(sub, key="finalize")["status"] == "submitted_not_accepted"
    assert acceptance.accept(key="finalize").status == "accepted"
    saved = acceptance.read(key="finalize")
    assert saved[0]["candidate"] == sub.candidate.model_dump(mode="json")
    assert acceptance.accept(key="finalize").status == "accepted"
    assert f.docs.read("dynamic-case-result") == (f.failed,)
    assert f.planner.read(f.docs.run_id) == original_control
    assert tuple(f.budget.journal.read()) == journal and len(f.wire.calls) == 2


@pytest.mark.parametrize("change", ["unknown", "error", "aggregate", "failure"])
def test_incomplete_or_changed_originals_cannot_be_finalized(change):
    f = fixture()
    completed = next(
        n
        for n in range(len(f.budget.journal.events) - 1, -1, -1)
        if f.budget.journal.events[n].get("phase") == "completed"
    )
    if change == "unknown":
        f.budget.journal.events.pop(completed)
    elif change == "error":
        f.budget.journal.events[completed]["record"]["error"] = "timeout"
    elif change == "aggregate":
        f.docs.saved["review-batches:synthetic-event:A:aggregate"][0]["candidate"]["items"][0][
            "reason"
        ] = "forged"
    else:
        f.docs.saved["dynamic-case-result"][0]["phases"][0]["error"] = "cancelled"
    with pytest.raises(ValueError):
        permission(f)
    assert f.docs.read(GRANT_KEY) is None and len(f.wire.calls) == 2


def test_new_role_changed_state_expiry_and_source_withdrawal_rejected():
    f = fixture()
    p, sub = permission(f)
    with pytest.raises(ValueError, match="finalization_scope"):
        p.check(f.state, role="vision")
    with pytest.raises(ValueError, match="finalization_scope"):
        p.check(f.state.model_copy(update={"config_checksum": digest("changed")}), role="review")
    p.clock = lambda: p.grant.deadline
    with pytest.raises(ValueError, match="finalization_deadline"):
        p.check(f.state, role="review")
    p.clock = lambda: f.now
    f.owner.withdrawn = True
    a = ResearchAcceptance(owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p)
    with pytest.raises(ValueError, match="source_withdrawn"):
        a.submit(sub, key="withdrawn")
    assert f.docs.read("accepted:withdrawn") is None


def test_expired_finalization_does_not_refresh_its_deadline():
    f = fixture()
    p, _ = permission(f)
    f.now = p.grant.deadline + timedelta(seconds=1)
    with pytest.raises(ValueError, match="finalization_deadline"):
        permission(f)
    assert f.docs.read(GRANT_KEY)[0]["deadline"] == p.grant.model_dump(mode="json")["deadline"]
