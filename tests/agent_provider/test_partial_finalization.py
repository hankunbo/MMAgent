"""Actual batch rejection at an injected HTTP boundary, not edited fake receipts."""

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.partial_finalization import GRANT_KEY, PartialReviewPermission
from tests.agent_provider.finalization_support import fixture


def permission(f):
    return PartialReviewPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.config, clock=lambda: f.now
    )


def test_partial_save_keeps_failure_control_and_all_confirmed_batches():
    f = fixture(partial=True)
    before, events = f.planner.read(f.docs.run_id), f.budget.journal.read()
    p, sub = permission(f)
    assert len(sub.batches) == 1 and sub.pending_batch_ids == ("batch:001",)
    assert len(p.grant.request_checksums) == 2
    assert not hasattr(p.budget, "reserve")
    a = ResearchAcceptance(owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p)
    assert a.submit(sub, key="partial-close")["status"] == "submitted_not_accepted"
    assert a.accept(key="partial-close").status == "accepted"
    saved = a.read(key="partial-close")
    assert len(saved[2]["scores"]["review_completion"]["pending_ids"]) == 2
    assert a.accept(key="partial-close").status == "accepted"
    assert a.read(key="partial-close") == saved
    assert f.docs.read("multimodal-case-result") == (f.failed,)
    assert f.planner.read(f.docs.run_id) == before and f.budget.journal.read() == events
    assert len(f.wire.calls) == 2


@pytest.mark.parametrize("fault", ["unknown", "timeout", "raw", "active", "source"])
def test_unconfirmed_or_changed_facts_reject_without_grant(fault):
    f = fixture(partial=True)
    if fault in ("unknown", "timeout"):
        index = max(
            i for i, e in enumerate(f.budget.journal.events) if e.get("phase") == "completed"
        )
        if fault == "unknown":
            f.budget.journal.events.pop(index)
        else:
            f.budget.journal.events[index]["record"]["error"] = "timeout"
    elif fault == "raw":
        key = next(k for k in f.docs.saved if ":batch:001:attempt:" in k)
        f.docs.saved[key][0]["raw_model_text"] += " "
    elif fault == "active":
        f.docs.sessions.state = f.docs.sessions.state.model_copy(update=dict(phase="select"))
    else:
        f.owner.withdrawn = True
    with pytest.raises(ValueError):
        permission(f)
    assert f.docs.read(GRANT_KEY) is None and len(f.wire.calls) == 2


def test_deadline_cannot_refresh_or_accept_rejected_request():
    f = fixture(partial=True)
    p, _ = permission(f)
    with pytest.raises(ValueError, match="request_not_accepted"):
        p.check(f.state, role="review", record=f.requests.records()[-1])
    p.clock = lambda: p.grant.deadline
    with pytest.raises(ValueError, match="deadline"):
        p.check(f.state, role="review")
    f.now = p.grant.deadline
    with pytest.raises(ValueError, match="deadline"):
        permission(f)
