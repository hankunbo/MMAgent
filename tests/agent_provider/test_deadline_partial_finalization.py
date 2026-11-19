"""A real parsed batch commits before deadline; no next request may be issued."""

import pytest

from packages.evidence_details import partial_finalization as closure
from packages.evidence_details.acceptance import ResearchAcceptance
from tests.agent_provider.finalization_support import fixture
from tests.agent_provider.test_partial_finalization import permission


def test_deadline_partial_retains_every_confirmed_local_answer_without_resending():
    f = fixture(partial=True, reasoned=True, deadline_partial=True)
    before, events = f.planner.read(f.docs.run_id), f.budget.journal.read()
    p, sub = permission(f)
    assert p.grant.schema_version == "partial-review-finalization/1.1.0"
    assert p.grant.rejected_response_checksums == ()
    assert sub.pending_batch_ids and len(f.wire.calls) == 1
    assert not hasattr(p.budget, "reserve")
    a = ResearchAcceptance(owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p)
    key = closure.closure_keys(f.docs)[2]
    assert a.submit(sub, key=key)["status"] == "submitted_not_accepted"
    assert a.accept(key=key).status == "accepted"
    saved = a.read(key=key)
    assert saved[2]["scores"]["review_completion"]["pending_ids"]
    assert a.accept(key=key).status == "accepted" and a.read(key=key) == saved
    assert f.docs.read("multimodal-case-result") == (f.failed,)
    assert f.planner.read(f.docs.run_id) == before and f.budget.journal.read() == events
    assert len(f.wire.calls) == 1
    f.now = p.grant.deadline
    with pytest.raises(ValueError, match="deadline"):
        permission(f)


@pytest.mark.parametrize("fault", ["unknown", "timeout", "active", "source", "claim", "history"])
def test_deadline_partial_refuses_unknown_stale_or_changed_provenance(fault):
    f = fixture(partial=True, reasoned=True, deadline_partial=True)
    if fault in {"unknown", "timeout"}:
        i = next(i for i, e in enumerate(f.budget.journal.events) if e.get("phase") == "completed")
        if fault == "unknown":
            f.budget.journal.events.pop(i)
        else:
            f.budget.journal.events[i]["record"]["error"] = "timeout"
    elif fault == "active":
        f.docs.sessions.state = f.docs.sessions.state.model_copy(update={"phase": "select"})
    elif fault == "source":
        f.owner.withdrawn = True
    elif fault == "claim":
        f.docs.write("review-batches:synthetic-event:A:batch:001:claim", ({"unknown": True},))
    else:
        p, _ = permission(f)
        f.docs.write("review-batches:synthetic-event:A:historical", ({"changed": True},))
        with pytest.raises(ValueError, match="historical"):
            p.check(f.state, role="review")
        return
    with pytest.raises(ValueError):
        permission(f)
    assert f.docs.read(closure.DEADLINE_GRANT_KEY) is None and len(f.wire.calls) == 1
