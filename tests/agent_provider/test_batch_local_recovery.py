"""Original transport/parser/owner response, interrupted before local result save."""

import pytest

from packages.evidence_details import partial_finalization as closure
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.batch_local_recovery import KEY, recover
from tests.agent_provider.finalization_support import fixture
from tests.agent_provider.test_partial_finalization import permission


def raw_fixture(**kwargs):
    return fixture(
        partial=True, reasoned=True, deadline_partial=True, raw_only_deadline=True, **kwargs
    )


def run(f):
    return recover(owner=f.owner, documents=f.docs, requests=f.requests)


def test_confirmed_raw_recovered_once_then_original_acceptance_without_dispatch():
    f = raw_fixture()
    before, events, original = f.planner.read(f.docs.run_id), f.budget.journal.read(), f.failed
    assert f.docs.read("review-batches:synthetic-event:A:batch:000:result") is None
    with pytest.raises(ValueError):
        permission(f)
    receipt = run(f)
    assert receipt["new_api_requests"] == 0 and receipt["new_score"] is False
    assert receipt["accepted_medical_evidence"] is False
    after = dict(f.docs.saved)
    assert run(f) == receipt and f.docs.saved == after
    p, sub = permission(f)
    a = ResearchAcceptance(owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p)
    key = closure.closure_keys(f.docs)[2]
    a.submit(sub, key=key)
    assert a.accept(key=key).status == "accepted"
    saved = a.read(key=key)
    assert saved[2]["scores"]["review_completion"]["pending_ids"]
    assert run(f) == receipt
    assert f.planner.read(f.docs.run_id) == before and f.budget.journal.read() == events
    assert f.docs.read("multimodal-case-result") == (original,) and len(f.wire.calls) == 1


@pytest.mark.parametrize(
    "fault", ["unknown", "raw", "active", "source", "parent", "claim", "intent"]
)
def test_missing_or_changed_binding_never_creates_a_result(fault):
    f = raw_fixture()
    if fault == "unknown":
        i = next(i for i, e in enumerate(f.budget.journal.events) if e.get("phase") == "completed")
        f.budget.journal.events.pop(i)
    elif fault == "raw":
        key = next(k for k in f.docs.saved if k.endswith(":batch:000:response"))
        f.docs.saved[key][0]["raw_model_text"] += " "
    elif fault == "active":
        f.docs.sessions.state = f.docs.sessions.state.model_copy(update={"phase": "select"})
    elif fault == "source":
        f.owner.withdrawn = True
    elif fault == "parent":
        f.docs.saved["dynamic-phase-input"] = ({"wrong": True},)
    elif fault == "claim":
        f.docs.saved["review-batches:synthetic-event:A:batch:001:claim"] = ({"unknown": True},)
    else:
        f.docs.saved["live-stage-intent:review"] = ({"event": {"code": "other"}},)
    before = dict(f.docs.saved)
    with pytest.raises(ValueError):
        run(f)
    assert f.docs.saved == before and f.docs.read(KEY) is None and len(f.wire.calls) == 1


def test_owner_changes_on_return_no_partial_local_write():
    f = raw_fixture()
    original, calls = f.owner.current, []

    def current():
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("source_withdrawn")
        return original()

    f.owner.current = current
    before = dict(f.docs.saved)
    with pytest.raises(ValueError, match="source_withdrawn"):
        run(f)
    assert f.docs.saved == before and len(f.wire.calls) == 1
