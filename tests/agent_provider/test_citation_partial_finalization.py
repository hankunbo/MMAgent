"""Only confirmed batches survive an exact citation rejection; no answer rewrite."""

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.partial_finalization import (
    CITATION_ACCEPTANCE_KEY,
    CITATION_GRANT_KEY,
    GRANT_KEY,
    PartialReviewPermission,
    accepted_citation_permission,
)
from packages.grounding.models import digest
from tests.agent_provider.finalization_support import fixture


def prepare(f, **kwargs):
    return PartialReviewPermission.prepare(
        owner=f.owner,
        documents=f.docs,
        requests=f.requests,
        config=f.config,
        clock=lambda: f.now,
        **kwargs,
    )


def test_explicit_policy_preserves_failed_raw_and_only_accepts_confirmed():
    f = fixture(partial=True, citation_rejection=True)
    before, events = f.planner.read(f.docs.run_id), f.budget.journal.read()
    attempt_key = next(k for k in f.docs.saved if ":batch:001:attempt:" in k)
    original_attempt = digest(f.docs.read(attempt_key))
    with pytest.raises(ValueError, match="not_local_binding_rejection"):
        prepare(f)
    assert f.docs.read(GRANT_KEY) is None
    p, sub = prepare(f, citation_rejection=True)
    assert p.grant.schema_version == "partial-review-finalization/1.2.0"
    assert len(sub.batches) == 1 and sub.pending_batch_ids == ("batch:001",)
    assert len(p.grant.request_checksums) == 2 and len(p.grant.rejected_response_checksums) == 1
    a = ResearchAcceptance(owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p)
    with pytest.raises(ValueError, match="original_acceptance_required"):
        accepted_citation_permission(f.state, f.docs, f.requests)
    key = CITATION_ACCEPTANCE_KEY
    assert a.submit(sub, key=key)["status"] == "submitted_not_accepted"
    assert a.accept(key=key).status == "accepted"
    saved = a.read(key=key)
    assert len(saved[2]["scores"]["review_completion"]["pending_ids"]) == 2
    cold = PartialReviewPermission(
        grant=f.docs.read(CITATION_GRANT_KEY)[0],
        config=f.config,
        documents=f.docs,
        requests=f.requests,
        clock=lambda: f.now,
    )
    cold.check(f.state, role="review")
    assert a.accept(key=key).status == "accepted"
    assert a.read(key=key) == saved
    assert accepted_citation_permission(f.state, f.docs, f.requests) == p.grant
    with pytest.raises(ValueError, match="request_not_accepted"):
        cold.check(f.state, role="review", record=f.requests.records()[-1])
    assert digest(f.docs.read(attempt_key)) == original_attempt
    assert f.docs.read("multimodal-case-result") == (f.failed,)
    assert f.planner.read(f.docs.run_id) == before and f.budget.journal.read() == events
    assert len(f.wire.calls) == 2 and not hasattr(cold.budget, "reserve")


@pytest.mark.parametrize("fault", ["unknown", "raw", "source", "binding", "active", "history"])
def test_other_failure_or_changed_binding_not_authorized(fault):
    f = fixture(partial=True, citation_rejection=fault != "binding")
    if fault == "unknown":
        i = max(i for i, e in enumerate(f.budget.journal.events) if e.get("phase") == "completed")
        f.budget.journal.events.pop(i)
    elif fault == "raw":
        key = next(k for k in f.docs.saved if ":batch:001:attempt:" in k)
        f.docs.saved[key][0]["raw_model_text"] += " "
    elif fault == "source":
        f.owner.withdrawn = True
    elif fault == "active":
        f.docs.sessions.state = f.docs.sessions.state.model_copy(update={"phase": "select"})
    elif fault == "history":
        f.docs.write("review-batches:synthetic-event:A:historical", ({"unexpected": True},))
    with pytest.raises(ValueError):
        prepare(f, citation_rejection=True)
    assert f.docs.read(CITATION_GRANT_KEY) is None and len(f.wire.calls) == 2


def test_deadline_does_not_renew_for_citation_closure():
    f = fixture(partial=True, citation_rejection=True)
    p, _ = prepare(f, citation_rejection=True)
    f.now = p.grant.deadline
    with pytest.raises(ValueError, match="deadline"):
        prepare(f, citation_rejection=True)
