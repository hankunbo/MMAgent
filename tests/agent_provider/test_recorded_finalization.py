"""Closed corrected requests can be saved, never resent or edited."""

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.recorded_finalization import GRANT_KEY, RecordedReviewPermission
from tests.agent_provider.finalization_support import fixture


def permission(f):
    return RecordedReviewPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.config, clock=lambda: f.now
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_all_corrected_batches_save_after_deadline_without_requests(dynamic):
    f = fixture(partial=dynamic, corrected=True)
    old, journal = f.planner.read(f.docs.run_id), f.budget.journal.read()
    p, sub = permission(f)
    assert len(sub.batches) == 2 and len(p.grant.request_checksums) == 3
    assert len(p.grant.rejected_response_checksums) == 1 and not hasattr(p.budget, "reserve")
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p
    )
    assert service.submit(sub, key="closed")["status"] == "submitted_not_accepted"
    assert service.accept(key="closed").status == "accepted"
    saved = service.read(key="closed")
    assert service.accept(key="closed").status == "accepted" and service.read(key="closed") == saved
    assert f.planner.read(f.docs.run_id) == old and f.budget.journal.read() == journal
    assert len(f.wire.calls) == 3 and f.docs.read("multimodal-case-result") == (f.failed,)
    with pytest.raises(ValueError, match="not_accepted"):
        p.check(f.state, role="review", record=f.requests.records()[0])
    f.now = p.grant.deadline
    with pytest.raises(ValueError, match="deadline"):
        permission(f)


@pytest.mark.parametrize(
    "fault", ["unknown", "timeout", "feedback", "raw", "incomplete", "active", "source"]
)
def test_wrong_or_incomplete_history_cannot_receive_grant(fault):
    f = fixture(corrected=True)
    if fault in {"unknown", "timeout"}:
        i = max(i for i, e in enumerate(f.budget.journal.events) if e.get("phase") == "completed")
        if fault == "unknown":
            f.budget.journal.events.pop(i)
        else:
            f.budget.journal.events[i]["record"]["error"] = "timeout"
    elif fault == "feedback":
        e = next(
            e
            for e in f.budget.journal.events
            if e.get("schema_version") == "review-schema-correction/1.0.0"
        )
        e["failed_record_checksum"] = "sha256:" + "0" * 64
    elif fault == "raw":
        key = next(
            k
            for k, values in f.docs.saved.items()
            if ":attempt:" in k and values[0].get("error") == "schema"
        )
        f.docs.saved[key][0]["raw_model_text"] += " "
    elif fault == "incomplete":
        del f.docs.saved["review-batches:synthetic-event:A:batch:001:result"]
    elif fault == "active":
        f.docs.sessions.state = f.docs.sessions.state.model_copy(update=dict(phase="select"))
    else:
        f.owner.withdrawn = True
    with pytest.raises(ValueError):
        permission(f)
    assert f.docs.read(GRANT_KEY) is None and len(f.wire.calls) == 3
