"""Read already accepted local batches through the original aggregation owner."""

import pytest

from packages.evidence_details import partial_finalization as closure
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.accepted_aggregation import AcceptedOriginReader, HistoricalRequests
from packages.grounding.models import digest, validate
from tests.agent_provider.finalization_support import fixture
from tests.agent_provider.test_partial_finalization import permission
from tests.expert_agent.test_batch_assembly import LiveDocuments


def accepted(*, cancelled=False):
    f = fixture(
        partial=True, reasoned=True, deadline_partial=True, deadline_cleanup_cancel=cancelled
    )
    # This original fixture's basic Documents only recognizes receipt v1;
    # reuse the established live-receipt inventory, not an unconditional True.
    f.docs.consumed = LiveDocuments.consumed.__get__(f.docs)
    assert f.docs.sessions.read(f.docs.run_id).cancelled is cancelled
    p, sub = permission(f)
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=p
    )
    key = closure.closure_keys(f.docs)[2]
    service.submit(sub, key=key)
    assert service.accept(key=key).status == "accepted"
    raw = f.state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:AGGREGATED"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    current = validate(DomainState, raw)
    reader = AcceptedOriginReader(
        documents=f.docs,
        requests=HistoricalRequests(f.budget.journal),
        state_reader=f.owner.current,
        acceptance_key=key,
    )
    return f, p, sub, current, reader


@pytest.mark.parametrize("cancelled", [False, True])
def test_expired_local_permission_reads_unchanged_accepted_items_without_requests(cancelled):
    f, p, sub, current, reader = accepted(cancelled=cancelled)
    before, journal = dict(f.docs.saved), f.budget.journal.read()
    f.now = p.grant.deadline
    proof, items, records = reader.read(current)
    assert items == sub.candidate.items and len(records) == 1
    assert proof.permission_checksum == digest(p.grant)
    assert proof.unused_request_ids == ()
    assert reader.read(current) == (proof, items, records)
    assert f.docs.saved == before and f.budget.journal.read() == journal
    assert len(f.wire.calls) == 1


@pytest.mark.parametrize(
    "fault", ["grant", "history", "failure", "submission", "unknown", "later_cancel"]
)
def test_partial_origin_rejects_changed_provenance(fault):
    f, _, _, current, reader = accepted()
    if fault == "grant":
        f.docs.saved[closure.DEADLINE_GRANT_KEY][0]["state_checksum"] = digest("changed")
    elif fault == "history":
        f.docs.write("review-batches:synthetic-event:A:historical", ({"changed": True},))
    elif fault == "failure":
        f.docs.saved["multimodal-case-result"][0]["unexpected_change"] = True
    elif fault == "submission":
        f.docs.saved.pop(closure.DEADLINE_SUBMISSION_KEY)
    elif fault == "later_cancel":
        f.docs.sessions.state = f.docs.sessions.state.model_copy(update={"cancelled": True})
    else:
        index = next(
            i for i, row in enumerate(f.budget.journal.events) if row.get("phase") == "completed"
        )
        f.budget.journal.events.pop(index)
    expected = (
        "unconfirmed_request_inventory" if fault == "unknown" else "partial_origin_binding_changed"
    )
    with pytest.raises(ValueError, match=expected):
        reader.read(current)
    assert len(f.wire.calls) == 1


def test_local_result_version_dispatch_is_bound_to_original_saved_result():
    from packages.multimodal_research.accepted_aggregation import local_origin_phase

    f, _, _, _, reader = accepted()
    local = dict(
        schema_version="multimodal-partial-finalization-result/1.1.0",
        original_result_checksum=digest(f.failed),
        review_closure_scope=dict(local_confirmed_only=True, historical_answers_reaccepted=False),
        phases=[dict(run_id=f.docs.run_id, acceptance_key=reader.acceptance_key, accepted=True)],
    )
    f.docs.write(closure.RESULT_KEY, (local,))
    assert local_origin_phase(f.docs, f.failed, local) == local["phases"][-1]
    for changed in (
        dict(local, original_result_checksum=digest("wrong")),
        dict(local, schema_version="unsupported/1.0"),
        dict(local, review_closure_scope=dict(local_confirmed_only=False)),
    ):
        with pytest.raises(ValueError, match="accepted_aggregation_local_finalization"):
            local_origin_phase(f.docs, f.failed, changed)
