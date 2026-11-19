"""Original partial acceptance -> exact pending partition, no fabricated scores."""

from types import SimpleNamespace

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.pending_review import read_pending_review
from packages.evidence_details.review_batches import collect_partial
from packages.grounding.models import digest
from tests.agent_provider.test_review_continuation import continuation_fixture


def partial_fixture(monkeypatch):
    f = continuation_fixture(format_failure=True, single_batches=True)
    old = f.old
    sub = collect_partial(old.owner.current(), old.documents, "origin:A")
    service = ResearchAcceptance(owner=old.owner, documents=old.documents, requests=old.budget)
    service.submit(sub, key="partial")
    assert service.accept(key="partial").status == "accepted"
    # The CPU fixture uses the original in-memory journal implementation.
    monkeypatch.setattr(
        "packages.evidence_details.pending_review.RuntimeAgentJournal", lambda _: old.budget.journal
    )
    return f, sub


def test_partial_acceptance_selects_only_never_sent_and_keeps_original(monkeypatch):
    f, sub = partial_fixture(monkeypatch)
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    value = read_pending_review(
        documents=f.old.documents, state=f.old.owner.current(), acceptance_key="partial"
    )
    assert value["accepted_ids"] == sorted(i.assessment_id for i in sub.candidate.items)
    assert value["selected_ids"] == sorted(
        i for b in sub.plan.batches[3:] for i in b.assessment_ids
    )
    assert value["quarantined_batches"][0]["batch_id"] == sub.plan.batches[2].id
    assert not value["quarantined_batches"][0]["new_reassessment_authorized"]
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))
    assert (
        read_pending_review(
            documents=f.old.documents, state=f.old.owner.current(), acceptance_key="partial"
        )
        == value
    )


@pytest.mark.parametrize("change", ["score", "claim", "context", "cancel", "ack"])
def test_pending_original_binding_fail_closed(monkeypatch, change):
    f, sub = partial_fixture(monkeypatch)
    old = f.old
    if change == "score":
        old.documents.saved["accepted:partial"][2]["scores"]["checksum"] = digest("wrong")
    elif change == "claim":
        old.documents.saved["review-batches:origin:A:batch:002:claim"][0]["input_checksum"] = (
            digest("wrong")
        )
    elif change == "cancel":
        control = old.documents.sessions.read(old.documents.run_id)
        old.documents.sessions = SimpleNamespace(
            read=lambda _: control.model_copy(update={"cancelled": True})
        )
    state = old.owner.current()
    if change == "context":
        state = state.model_copy(update={"config_checksum": digest("changed")})
    with pytest.raises(ValueError):
        read_pending_review(
            documents=old.documents,
            state=state,
            acceptance_key="partial",
            acknowledge_unknown_request="request:wrong" if change == "ack" else None,
        )


def test_pending_increment_excludes_memory_whole_case_archival():
    from packages.multimodal_research.memory import archive

    assert archive(SimpleNamespace(protocol={"review_increment_only": True}), {}) == dict(
        status="not_archived_pending_review_increment", eligible_for_planning=False
    )
