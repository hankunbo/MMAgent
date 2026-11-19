"""Original acceptance owners, one fresh reconstruction and late-bound guards."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.accepted_aggregation import ACCEPTANCE_KEY, AggregationPermission
from tests.expert_agent.test_accepted_aggregation import accepted_origin, aggregate_fixture


def prepared(monkeypatch, tmp_path):
    original = accepted_origin(monkeypatch, tmp_path)
    a = aggregate_fixture(original)
    permission, submission = AggregationPermission.prepare(
        owner=a.owner, documents=a.documents, requests=a.requests, config=original.config
    )
    service = ResearchAcceptance(
        owner=a.owner, documents=a.documents, requests=a.requests, live_permission=permission
    )
    return original, a, permission, submission, service


def test_single_validation_plus_fresh_acceptance_provenance_no_cross_call_cache(
    monkeypatch, tmp_path
):
    original, a, _, submission, service = prepared(monkeypatch, tmp_path)
    current, calls = a.source.current, []

    def counted(state):
        calls.append(state)
        return current(state)

    monkeypatch.setattr(a.source, "current", counted)
    old = dict(original.documents.saved)
    assert service.submit(submission, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    assert len(calls) == 1
    assert service.accept(key=ACCEPTANCE_KEY).status == "accepted"
    assert len(calls) == 3
    audit = list(a.documents.audit)
    assert service.accept(key=ACCEPTANCE_KEY).status == "accepted"
    assert len(calls) == 3 and a.documents.audit == audit
    assert original.documents.saved == old
    assert len(original.wire.calls) == 1 and not a.requests.journal.read()


def test_expired_permission_stops_before_expensive_proof(monkeypatch, tmp_path):
    _, a, permission, submission, service = prepared(monkeypatch, tmp_path)
    permission.clock = lambda: permission.grant.deadline

    def forbidden(_):
        raise AssertionError("expired grant must not reconstruct all origins")

    monkeypatch.setattr(a.source, "current", forbidden)
    receipt = service.submit(submission, key=ACCEPTANCE_KEY)
    assert receipt.status == "rejected"
    assert not a.requests.journal.read()


@pytest.mark.parametrize("fault", ["deadline", "cancel"])
def test_final_provenance_read_cannot_outlive_local_permission(monkeypatch, tmp_path, fault):
    _, a, permission, submission, service = prepared(monkeypatch, tmp_path)
    assert service.submit(submission, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    original = a.requests.records
    control = a.documents.sessions.read(a.documents.run_id)

    def after_real_provenance():
        values = original()
        if fault == "deadline":
            permission.clock = lambda: permission.grant.deadline
        else:
            a.documents.sessions.read = lambda _: control.model_copy(update={"cancelled": True})
        return values

    monkeypatch.setattr(a.requests, "records", after_real_provenance)
    result = service.accept(key=ACCEPTANCE_KEY)
    assert result.status == "rejected" and result.scores_checksum is None
    saved = a.documents.read("accepted:" + ACCEPTANCE_KEY)
    assert len(saved) == 1 and saved[0]["status"] == "rejected"
    assert not a.requests.journal.read()


@pytest.mark.parametrize("fault", ["deadline", "control", "changed_candidate", "foreign_owner"])
def test_same_current_owner_proof_keeps_each_boundary(monkeypatch, tmp_path, fault):
    _, a, permission, submission, _ = prepared(monkeypatch, tmp_path)
    current = a.source.current
    clock = [permission.grant.issued_at + timedelta(seconds=1)]
    permission.clock = lambda: clock[0]
    control = a.documents.sessions.read(a.documents.run_id)

    def changes_after_original(state):
        result = current(state)
        if fault == "deadline":
            clock[0] = permission.grant.deadline
        elif fault == "control":
            a.documents.sessions.read = lambda _: control.model_copy(update={"cancelled": True})
        return result

    monkeypatch.setattr(a.source, "current", changes_after_original)
    if fault == "changed_candidate":
        submission = submission.model_copy(update={"source_checksums": ()})
    elif fault == "foreign_owner":
        a.owner.accepted_review_set = SimpleNamespace(current=current)
    with pytest.raises(ValueError, match="accepted_aggregation_"):
        permission.check_submission(a.state, submission)
    assert not a.requests.journal.read()
    assert a.documents.read("accepted:" + ACCEPTANCE_KEY) is None
