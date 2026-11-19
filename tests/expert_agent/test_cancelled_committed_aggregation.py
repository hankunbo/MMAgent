"""Narrow cancelled-origin adapter contract; persistent commit checked separately."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.evidence_details.accepted_aggregation import (
    CommittedClosureOriginReader,
    HistoricalRequests,
)
from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal
from tests.expert_agent.test_accepted_aggregation import (
    ORIGINAL_KEY,
    accepted_origin,
    aggregate_fixture,
)


@pytest.fixture
def committed(monkeypatch, tmp_path):
    f = accepted_origin(monkeypatch, tmp_path)
    a = aggregate_fixture(f)
    control = f.documents.sessions.read(f.documents.run_id).model_copy(update={"cancelled": True})
    monkeypatch.setattr(f.documents.sessions, "read", lambda _: control)
    campaign = SimpleNamespace()
    proof = dict(
        schema_version="renewal-recorded-origin/1.0.0",
        closure_checksum=digest("original-closure"),
        confirmation=dict(
            run_id=f.documents.run_id,
            acceptance_key=ORIGINAL_KEY,
            parent_checksum=digest(f.state),
            accepted_checksum=digest(f.documents.read("accepted:" + ORIGINAL_KEY)),
            journal_checksum=digest(f.ports.budget.journal.read()),
        ),
    )
    calls = []

    def original(c, rid):
        assert c is campaign and rid == f.documents.run_id
        calls.append(rid)
        return proof

    monkeypatch.setattr(campaign_renewal, "recorded_acquisition_origin", original)
    reader = CommittedClosureOriginReader(
        campaign=campaign,
        closure_checksum=proof["closure_checksum"],
        documents=f.documents,
        requests=HistoricalRequests(f.ports.budget.journal),
        state_reader=lambda: f.state,
        acceptance_key=ORIGINAL_KEY,
    )
    return SimpleNamespace(f=f, a=a, reader=reader, proof=proof, calls=calls, control=control)


def test_default_reader_still_refuses_cancelled_normal_key(committed):
    with pytest.raises(ValueError, match="accepted_aggregation_origin_not_terminal"):
        committed.a.reader.read(committed.a.state)
    assert not committed.calls


def test_closure_cannot_bypass_other_original_permission_checks(committed):
    f = committed.f
    before = deepcopy(f.documents.saved)
    journal = f.ports.budget.journal.read()
    # This fixture used a local ExplanationGrant bound to its prior Control.
    # The separate closure does not authorize ignoring that original binding.
    with pytest.raises(ValueError, match="accepted_aggregation_local_origin_permission"):
        committed.reader.read(committed.a.state)
    assert f.documents.saved == before and f.ports.budget.journal.read() == journal
    assert len(f.wire.calls) == 1 and committed.calls


@pytest.mark.parametrize(
    "field",
    ["run_id", "acceptance_key", "parent_checksum", "accepted_checksum", "journal_checksum"],
)
def test_wrong_confirmation_cannot_authorize_reuse(committed, field):
    committed.proof["confirmation"][field] = "wrong"
    with pytest.raises(ValueError, match="accepted_aggregation_committed_closure_binding"):
        committed.reader.read(committed.a.state)


@pytest.mark.parametrize("field", ["closure_checksum", "schema_version"])
def test_wrong_closure_or_different_local_acceptance_route_refused(committed, field):
    committed.proof[field] = "wrong"
    with pytest.raises(ValueError, match="accepted_aggregation_committed_closure_binding"):
        committed.reader.read(committed.a.state)


def test_original_owner_unknown_or_missing_commit_is_not_overridden(committed, monkeypatch):
    def reject(*args):
        raise ValueError("committed_review_unknown_request")

    monkeypatch.setattr(campaign_renewal, "recorded_acquisition_origin", reject)
    with pytest.raises(ValueError, match="committed_review_unknown_request"):
        committed.reader.read(committed.a.state)


@pytest.mark.parametrize("change", ["phase", "pending", "after_proof"])
def test_control_must_be_terminal_and_still_current(committed, monkeypatch, change):
    old = committed.control
    current = old.model_copy(update={"phase": "review"})
    if change == "pending":
        current = old.model_copy(update={"pending": object()})
    if change == "after_proof":
        calls = iter((old, current))
        monkeypatch.setattr(committed.f.documents.sessions, "read", lambda _: next(calls))
    else:
        monkeypatch.setattr(committed.f.documents.sessions, "read", lambda _: current)
    with pytest.raises(
        ValueError, match="accepted_aggregation_(origin_not_terminal|committed_closure_binding)"
    ):
        committed.reader.read(committed.a.state)
