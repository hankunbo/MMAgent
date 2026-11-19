"""Narrow adapter tests; real original-owner evidence is separately recorded."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.expert_agent import committed_review
from packages.evidence_details import recorded_finalization
from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from tests.evaluation_preparation.test_directional_review import fixture


@pytest.fixture
def closure(monkeypatch):
    f = fixture()
    state = f.owner.current()
    rid = state.material.scope.run_id
    phase = dict(run_id=rid, target=None, accepted=False, acceptance_key=None, error="deadline")
    old = dict(complete=False, phases=[phase], new_requests=3)
    proof = dict(acceptance_key="original:domain", parent_checksum=digest(state))
    saved = dict(
        schema_version="multimodal-recorded-finalization-result/1.2.0",
        complete=True,
        original_requests=3,
        new_requests=0,
        original_result_checksum=digest(old),
        local_review_closure=dict(confirmation=proof),
        phases=[
            dict(
                phase,
                accepted=True,
                acceptance_key="original:domain",
                error=None,
                evidence_limited=True,
            )
        ],
    )
    values = {
        recorded_finalization.RESULT_KEY: (saved,),
        "dynamic-phase-input": (state.model_dump(mode="json"),),
    }
    docs = SimpleNamespace(read=lambda key: values.get(key))
    parent = SimpleNamespace(documents=lambda _: docs, config=f.config)
    monkeypatch.setattr(recorded_finalization, "original_failure", lambda _: old)

    def owner_read(documents, *, parent, config):
        assert documents is docs and parent == state and config == f.config
        return proof

    monkeypatch.setattr(committed_review, "read_committed_review", owner_read)
    return SimpleNamespace(parent=parent, run_id=rid, values=values, saved=saved, old=old)


def test_exact_original_commit_and_later_closure_are_bound_without_mutation(closure):
    before = deepcopy(closure.values)
    proof = renewal.recorded_acquisition_origin(closure.parent, closure.run_id)
    assert proof["closure_checksum"] == digest(closure.saved)
    assert proof["original_result_checksum"] == digest(closure.old)
    assert before == closure.values and closure.old["complete"] is False


@pytest.mark.parametrize("change", ["original", "phase", "requests", "missing"])
def test_wrong_closure_binding_rejected(closure, change):
    if change == "original":
        closure.saved["original_result_checksum"] = digest("wrong")
    elif change == "phase":
        closure.saved["phases"][0]["acceptance_key"] = "other:domain"
    elif change == "requests":
        closure.saved["new_requests"] = 1
    else:
        closure.values.pop(recorded_finalization.RESULT_KEY)
    with pytest.raises(ValueError, match="renewal_recorded_closure"):
        renewal.recorded_acquisition_origin(closure.parent, closure.run_id)


@pytest.fixture
def local_closure(closure, monkeypatch):
    """CLI shape only; original grant/receipt proof is tested separately on PG."""
    calls = []
    closure.parent.base = SimpleNamespace(check=lambda: calls.append("current"))
    proof = dict(
        schema_version="recorded-review-acceptance-proof/1.0.0",
        run_id=closure.run_id,
        acceptance_key="recorded-partition-finalization:1",
        grant_checksum=digest("original local grant"),
    )
    closure.saved.update(
        schema_version="multimodal-recorded-finalization-result/1.1.0",
        finalization_grant_checksum=proof["grant_checksum"],
    )
    closure.saved.pop("local_review_closure")
    closure.saved["phases"][0]["acceptance_key"] = proof["acceptance_key"]
    monkeypatch.setattr(recorded_finalization, "RecordedBatchRequests", lambda journal: journal)

    def original_owner(**kwargs):
        state = kwargs["owner"].current()
        assert state.config_checksum == digest(kwargs["config"])
        calls.append("saved-proof")
        return proof

    monkeypatch.setattr(recorded_finalization, "read_saved_acceptance", original_owner)
    closure.calls = calls
    return closure


def test_local_grant_route_binds_original_owner_proof_and_new_origin_version(local_closure):
    f = local_closure
    before = deepcopy(f.values)
    proof = renewal.recorded_acquisition_origin(f.parent, f.run_id)
    assert proof["schema_version"] == "renewal-recorded-origin/1.1.0"
    assert proof["confirmation"]["acceptance_key"] == "recorded-partition-finalization:1"
    assert f.calls == ["current", "saved-proof"]
    assert before == f.values


def test_reader_state_checks_identity_without_recursively_rebuilding_proof(
    local_closure, monkeypatch
):
    from packages.evidence_details.accepted_review_reuse import RecordedAcceptedReviewReader

    f = local_closure
    origin = renewal.recorded_acquisition_origin(f.parent, f.run_id)
    monkeypatch.setattr(
        renewal,
        "recorded_acquisition_origin",
        lambda *args: pytest.fail("full proof belongs to public current(), not state_reader"),
    )
    reader = renewal.accepted_reader(f.parent, f.run_id, recorded_origin=origin)
    assert isinstance(reader, RecordedAcceptedReviewReader)
    state = reader.state_reader()
    assert digest(state) == digest(f.values["dynamic-phase-input"][0])
    assert f.calls == ["current", "saved-proof", "current"]
    f.saved["new_requests"] = 1
    with pytest.raises(ValueError, match="renewal_recorded_origin_changed"):
        reader.state_reader()


@pytest.mark.parametrize("fault", ["grant", "phase", "original", "requests"])
def test_local_closure_cannot_replace_owner_bound_origin(local_closure, fault):
    f = local_closure
    if fault == "grant":
        f.saved["finalization_grant_checksum"] = digest("wrong")
    elif fault == "phase":
        f.saved["phases"][0]["acceptance_key"] = "other"
    elif fault == "original":
        f.saved["original_result_checksum"] = digest("wrong")
    else:
        f.saved["new_requests"] = 1
    with pytest.raises(ValueError, match="renewal_recorded_local_closure_binding"):
        renewal.recorded_acquisition_origin(f.parent, f.run_id)


@pytest.mark.parametrize("corrupt", [False, True])
def test_recorded_ancestor_uses_accepted_fact_not_later_stopped_request(local_closure, corrupt):
    from packages.evidence_details.accepted_review_reuse import RecordedAcceptedReviewReader

    f = local_closure
    proof = renewal.recorded_acquisition_origin(f.parent, f.run_id)
    if corrupt:
        proof["confirmation"]["run_id"] = "RUN:WRONG"
    tip_docs = SimpleNamespace(
        read=lambda key: (
            ({"accepted": True, "target": None, "acceptance_key": "current"},)
            if key == "dynamic-phase-result"
            else None
        )
    )
    tip = SimpleNamespace(
        parent=f.parent,
        config=f.parent.config,
        documents=lambda rid: tip_docs,
        protocol=dict(
            schema_version="multimodal-campaign-renewal/1.51.0",
            accepted_fact_run_id=f.run_id,
            recorded_review_origin=proof,
            tip=dict(run_id="RUN:LATER_STOPPED_OBSERVATION"),
        ),
    )
    reader = renewal.accepted_reader(tip, "RUN:CURRENT")
    if corrupt:
        with pytest.raises(ValueError, match="renewal_recorded_origin_changed"):
            reader.ancestor_reader(f.run_id, ())
    else:
        ancestor = reader.ancestor_reader(f.run_id, ())
        assert isinstance(ancestor, RecordedAcceptedReviewReader)
        assert digest(ancestor.state_reader()) == digest(f.values["dynamic-phase-input"][0])


@pytest.mark.parametrize(
    "fault",
    [None, "missing_owner", "wrong_closure", "wrong_original", "missing_saved", "wrong_run"],
)
def test_aggregation_consumes_only_original_owner_confirmed_closure(closure, monkeypatch, fault):
    from packages.multimodal_research.accepted_aggregation import local_origin_phase

    f = closure
    docs = f.parent.documents(f.run_id)
    docs.run_id = f.run_id
    proof = renewal.recorded_acquisition_origin(f.parent, f.run_id)
    if fault == "wrong_closure":
        proof["closure_checksum"] = digest("other closure")
    if fault == "wrong_original":
        proof["original_result_checksum"] = digest("other failed result")
    if fault == "missing_saved":
        f.values.pop(recorded_finalization.RESULT_KEY)
    if fault == "wrong_run":
        docs.run_id = "RUN:WRONG"
    calls = []

    def verified_owner(campaign, run_id):
        assert campaign is f.parent
        calls.append(run_id)
        return proof

    monkeypatch.setattr(renewal, "recorded_acquisition_origin", verified_owner)
    campaign = None if fault == "missing_owner" else f.parent
    if fault:
        with pytest.raises(
            ValueError,
            match="accepted_aggregation_.*binding|accepted_aggregation_committed_origin_owner_required",
        ):
            local_origin_phase(docs, f.old, f.saved, campaign=campaign)
    else:
        before = deepcopy(f.values)
        assert local_origin_phase(docs, f.old, f.saved, campaign=campaign) == f.saved["phases"][-1]
        assert f.values == before
        assert calls == [f.run_id]
