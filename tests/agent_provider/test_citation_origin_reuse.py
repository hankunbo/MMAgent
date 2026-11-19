"""A saved citation closure is provenance, not permission to repeat its rejection."""

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.accepted_aggregation import AcceptedOriginReader
from packages.evidence_details.partial_finalization import CITATION_ACCEPTANCE_KEY
from packages.grounding.models import digest, validate
from packages.multimodal_research.accepted_aggregation import local_origin_phase
from tests.agent_provider.finalization_support import fixture
from tests.agent_provider.test_citation_partial_finalization import prepare
from tests.expert_agent.test_batch_assembly import LiveDocuments


def current_probe(state):
    raw = state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:CITATION_CURRENT_PROOF"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    return validate(DomainState, raw)


def accepted(monkeypatch):
    # The minimal agent Documents double intentionally never marks a graph as
    # consumed. Reuse the original acceptance-aware test port, not a bypass.
    monkeypatch.setattr("tests.agent_provider.finalization_support.Documents", LiveDocuments)
    f = fixture(partial=True, citation_rejection=True)
    permission, sub = prepare(f, citation_rejection=True)
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    service.submit(sub, key=CITATION_ACCEPTANCE_KEY)
    assert service.accept(key=CITATION_ACCEPTANCE_KEY).status == "accepted"
    f.now = permission.grant.deadline
    return f, permission, sub


def test_original_accepted_reader_retains_only_confirmed_items_after_local_grant_expiry(
    monkeypatch,
):
    f, permission, sub = accepted(monkeypatch)
    before, events = f.planner.read(f.docs.run_id), f.budget.journal.read()
    reader = AcceptedOriginReader(
        documents=f.docs,
        requests=f.requests,
        state_reader=f.owner.current,
        acceptance_key=CITATION_ACCEPTANCE_KEY,
    )
    proof, items, checked = reader.read(current_probe(f.state))
    assert {i.assessment_id for i in items} == {i.assessment_id for i in sub.candidate.items}
    assert proof.permission_checksum == digest(permission.grant)
    assert f.requests.records()[-1].request_id in proof.unused_request_ids
    assert reader.read(current_probe(f.state)) == (proof, items, checked)
    assert f.planner.read(f.docs.run_id) == before and f.budget.journal.read() == events
    assert len(f.wire.calls) == 2
    with pytest.raises(ValueError, match="deadline"):
        permission.check(f.state, role="review")
    f.owner.withdrawn = True
    with pytest.raises(ValueError):
        reader.read(current_probe(f.state))


def test_local_result_version_requires_exact_stored_binding_and_no_repaired_answer(monkeypatch):
    f, permission, _ = accepted(monkeypatch)
    local = dict(
        schema_version="multimodal-partial-finalization-result/1.2.0",
        original_result_checksum=digest(f.failed),
        phases=[
            dict(
                run_id=f.docs.run_id,
                acceptance_key=CITATION_ACCEPTANCE_KEY,
                accepted=True,
                target=None,
            )
        ],
        review_closure_scope=dict(
            local_confirmed_only=True,
            historical_answers_reaccepted=False,
            rejected_answers_repaired=False,
            campaign_execution_complete=False,
        ),
    )
    f.docs.write("citation-partial-finalization-result", (local,))
    assert local_origin_phase(f.docs, f.failed, local) == local["phases"][-1]
    changed = dict(
        local,
        review_closure_scope=dict(local["review_closure_scope"], rejected_answers_repaired=True),
    )
    with pytest.raises(ValueError, match="local_finalization_binding"):
        local_origin_phase(f.docs, f.failed, changed)
    reader = AcceptedOriginReader(
        documents=f.docs,
        requests=f.requests,
        state_reader=f.owner.current,
        acceptance_key=CITATION_ACCEPTANCE_KEY,
    )
    f.docs.saved["accepted:" + CITATION_ACCEPTANCE_KEY][-1]["permission_checksum"] = digest(
        "unbound"
    )
    with pytest.raises(ValueError, match="original_acceptance_required"):
        reader.read(current_probe(f.state))
