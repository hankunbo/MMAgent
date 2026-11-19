"""Accepted judgments recombine through the original owner, without new Review."""

from types import SimpleNamespace

import pytest

from agents.expert_agent.reporting import read_research_report, render_research_html
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState, read_records
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    AcceptedOriginReader,
    AcceptedReviewSet,
    AggregationPermission,
    AggregationRequests,
    HistoricalRequests,
)
from packages.evidence_details.recorded_explanation import (
    ACCEPTANCE_KEY as ORIGINAL_KEY,
)
from packages.evidence_details.recorded_explanation import (
    RecordedReviewPermission,
)
from packages.grounding.models import digest, validate
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.expert_agent.explanation_support import failed_explanation
from tests.expert_agent.test_batch_assembly import LiveDocuments


def accepted_origin(monkeypatch, tmp_path):
    f = failed_explanation(monkeypatch, tmp_path)
    permission, submission = RecordedReviewPermission.prepare(
        owner=f.owner, documents=f.documents, requests=f.requests, config=f.config
    )
    service = ResearchAcceptance(
        owner=f.owner, documents=f.documents, requests=f.requests, live_permission=permission
    )
    assert service.submit(submission, key=ORIGINAL_KEY)["status"] == "submitted_not_accepted"
    assert service.accept(key=ORIGINAL_KEY).status == "accepted"
    return f


def aggregate_fixture(f):
    raw = f.state.model_dump(mode="json")
    material = raw["material"]
    material["scope"]["run_id"] = "RUN:AGGREGATED"
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    state = validate(DomainState, raw)
    docs = LiveDocuments()
    docs.run_id, docs.root = state.material.scope.run_id, f.documents.root
    control = f.documents.sessions.read(f.binding.run_id)
    control = control.model_copy(
        update={"binding": control.binding.model_copy(update={"run_id": docs.run_id})}
    )
    docs.sessions = SimpleNamespace(read=lambda _: control)
    reader = AcceptedOriginReader(
        documents=f.documents,
        requests=HistoricalRequests(f.ports.budget.journal),
        state_reader=lambda: f.state,
        acceptance_key=ORIGINAL_KEY,
    )
    source = AcceptedReviewSet([reader], f.config)
    owner = SimpleNamespace(current=lambda: state, accepted_review_set=source)
    requests = AggregationRequests(source, owner, RuntimeAgentJournal(docs))
    return SimpleNamespace(
        owner=owner, documents=docs, requests=requests, state=state, source=source, reader=reader
    )


def test_original_local_acceptance_zero_request_aggregate_and_replay(monkeypatch, tmp_path):
    f = accepted_origin(monkeypatch, tmp_path)
    original, events = dict(f.documents.saved), f.ports.budget.journal.read()
    a = aggregate_fixture(f)
    permission, submission = AggregationPermission.prepare(
        owner=a.owner, documents=a.documents, requests=a.requests, config=f.config
    )
    service = ResearchAcceptance(
        owner=a.owner, documents=a.documents, requests=a.requests, live_permission=permission
    )
    assert service.submit(submission, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    receipt = service.accept(key=ACCEPTANCE_KEY)
    assert receipt.status == "accepted" and receipt.new_medical_review is False
    bundle = service.read(key=ACCEPTANCE_KEY)
    records = read_records(bundle[0])
    assert records.new_model_requests == 0 and records.origins[0].acceptance_key == ORIGINAL_KEY
    assert a.documents.consumed(a.state.material.scope.graph)
    report = read_research_report(service, key=ACCEPTANCE_KEY, parent=a.state)
    assert "零新模型请求" in render_research_html(report)
    before = list(a.documents.audit)
    assert service.accept(key=ACCEPTANCE_KEY) == receipt
    assert a.documents.audit == before and not a.requests.journal.read()
    assert f.documents.saved == original and f.ports.budget.journal.read() == events
    assert len(f.wire.calls) == 1
    assert records.candidate.items == submission.candidate.items


@pytest.mark.parametrize("fault", ["duplicate", "scope", "missing", "source", "pending"])
def test_aggregation_refuses_incomplete_changed_or_duplicate_origin(monkeypatch, tmp_path, fault):
    f = accepted_origin(monkeypatch, tmp_path)
    a = aggregate_fixture(f)
    if fault == "duplicate":
        a.source.readers = (a.reader, a.reader)
    elif fault == "scope":
        a.reader.acceptance_key = "foreign"
    elif fault == "missing":
        f.documents.saved.pop("accepted:" + ORIGINAL_KEY)
    elif fault == "source":
        state = f.state.model_copy(update={"config_checksum": digest("changed")})
        a.reader.state_reader = lambda: state
    else:
        monkeypatch.setattr(
            a.reader.requests.journal,
            "read",
            lambda: [{"phase": "reserved", "request_id": "missing"}],
        )
    with pytest.raises(ValueError):
        a.source.current(a.state)
    assert not a.documents.saved and len(f.wire.calls) == 1
