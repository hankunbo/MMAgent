"""Old bound failures close only with complete, owner-bound saved Review."""

from copy import deepcopy

import pytest

from packages.evidence_details import recorded_finalization as recorded
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import digest
from tests.agent_provider.lineage_finalization_support import chain


def test_depth_failure_local_grant_original_acceptance_readback(monkeypatch):
    f = chain(monkeypatch)
    before = (
        digest(f.documents.read("multimodal-case-result")),
        digest(f.budget.journal.read()),
        digest(f.documents.sessions.read(f.documents.run_id)),
    )
    permission, sub = recorded.RecordedReviewPermission.prepare(
        owner=f.owner, documents=f.documents, requests=f.requests, config=f.config
    )
    assert permission.grant.schema_version == "recorded-review-finalization/1.4.0"
    assert sub == f.submission
    service = ResearchAcceptance(
        owner=f.owner, documents=f.documents, requests=f.requests, live_permission=permission
    )
    key = permission.acceptance_key
    service.submit(sub, key=key)
    assert service.accept(key=key).status == "accepted"
    assert (
        recorded.read_saved_acceptance(
            owner=f.owner, documents=f.documents, requests=f.requests, config=f.config
        )["acceptance_key"]
        == key
    )
    assert before == (
        digest(f.documents.read("multimodal-case-result")),
        digest(f.budget.journal.read()),
        digest(f.documents.sessions.read(f.documents.run_id)),
    )
    assert len(f.wire.calls) == 1


def test_depth_failure_proof_cannot_replace_complete_records(monkeypatch):
    f = chain(monkeypatch)
    docs = f.documents
    saved = deepcopy(docs.saved)
    assert recorded.lineage_capacity_proof(docs)
    prefix = "review-batches:lineage5:A"
    for key in [
        prefix + ":aggregate",
        prefix + ":historical",
        prefix + ":plan",
        prefix + ":" + f.submission.batches[0].batch_id + ":response",
    ]:
        docs.saved = deepcopy(saved)
        del docs.saved[key]
        assert recorded.lineage_capacity_proof(docs) is None
    docs.saved = deepcopy(saved)
    docs.saved["multimodal-case-result"][0]["phases"][0]["error"] = "network_timeout"
    assert recorded.lineage_capacity_proof(docs) is None
    docs.saved = deepcopy(saved)
    f.budget.journal.append(dict(phase="reserved", request_id="unknown"))
    with pytest.raises(ValueError, match="recorded_finalization_unknown_or_unsupported"):
        recorded.RecordedReviewPermission.prepare(
            owner=f.owner, documents=docs, requests=f.requests, config=f.config
        )
    assert len(f.wire.calls) == 1
