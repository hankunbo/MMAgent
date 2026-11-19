"""Historical local closure proof does not renew expired execution permission."""

from copy import deepcopy

import pytest

from packages.evidence_details import recorded_finalization as recorded
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import digest
from tests.agent_provider.finalization_support import fixture


def closed(*, reasoned=False):
    f = fixture(corrected=True, partial=reasoned, reasoned=reasoned)
    permission, sub = recorded.RecordedReviewPermission.prepare(
        owner=f.owner,
        documents=f.docs,
        requests=f.requests,
        config=f.config,
        clock=lambda: f.now,
    )
    key = permission.acceptance_key
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    service.submit(sub, key=key)
    assert service.accept(key=key).status == "accepted"
    # The legacy in-memory fixture recognizes only offline receipt v1.0.
    # Model the same exact live parent consumption exposed by RuntimeDocuments;
    # the persistent test below uses that original owner without this double.
    f.docs.consumed = lambda graph: (
        f.docs.read("accepted:" + key)[3]["status"] == "accepted"
        and all(
            f.docs.read("accepted:" + key)[3]["parent_graph"][name] == getattr(graph, name)
            for name in ("code", "checksum", "version")
        )
    )
    f.now = permission.grant.deadline
    return f, permission


def read(f):
    return recorded.read_saved_acceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.config
    )


def test_expired_local_grant_proves_committed_fact_without_new_authority():
    f, permission = closed()
    before = (
        deepcopy(f.docs.saved),
        deepcopy(f.budget.journal.read()),
        f.planner.read(f.docs.run_id),
    )
    proof = read(f)
    assert proof["acceptance_key"] == permission.acceptance_key
    assert proof["grant_checksum"] == digest(permission.grant)
    assert proof["semantics"] == "saved_acceptance_not_new_dispatch_authorization"
    assert read(f) == proof
    with pytest.raises(ValueError, match="recorded_finalization_deadline"):
        permission.check(f.state, role="review")
    assert before == (f.docs.saved, f.budget.journal.read(), f.planner.read(f.docs.run_id))
    assert len(f.wire.calls) == 3


@pytest.mark.parametrize(
    "fault", ["grant", "receipt", "submission", "score", "source", "unknown", "control"]
)
def test_saved_closure_rejects_changed_or_incomplete_proof(fault):
    f, permission = closed()
    if fault == "grant":
        f.docs.saved[permission.grant_key][0]["original_grant_checksum"] = digest("changed")
    elif fault == "receipt":
        f.docs.saved["accepted:" + permission.acceptance_key][3]["permission_checksum"] = digest(
            "changed"
        )
    elif fault == "submission":
        f.docs.saved["submission:" + permission.acceptance_key][0]["submission"]["candidate"][
            "summary"
        ] = "changed"
    elif fault == "score":
        f.docs.saved["accepted:" + permission.acceptance_key][2]["scores"]["checksum"] = digest(
            "changed"
        )
    elif fault == "source":
        f.owner.withdrawn = True
    elif fault == "unknown":
        f.budget.journal.events.append(dict(phase="reserved", request_id="unknown"))
    else:
        f.docs.sessions.state = f.docs.sessions.state.model_copy(update=dict(phase="select"))
    with pytest.raises(ValueError):
        read(f)
    assert len(f.wire.calls) == 3
