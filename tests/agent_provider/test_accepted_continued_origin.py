"""A committed same-contract continuation can seed later demand acquisition."""

from copy import deepcopy

import pytest

from agents.provider_agents.batched_review import run_batches
from packages.grounding.models import digest, validate
from tests.agent_provider.test_accepted_review_reuse import accepted_successor
from tests.agent_provider.test_review_continuation import continuation_fixture


def completed_continuation(
    pg=None, next_documents=None, next_binding=None, third_documents=None, third_binding=None
):
    f = continuation_fixture(pg, next_documents, next_binding)
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="continued:A", round_id="A"
    )
    f.acceptance.submit(sub, key="closed")
    assert f.acceptance.accept(key="closed").status == "accepted"
    assert sub.schema_version == "research-batched-submission/5.0.0"
    newer = accepted_successor(
        f, acceptance_key="closed", documents=third_documents, binding=third_binding
    )
    original = f.owner.confirmed_review_reader

    def ancestor(run_id, lineage):
        assert run_id == original.documents.run_id
        assert f.documents.run_id in lineage
        return original

    newer.owner.confirmed_review_reader.ancestor_reader = ancestor
    return f, newer


def test_accepted_continuation_new_item_keeps_all_three_original_requests():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.evidence_details.acceptance_models import read_records
    from packages.evidence_details.accepted_review_reuse import AcceptedReviewOriginV13
    from packages.evidence_details.review_batches import read_submission

    f, g = completed_continuation()
    before = digest(f.old.documents.saved), digest(f.documents.saved)
    sub = run_batches(
        g.agent, state=g.owner.current(), documents=g.documents, key="new", round_id="A"
    )
    assert sub.schema_version == "research-batched-submission/14.0.0"
    assert sub.historical.schema_version == "accepted-review-origin/1.4.0"
    # Only the three previously assessed items are reusable; the two pending
    # items retain their existing semantics and are not promoted by reuse.
    assert len(sub.historical.assessment_ids) == 3
    assert len(g.budget.records()) == 2 and len(sub.candidate.items) == 6
    assert read_submission(sub.model_dump(mode="json")) == sub
    old = sub.historical.model_dump(mode="json")
    old["schema_version"] = "accepted-review-origin/1.3.0"
    with pytest.raises(ValueError):
        validate(AcceptedReviewOriginV13, old)
    g.acceptance.submit(sub, key="new-accepted")
    receipt = g.acceptance.accept(key="new-accepted")
    assert receipt.status == "accepted"
    records = read_records(g.acceptance.read(key="new-accepted")[0])
    assert records.schema_version == "case-observation-interpretation/2.16.0"
    assert len(records.request_ids) == 5 and len(set(records.request_ids)) == 5
    report = read_research_report(g.acceptance, key="new-accepted", parent=g.owner.current())
    assert len(report["scores"]["reviews"]) == 6
    assert "accepted-review-origin/1.4.0" in render_research_html(report)
    count = len(g.documents.audit)
    assert g.acceptance.accept(key="new-accepted") == receipt
    assert len(g.documents.audit) == count and len(g.budget.records()) == 2
    assert before == (digest(f.old.documents.saved), digest(f.documents.saved))


@pytest.mark.parametrize("fault", ["ancestor_missing", "ancestor_unknown", "receipt", "response"])
def test_continued_accepted_origin_cannot_bypass_original_proof(fault):
    f, g = completed_continuation()
    if fault == "ancestor_missing":
        g.owner.confirmed_review_reader.ancestor_reader = None
        error = "accepted_review_reuse_ancestor_owner_required"
    elif fault == "ancestor_unknown":
        f.old.budget.journal.append(dict(phase="reserved", request_id="unknown"))
        error = "review_continuation_unknown_or_unaccounted_request"
    elif fault == "receipt":
        f.documents.saved["accepted:closed"][3]["records_checksum"] = digest("changed")
        error = "accepted_review_reuse_receipt_binding"
    else:
        f.old.documents.saved["review-batches:origin:A:batch:000:response"][0][
            "raw_model_text"
        ] += " "
        error = "batch_response_binding"
    before = deepcopy(g.documents.saved)
    with pytest.raises(ValueError, match=error):
        run_batches(
            g.agent, state=g.owner.current(), documents=g.documents, key="new", round_id="A"
        )
    assert not g.budget.records() and g.documents.saved == before


def test_production_budget_successor_resolves_bound_confirmed_owner():
    from types import SimpleNamespace

    from packages.evidence_details.review_continuation import ConfirmedReviewReader
    from packages.multimodal_research.campaign_renewal import accepted_reader

    f, _ = completed_continuation()
    f.old.documents.write("dynamic-phase-input", (f.old.owner.current(),))
    f.documents.write(
        "dynamic-phase-result", (dict(accepted=True, target=None, acceptance_key="closed"),)
    )
    original_id = f.old.documents.run_id

    def original_documents(run_id):
        assert run_id == original_id
        return f.old.documents

    parent = SimpleNamespace(
        documents=lambda _: f.documents,
        protocol=dict(
            schema_version="multimodal-campaign-renewal/1.31.0", tip=dict(run_id=original_id)
        ),
        parent=SimpleNamespace(documents=original_documents),
    )
    reader = accepted_reader(parent, f.documents.run_id)
    origin = reader.ancestor_reader(original_id, (f.documents.run_id,))
    assert isinstance(origin, ConfirmedReviewReader)
    assert origin.documents is f.old.documents
    assert origin.state_reader() == f.old.owner.current()
