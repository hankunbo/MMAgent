"""A partitioned accepted ancestor can precede an ordinary, unmixed plan."""

import pytest

from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.review_batches import make_plan, read_submission
from packages.grounding.models import digest
from tests.agent_provider.test_accepted_review_reuse import accepted_successor, state_from
from tests.evaluation_preparation.test_partition_increment import partition_fixture


def ordinary_successor(pg=None, second=None, second_binding=None, third=None, third_binding=None):
    previous = partition_fixture(pg=pg, documents=second, binding=second_binding)
    current = accepted_successor(
        previous, acceptance_key="new", documents=third, binding=third_binding
    )
    raw = current.owner.current().model_dump(mode="json")
    retained = {
        item.assessment_id
        for item in previous.submission.candidate.items
        if item.status == "assessed"
    }
    keep = retained | {"item:third"}
    policy = raw["policy"]
    policy["items"] = [item for item in policy["items"] if item["id"] in keep]
    policy["requirements"] = [item for item in policy["requirements"] if item["id"] in keep]
    # Historical unassessed observations remain visible, but are not re-reviewed.
    groups = []
    for group in policy["groups"]:
        group["slots"] = [[key for key in slot if key in keep] for slot in group["slots"]]
        group["slots"] = [slot for slot in group["slots"] if slot]
        if group["slots"]:
            groups.append(group)
    policy["groups"] = groups
    for item in policy["items"]:
        if item["id"] == "item:third":
            item["id"] = "zz:third"
    for item in policy["requirements"]:
        if item["id"] == "item:third":
            item["id"] = "zz:third"
    for group in policy["groups"]:
        group["slots"] = [
            ["zz:third" if key == "item:third" else key for key in slot] for slot in group["slots"]
        ]
    current.owner.state = state_from(raw)
    origin = current.owner.confirmed_review_reader.current(current.owner.current())
    plan = make_plan(
        current.owner.current(), current.agent.provider.config.review_batching, key="third:A"
    )
    assert origin.schema_version == "accepted-review-origin/1.3.0"
    assert not any(
        set(batch.assessment_ids) & retained and not set(batch.assessment_ids) <= retained
        for batch in plan.batches
    )
    return previous, current


def test_ordinary_plan_after_partitioned_ancestor_preserves_scores_and_provenance():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.evidence_details.acceptance_models import read_records

    previous, current = ordinary_successor()
    before = digest(previous.documents.saved), digest(previous.old.documents.saved)
    submission = run_batches(
        current.agent,
        state=current.owner.current(),
        documents=current.documents,
        key="third:A",
        round_id="A",
    )
    assert submission.schema_version == "research-batched-submission/15.0.0"
    assert submission.plan.schema_version == "review-batch-plan/1.0.0"
    assert len(current.budget.records()) == 1
    assert read_submission(submission.model_dump(mode="json")) == submission
    current.acceptance.submit(submission, key="third")
    receipt = current.acceptance.accept(key="third")
    assert receipt.status == "accepted"
    records = read_records(current.acceptance.read(key="third")[0])
    assert records.schema_version == "case-observation-interpretation/2.17.0"
    report = read_research_report(current.acceptance, key="third", parent=current.owner.current())
    assert "accepted-review-origin/1.3.0" in render_research_html(report)
    audit = len(current.documents.audit)
    assert current.acceptance.accept(key="third") == receipt
    assert len(current.documents.audit) == audit and len(current.budget.records()) == 1
    assert before == (digest(previous.documents.saved), digest(previous.old.documents.saved))


def failed_ordinary_successor(current, monkeypatch):
    """Reproduce the original dispatch bug after the original provider saves its batch."""
    from packages.evidence_details import review_continuation
    from packages.evidence_details.accepted_review_reuse import ReusedAcceptedSubmissionV9
    from packages.grounding.models import validate

    aggregate = review_continuation.aggregate_continued

    def legacy(*args, **kwargs):
        submission = aggregate(*args, **kwargs)
        if submission.schema_version != "research-batched-submission/15.0.0":
            return submission
        raw = submission.model_dump(mode="json")
        raw["schema_version"] = "research-batched-submission/9.0.0"
        return validate(ReusedAcceptedSubmissionV9, raw)

    with monkeypatch.context() as patch:
        patch.setattr(review_continuation, "aggregate_continued", legacy)
        with pytest.raises(
            ValueError, match="2 validation errors for ReusedAcceptedSubmissionV9"
        ) as error:
            run_batches(
                current.agent,
                state=current.owner.current(),
                documents=current.documents,
                key="third:A",
                round_id="A",
            )
    docs = current.documents
    docs.run_id = current.owner.current().material.scope.run_id
    docs.write("live-stage-intent:review", ({"event": {"code": "third"}},))
    phase = dict(run_id=docs.run_id, target=None, accepted=False, error=str(error.value)[:160])
    docs.write("dynamic-phase-result", (phase,))
    docs.write("multimodal-case-result", (dict(complete=False, phases=[phase]),))
    assert docs.read("review-batches:third:A:aggregate") is None
    return phase


def test_saved_complete_batch_compatibility_reconstruction_is_zero_request(monkeypatch):
    from packages.evidence_details.recorded_finalization import (
        origin_compatibility_proof,
        original_failure,
        reconstruct,
    )

    _, current = ordinary_successor()
    phase = failed_ordinary_successor(current, monkeypatch)
    before = digest(current.documents.saved), digest(current.budget.journal.read())
    assert origin_compatibility_proof(current.documents)
    assert original_failure(current.documents)["phases"] == [phase]
    for _ in range(2):
        sub, records, rejected = reconstruct(
            current.owner.current(), current.documents, current.budget, owner=current.owner
        )
        assert sub.schema_version == "research-batched-submission/15.0.0"
        assert len(records) == 1 and not rejected
    assert before == (digest(current.documents.saved), digest(current.budget.journal.read()))


@pytest.mark.parametrize("fault", ["error", "origin", "batch", "aggregate", "owner", "unknown"])
def test_compatibility_cannot_authorize_other_failure_or_changed_evidence(monkeypatch, fault):
    from packages.evidence_details.recorded_finalization import original_failure, reconstruct

    previous, current = ordinary_successor()
    failed_ordinary_successor(current, monkeypatch)
    docs, owner = current.documents, current.owner
    if fault == "error":
        docs.saved["multimodal-case-result"][0]["phases"][0]["error"] = "network_timeout"
    elif fault == "origin":
        docs.saved["review-batches:third:A:historical"][0]["acceptance_checksum"] = digest("wrong")
    elif fault == "batch":
        key = next(
            key
            for key in docs.saved
            if key.startswith("review-batches:third:A:batch:") and key.endswith(":result")
        )
        del docs.saved[key]
    elif fault == "aggregate":
        docs.write("review-batches:third:A:aggregate", ({"wrong": True},))
    elif fault == "owner":
        owner = None
    else:
        previous.budget.journal.append(dict(phase="reserved", request_id="UNKNOWN"))
    before = digest(docs.saved), digest(current.budget.journal.read())
    with pytest.raises(
        ValueError,
        match="only_closed_deadline|partition_origin_changed|complete_batches_required|original_reader_required|accepted_review_reuse_unknown_request",
    ):
        original_failure(docs)
        reconstruct(current.owner.current(), docs, current.budget, owner=owner)
    assert before == (digest(docs.saved), digest(current.budget.journal.read()))
