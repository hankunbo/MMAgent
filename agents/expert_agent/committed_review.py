"""Read an already committed Review through its exact intent and domain owner.

No Provider, acceptance write, successor install or Control transition is exposed.
Current-use/source checks remain the caller's original Phase responsibility.
"""

from types import SimpleNamespace

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_batches import (
    BatchedSubmission,
    read_submission,
    validate_batch_receipts,
)
from packages.grounding.models import digest, validate
from packages.llm_provider.models import read_request_record
from packages.llm_provider.provider import RequestBudget
from packages.planner_control.models import Reference
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .models import CommittedReviewReadback, ExpertDecision, ReviewFrame, read_result
from .reporting import read_research_report


def confirmed_transport(requests):
    """Reuse the request owner's distinction between local and remote completion."""
    counts = RequestBudget.transport_counts(requests)
    if counts["dispatch_unknown"] or counts["remote_response_unknown"]:
        raise ValueError("committed_review_unknown_request")
    return counts


def read_committed_review(documents, *, parent, config, confirmed_review_reader=None):
    """None means no accepted domain commit; contradictory bindings raise."""
    return read_committed_review_binding(
        documents,
        parent=parent,
        config_checksum=digest(config),
        confirmed_review_reader=confirmed_review_reader,
    )


def read_committed_review_binding(
    documents, *, parent, config_checksum, confirmed_review_reader=None
):
    """Historical proof only: an original config digest is not current-use authority."""
    intent = documents.read("live-stage-intent:review")
    if not intent:
        return None
    if len(intent) != 1:
        raise ValueError("committed_review_binding")
    stage = intent[0]
    event = validate(Reference, stage["event"])
    key = event.code + ":domain"
    accepted = documents.read("accepted:" + key)
    if not accepted or accepted[-1].get("status") != "accepted":
        return None
    parent = validate(DomainState, parent)
    documents.validate_scope(parent.material.scope)

    def one(name):
        values = documents.read(name)
        if not values or len(values) != 1:
            raise ValueError("committed_review_binding")
        return values[0]

    prefix = "review:" + event.code
    original = one(prefix + ":intent")
    frame = validate(ReviewFrame, original["frame"])
    command = one(prefix + ":command")
    decision = validate(ExpertDecision, one(prefix + ":decision"))
    selected = read_submission(one(prefix + ":selected")["submission"])
    submission = read_submission(one("submission:" + key)["submission"])
    if (
        stage["role"] != "review"
        or stage["config_checksum"] != config_checksum
        or parent.config_checksum != config_checksum
        or parent.material.scope.run_id != documents.run_id
        or validate(DomainState, original["state"]) != parent
        or frame.event != event
        or frame.scope != parent.material.scope
        or command
        != dict(
            event=stage["event"],
            proposed_action_id=stage["proposed_action_id"],
            role="review",
            abc=False,
        )
        or decision.event != event
        or decision.scope != frame.scope
        or decision.action_set_checksum != frame.actions.checksum
        or decision.policy_checksum != digest(frame.policy)
        or decision.decision not in {"accept", "baseline"}
        or decision.selected_action_id != stage["proposed_action_id"]
        or decision.proposed_action_id != stage["proposed_action_id"]
        or not any(
            a.id == stage["proposed_action_id"]
            and a.kind == "request_review"
            and a.branch == "research_review"
            for a in frame.actions.actions
        )
        or selected != submission
        or submission.scope != parent.material.scope
        or decision.proposal_checksum != digest(submission.candidate)
        or digest(submission) != accepted[-1]["submission_checksum"]
        or submission.observation_checksums
        != tuple(o.reference for o in parent.material.observations)
        or submission.source_checksums != tuple(s.reference for s in parent.material.sources)
    ):
        raise ValueError("committed_review_binding")
    journal = RuntimeAgentJournal(documents)
    events = journal.read()
    records = tuple(
        read_request_record(e["record"]) for e in events if e.get("phase") == "completed"
    )
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    if len(set(reserved)) != len(reserved) or sorted(reserved) != sorted(
        r.request_id for r in records
    ):
        raise ValueError("committed_review_unknown_request")
    requests = SimpleNamespace(records=lambda: records, journal=journal, used=len(reserved))
    confirmed_transport(requests)
    if not isinstance(submission, BatchedSubmission):
        # This closure belongs to the installed batched Review stage, not an
        # implicit upgrade of legacy single-request or continued submissions.
        raise ValueError("committed_review_submission_version")
    confirmed_batches(
        parent, submission, documents, requests, confirmed_review_reader=confirmed_review_reader
    )
    acceptance = ResearchAcceptance(
        owner=SimpleNamespace(current=lambda: parent), documents=documents, requests=None
    )
    report = read_research_report(acceptance, key=key, parent=parent)
    for name in (prefix + ":result", "live-stage:review"):
        saved = documents.read(name)
        if saved:
            result = read_result(saved[0])
            if (
                result["run_id"],
                result["event"],
                result.get("acceptance_key"),
                result.get("domain_status"),
                result.get("report_checksum"),
            ) != (documents.run_id, stage["event"], key, "accepted", digest(report)):
                raise ValueError("committed_review_result_binding")
    return CommittedReviewReadback(
        schema_version="committed-review-readback/1.0.0",
        run_id=documents.run_id,
        event=stage["event"],
        acceptance_key=key,
        parent_checksum=digest(parent),
        accepted_checksum=digest(accepted),
        report_checksum=digest(report),
        intent_checksum=digest(intent),
        journal_checksum=digest(events),
    ).model_dump(mode="json")


def confirmed_batches(parent, submission, documents, requests, *, confirmed_review_reader=None):
    """Original current ancestor owner validates reuse; history alone is not authority."""
    from packages.evidence_details.accepted_review_reuse import (
        ReusedAcceptedSubmission,
        ReusedAcceptedSubmissionV7,
    )
    from packages.evidence_details.review_continuation import (
        ContinuedReasonedSubmission,
        check_continued,
    )

    if isinstance(
        submission,
        (
            ContinuedReasonedSubmission,
            ReusedAcceptedSubmission,
            ReusedAcceptedSubmissionV7,
        ),
    ):
        if confirmed_review_reader is None:
            raise ValueError("committed_review_ancestor_owner_required")
        return check_continued(
            parent,
            submission,
            documents=documents,
            requests=requests,
            owner=SimpleNamespace(confirmed_review_reader=confirmed_review_reader),
        )
    return validate_batch_receipts(parent, submission, documents, requests)
