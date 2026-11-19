"""Zero-dispatch save of ALL confirmed local batches after rejection or deadline.

The original stopped run, errors, requests and reports remain immutable. This
permission cannot reserve, call a provider, install a successor or resume a run.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from packages.grounding.models import Checksum, digest, validate
from packages.llm_provider.models import read_request_record
from packages.object_store import checksum

from .finalization import FinalizationGrant
from .live_permission import ResearchLiveGrant, ResearchLivePermission, validate_live_config
from .review_batches import (
    ReviewBatchProposal,
    batch_payload,
    check_batch,
    collect_partial,
    read_response,
    validate_batch_receipts,
)

GRANT_KEY = "partial-review-finalization:grant:1"
SUBMISSION_KEY = "partial-review-finalization:submission:1"
RESULT_KEY = "multimodal-partial-finalization-result"
REPORT_KEY = "multimodal-partial-finalization-report"
ACCEPTANCE_KEY = "partial-review-finalization:1"
DEADLINE_GRANT_KEY = "deadline-partial-review-finalization:grant:1"
DEADLINE_SUBMISSION_KEY = "deadline-partial-review-finalization:submission:1"
DEADLINE_ACCEPTANCE_KEY = "deadline-partial-review-finalization:1"
CITATION_GRANT_KEY = "citation-partial-review-finalization:grant:1"
CITATION_SUBMISSION_KEY = "citation-partial-review-finalization:submission:1"
CITATION_ACCEPTANCE_KEY = "citation-partial-review-finalization:1"


class PartialFinalizationGrant(FinalizationGrant):
    schema_version: Literal["partial-review-finalization/1.0.0"] = (
        "partial-review-finalization/1.0.0"
    )
    purpose: Literal["save_all_confirmed_partial_reviews_only"] = (
        "save_all_confirmed_partial_reviews_only"
    )
    rejected_response_checksums: tuple[Checksum, ...]


class DeadlinePartialGrant(PartialFinalizationGrant):
    """Only local confirmed responses; historical answers are NOT reaccepted."""

    schema_version: Literal["partial-review-finalization/1.1.0"] = (
        "partial-review-finalization/1.1.0"
    )
    purpose: Literal["save_local_confirmed_batches_after_deadline_only"] = (
        "save_local_confirmed_batches_after_deadline_only"
    )
    excluded_historical_checksum: Checksum | None


class CitationPartialGrant(PartialFinalizationGrant):
    """Exclude a precisely reproduced citation rejection; never repair its answer."""

    schema_version: Literal["partial-review-finalization/1.2.0"] = (
        "partial-review-finalization/1.2.0"
    )
    purpose: Literal["save_local_confirmed_batches_after_citation_rejection_only"] = (
        "save_local_confirmed_batches_after_citation_rejection_only"
    )


def closure_keys(documents, *, citation_rejection=False):
    if citation_rejection:
        if (
            original_failure(documents)["phases"][-1]["error"]
            != "candidate_failed_or_context_changed"
        ):
            raise ValueError("partial_finalization_citation_failure_kind")
        return CITATION_GRANT_KEY, CITATION_SUBMISSION_KEY, CITATION_ACCEPTANCE_KEY
    if original_failure(documents)["phases"][-1]["error"] == "deadline":
        return DEADLINE_GRANT_KEY, DEADLINE_SUBMISSION_KEY, DEADLINE_ACCEPTANCE_KEY
    return GRANT_KEY, SUBMISSION_KEY, ACCEPTANCE_KEY


def historical_checksum(documents):
    intent = documents.read("live-stage-intent:review")
    if not intent:
        raise ValueError("partial_finalization_intent_missing")
    saved = documents.read("review-batches:" + intent[0]["event"]["code"] + ":A:historical")
    return digest(saved) if saved is not None else None


class ClosedReviewRequests:
    def __init__(self, journal):
        if not journal.durable:
            raise ValueError("partial_finalization_durable_journal_required")
        self.journal = journal

    def records(self):
        events = self.journal.read()
        reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        records = tuple(
            read_request_record(e["record"]) for e in events if e.get("phase") == "completed"
        )
        if (
            not records
            or len(set(reserved)) != len(reserved)
            or sorted(reserved) != sorted(r.request_id for r in records)
            or any(
                r.simulated
                or r.api_requests != 1
                or r.role != "review"
                or r.error not in (None, "scope")
                or r.retry_index != 0
                for r in records
            )
        ):
            raise ValueError("partial_finalization_unconfirmed_or_unsupported_request")
        return records


def original_failure(documents):
    saved = documents.read("multimodal-case-result")
    if not saved:
        raise ValueError("partial_finalization_failure_missing")
    result = saved[0]
    phases = result.get("phases", [])
    if (
        result.get("complete") is not False
        or not phases
        or phases[-1].get("run_id") != documents.run_id
        or phases[-1].get("target") is not None
        or phases[-1].get("accepted") is not False
        or phases[-1].get("error") not in {"candidate_failed_or_context_changed", "deadline"}
        or documents.read("dynamic-phase-result") != (phases[-1],)
    ):
        raise ValueError("partial_finalization_failure_binding")
    return result


def reconstruct(state, documents, requests, *, citation_rejection=False):
    if citation_rejection:
        closure_keys(documents, citation_rejection=True)
        if historical_checksum(documents) is not None:
            raise ValueError("partial_finalization_citation_local_only")
    intent = documents.read("live-stage-intent:review")
    if not intent:
        raise ValueError("partial_finalization_intent_missing")
    key = intent[0]["event"]["code"] + ":A"
    submission = collect_partial(state, documents, key)
    confirmed = validate_batch_receipts(state, submission, documents, requests)
    records = requests.records()
    if original_failure(documents)["phases"][-1]["error"] == "deadline":
        # Every real request must correspond to an original locally saved result.
        # Historical reuse markers do not supply additional accepted items here.
        if tuple(confirmed) != tuple(records) or any(r.error for r in records):
            raise ValueError("partial_finalization_deadline_unaccounted_request")
        if any(
            documents.read("review-batches:" + key + ":" + b + ":claim")
            for b in submission.pending_batch_ids
        ):
            raise ValueError("partial_finalization_deadline_unknown_claim")
        return submission, records, ()
    rejected = []
    for record in records:
        if record in confirmed:
            continue
        batches = [
            b
            for b in submission.plan.batches
            if digest(batch_payload(state, submission.plan, b)) == record.input_checksum
        ]
        if len(batches) != 1 or record.error != "scope":
            raise ValueError("partial_finalization_unaccounted_request")
        batch = batches[0]
        if batch.id not in submission.pending_batch_ids:
            raise ValueError("partial_finalization_rejected_confirmed_batch")
        saved = documents.read(
            "review-batches:" + key + ":" + batch.id + ":attempt:" + record.request_id
        )
        if not saved:
            raise ValueError("partial_finalization_rejected_response_missing")
        response = read_response(saved[0])
        if (
            response.error != "scope"
            or response.finish_reason != "stop"
            or response.request_id != record.request_id
            or response.record_checksum != digest(record)
            or checksum(response.raw_model_text.encode()) != response.raw_checksum
        ):
            raise ValueError("partial_finalization_rejected_response_binding")
        from packages.llm_provider.provider import strict_json
        from packages.llm_provider.review_response import decode

        if hasattr(response, "provider_config"):
            if digest(response.provider_config) != state.config_checksum:
                raise ValueError("partial_finalization_rejected_config")
            candidate, _ = decode(response.raw_model_text, response.provider_config)
        else:
            candidate = validate(ReviewBatchProposal, strict_json(response.raw_model_text))
        try:
            check_batch(state, submission.plan, batch, candidate)
        except ValueError as error:
            expected_error = (
                "review_citation_scope" if citation_rejection else "batch_candidate_binding"
            )
            if str(error) != expected_error:
                raise ValueError("partial_finalization_not_local_binding_rejection") from error
        else:
            raise ValueError("partial_finalization_rejection_not_reproduced")
        rejected.append(digest(response))
    if not rejected or len({r.input_checksum for r in records}) != len(records):
        raise ValueError("partial_finalization_requires_unique_confirmed_attempts")
    return submission, records, tuple(rejected)


def accepted_deadline_permission(state, documents, requests, submission, original_grant):
    """Verify provenance of a committed acceptance, not permission for new work.

    The acceptance owner must already have read the exact accepted receipt.
    Local grant expiry does not erase that fact; current material, control,
    journal, failure and excluded-history bindings are still revalidated.
    """
    raw = documents.read(DEADLINE_GRANT_KEY)
    if not raw or closure_keys(documents)[0] != DEADLINE_GRANT_KEY:
        raise ValueError("partial_origin_grant_missing")
    grant = validate(DeadlinePartialGrant, raw[0])
    current, records, rejected = reconstruct(state, documents, requests)
    control = documents.sessions.read(documents.run_id)
    if (
        control is None
        or control.phase != "stopped"
        or control.pending
        or grant.run_id != documents.run_id
        or grant.case_code != state.material.scope.case_code
        or grant.state_checksum != digest(state)
        or grant.control_checksum != digest(control)
        or grant.original_grant_checksum != digest(original_grant)
        or grant.failed_result_checksum != digest(original_failure(documents))
        or grant.excluded_historical_checksum != historical_checksum(documents)
        or grant.aggregate_key != DEADLINE_SUBMISSION_KEY
        or grant.aggregate_checksum != digest(submission)
        or current != submission
        or documents.read(DEADLINE_SUBMISSION_KEY) != (submission.model_dump(mode="json"),)
        or grant.request_checksums != tuple(digest(r) for r in records)
        or grant.rejected_response_checksums != rejected
    ):
        raise ValueError("partial_origin_binding_changed")
    return grant


def accepted_citation_permission(state, documents, requests):
    """Committed fact provenance only: expiry cannot authorize another dispatch."""
    raw = documents.read(CITATION_GRANT_KEY)
    if not raw:
        raise ValueError("citation_partial_origin_missing")
    grant = validate(CitationPartialGrant, raw[0])
    sub, records, rejected = reconstruct(state, documents, requests, citation_rejection=True)
    control = documents.sessions.read(documents.run_id)
    old = validate(ResearchLiveGrant, documents.read("research-live-grant")[0])
    accepted = documents.read("accepted:" + CITATION_ACCEPTANCE_KEY)
    if (
        not accepted
        or len(accepted) != 4
        or accepted[-1].get("status") != "accepted"
        or accepted[-1].get("permission_checksum") != digest(grant)
        or accepted[-1].get("submission_checksum") != digest(sub)
    ):
        raise ValueError("citation_partial_original_acceptance_required")
    if (
        control is None
        or control.phase != "stopped"
        or control.pending is not None
        or grant.run_id != documents.run_id
        or grant.case_code != state.material.scope.case_code
        or grant.control_checksum != digest(control)
        or grant.state_checksum != digest(state)
        or grant.original_grant_checksum != digest(old)
        or grant.failed_result_checksum != digest(original_failure(documents))
        or grant.aggregate_key != CITATION_SUBMISSION_KEY
        or grant.aggregate_checksum != digest(sub)
        or documents.read(CITATION_SUBMISSION_KEY) != (sub.model_dump(mode="json"),)
        or grant.request_checksums != tuple(digest(r) for r in records)
        or grant.rejected_response_checksums != rejected
    ):
        raise ValueError("citation_partial_origin_changed")
    documents.validate_scope(state.material.scope)
    return grant


class PartialReviewPermission(ResearchLivePermission):
    def __init__(self, *, grant, config, documents, requests, clock=None):
        raw = grant.model_dump(mode="json") if hasattr(grant, "model_dump") else grant
        grant_type = (
            CitationPartialGrant
            if raw.get("schema_version") == "partial-review-finalization/1.2.0"
            else DeadlinePartialGrant
            if raw.get("schema_version") == "partial-review-finalization/1.1.0"
            else PartialFinalizationGrant
        )
        self.grant = validate(grant_type, raw)
        self.config = validate_live_config(config)
        self.documents, self.budget = documents, requests
        self.clock = clock or (lambda: datetime.now(UTC))
        if requests.journal.documents is not documents:
            raise ValueError("partial_finalization_journal_owner")

    @classmethod
    def prepare(cls, *, owner, documents, requests, config, clock=None, citation_rejection=False):
        now = (clock or (lambda: datetime.now(UTC)))()
        state = owner.current()
        original = original_failure(documents)
        grant_key, submission_key, _ = closure_keys(
            documents, citation_rejection=citation_rejection
        )
        deadline_partial = grant_key == DEADLINE_GRANT_KEY
        control = documents.sessions.read(documents.run_id)
        if control is None or control.phase != "stopped" or control.pending is not None:
            raise ValueError("partial_finalization_requires_stopped_control")
        if documents.consumed(state.material.scope.graph):
            raise ValueError("partial_finalization_parent_consumed")
        sub, records, rejected = reconstruct(
            state, documents, requests, citation_rejection=citation_rejection
        )
        old = validate(ResearchLiveGrant, documents.read("research-live-grant")[0])
        expected = dict(
            run_id=documents.run_id,
            case_code=state.material.scope.case_code,
            original_grant_checksum=digest(old),
            failed_result_checksum=digest(original),
            control_checksum=digest(control),
            state_checksum=digest(state),
            aggregate_key=submission_key,
            aggregate_checksum=digest(sub),
            request_checksums=tuple(digest(r) for r in records),
            rejected_response_checksums=rejected,
        )
        if deadline_partial:
            expected["excluded_historical_checksum"] = historical_checksum(documents)
        grant_type = (
            CitationPartialGrant
            if citation_rejection
            else DeadlinePartialGrant
            if deadline_partial
            else PartialFinalizationGrant
        )
        with documents.session():
            prior = documents.read(grant_key)
            grant = (
                validate(grant_type, prior[0])
                if prior
                else grant_type(**expected, issued_at=now, deadline=now + timedelta(minutes=15))
            )
            if any(getattr(grant, k) != v for k, v in expected.items()):
                raise ValueError("partial_finalization_key_content_conflict")
            permission = cls(
                grant=grant, config=config, documents=documents, requests=requests, clock=clock
            )
            permission._check(state, role="review", persisted=False)
            documents.write(submission_key, (sub,))
            documents.write(grant_key, (grant,))
        return permission, sub

    def _check(self, state, *, role, record=None, persisted=True):
        g, docs, scope = self.grant, self.documents, state.material.scope
        citation_rejection = isinstance(g, CitationPartialGrant)
        grant_key, submission_key, _ = closure_keys(docs, citation_rejection=citation_rejection)
        if citation_rejection:
            if grant_key != CITATION_GRANT_KEY:
                raise ValueError("partial_finalization_failure_version")
        elif isinstance(g, DeadlinePartialGrant):
            if historical_checksum(docs) != g.excluded_historical_checksum:
                raise ValueError("partial_finalization_historical_changed")
            if grant_key != DEADLINE_GRANT_KEY:
                raise ValueError("partial_finalization_failure_version")
        elif grant_key != GRANT_KEY:
            raise ValueError("partial_finalization_failure_version")
        if self.clock() >= g.deadline:
            raise ValueError("partial_finalization_deadline")
        if role != "review" or digest(state) != g.state_checksum:
            raise ValueError("partial_finalization_scope")
        if persisted and (
            docs.read(grant_key) != (g.model_dump(mode="json"),)
            or digest(docs.read(submission_key)[0]) != g.aggregate_checksum
        ):
            raise ValueError("partial_finalization_grant_changed")
        control = docs.sessions.read(docs.run_id)
        old = validate(ResearchLiveGrant, docs.read("research-live-grant")[0])
        if (
            control is None
            or control.phase != "stopped"
            or control.pending is not None
            or digest(control) != g.control_checksum
            or digest(original_failure(docs)) != g.failed_result_checksum
            or digest(old) != g.original_grant_checksum
            or (scope.run_id, scope.case_code) != (g.run_id, g.case_code)
            or (
                scope.run_id,
                scope.case_code,
                scope.input_checksum,
                scope.visible_checksum,
                scope.policy_checksum,
            )
            != (
                old.run_id,
                old.case_code,
                old.input_checksum,
                old.visible_checksum,
                old.policy_checksum,
            )
            or state.config_checksum != old.config_checksum
            or digest(self.config) != old.config_checksum
            or old.root_checksum != docs.root.checksum
            or "review" not in old.roles
            or scope.graph.kind != "research_artifact"
            or any(s.purpose != "research_material" for s in state.material.sources)
        ):
            raise ValueError("partial_finalization_original_binding")
        sub, records, rejected = reconstruct(
            state, docs, self.budget, citation_rejection=citation_rejection
        )
        if (digest(sub), tuple(digest(r) for r in records), rejected) != (
            g.aggregate_checksum,
            g.request_checksums,
            g.rejected_response_checksums,
        ):
            raise ValueError("partial_finalization_records_changed")
        for r in records:
            if (
                r.scope != scope
                or r.deadline > old.deadline
                or r.config_checksum != old.config_checksum
                or r.prompt_checksum != state.prompt_checksum
                or (r.provider, r.model, r.endpoint)
                != (
                    self.config.transport.provider,
                    self.config.transport.model,
                    self.config.transport.endpoint,
                )
            ):
                raise ValueError("partial_finalization_request_binding")
        if record is not None and (record not in records or record.error is not None):
            raise ValueError("partial_finalization_request_not_accepted")
        docs.validate_scope(scope)

    def check(self, state, *, role=None, record=None):
        self._check(state, role=role, record=record)
