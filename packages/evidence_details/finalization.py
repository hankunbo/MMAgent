"""Explicit, zero-dispatch finalization of a complete, previously recorded review.

This is a new local authorization to save facts, not an extension of a live run.
The stopped Control, original grant, request journal and failed report are immutable.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Code, Value, digest, validate
from packages.llm_provider.models import read_request_record

from .live_permission import ResearchLiveGrant, ResearchLivePermission, validate_live_config
from .review_batches import BatchedSubmission, read_submission, validate_batch_receipts

GRANT_KEY = "confirmed-review-finalization:grant:1"
RESULT_KEY = "dynamic-case-confirmed-result"
REPORT_KEY = "dynamic-case-confirmed-report"


class FinalizationGrant(Value):
    schema_version: Literal["confirmed-review-finalization/1.0.0"] = (
        "confirmed-review-finalization/1.0.0"
    )
    purpose: Literal["save_confirmed_reviews_only"] = "save_confirmed_reviews_only"
    run_id: Code
    case_code: Code
    original_grant_checksum: Checksum
    failed_result_checksum: Checksum
    control_checksum: Checksum
    state_checksum: Checksum
    aggregate_key: str = Field(min_length=1, max_length=300)
    aggregate_checksum: Checksum
    request_checksums: tuple[Checksum, ...] = Field(min_length=1, max_length=20)
    issued_at: datetime
    deadline: datetime
    new_requests: Literal[0] = 0
    install_successor: Literal[False] = False

    @model_validator(mode="after")
    def bounded(self):
        if (
            self.issued_at.tzinfo is None
            or self.deadline.tzinfo is None
            or not timedelta(0) < self.deadline - self.issued_at <= timedelta(minutes=15)
            or len(set(self.request_checksums)) != len(self.request_checksums)
        ):
            raise ValueError("finalization_bound")
        return self


class RecordedRequests:
    """Read interface only: deliberately no reserve, provider or dispatch methods."""

    def __init__(self, journal):
        if not journal.durable:
            raise ValueError("finalization_durable_journal_required")
        self.journal = journal

    def records(self):
        events = self.journal.read()
        reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        records = [
            read_request_record(e["record"]) for e in events if e.get("phase") == "completed"
        ]
        if (
            len(set(reserved)) != len(reserved)
            or sorted(reserved) != sorted(r.request_id for r in records)
            or any(
                r.error or r.simulated or r.api_requests != 1 or r.role != "review" for r in records
            )
        ):
            raise ValueError("finalization_requires_all_confirmed_reviews")
        return tuple(records)


def original_failure(documents):
    values = documents.read("dynamic-case-result")
    if not values:
        raise ValueError("finalization_original_result_missing")
    result = values[0]
    if (
        result.get("complete") is not False
        or not result.get("phases")
        or result["phases"][-1].get("error") != "deadline"
        or result["phases"][-1].get("run_id") != documents.run_id
    ):
        raise ValueError("finalization_only_deadline_stopped_review")
    return result


class ConfirmedReviewPermission(ResearchLivePermission):
    """Narrow explicit grant; never usable for Source/Vision/new model requests."""

    def __init__(self, *, grant, config, documents, requests, clock=None):
        self.grant = validate(FinalizationGrant, grant)
        self.config = validate_live_config(config)
        self.documents, self.budget = documents, requests
        self.clock = clock or (lambda: datetime.now(UTC))
        if getattr(requests.journal, "documents", None) is not documents:
            raise ValueError("finalization_journal_owner")

    @classmethod
    def prepare(cls, *, owner, documents, requests, config, clock=None):
        now = (clock or (lambda: datetime.now(UTC)))()
        state = owner.current()  # Real owner must re-read input/source/policy first.
        original = original_failure(documents)
        control = documents.sessions.read(documents.run_id)
        if control is None or control.phase != "stopped" or control.pending is not None:
            raise ValueError("finalization_requires_closed_control")
        intent = documents.read("live-stage-intent:review")
        if not intent:
            raise ValueError("finalization_review_intent_missing")
        aggregate_key = "review-batches:" + intent[0]["event"]["code"] + ":A:aggregate"
        values = documents.read(aggregate_key)
        if not values:
            raise ValueError("finalization_complete_aggregate_required")
        submission = read_submission(values[0])
        if type(submission) is not BatchedSubmission:
            raise ValueError("finalization_complete_aggregate_required")
        records = validate_batch_receipts(state, submission, documents, requests)
        if tuple(records) != requests.records():
            raise ValueError("finalization_no_omitted_requests")
        old = validate(ResearchLiveGrant, documents.read("research-live-grant")[0])
        expected = dict(
            run_id=documents.run_id,
            case_code=state.material.scope.case_code,
            original_grant_checksum=digest(old),
            failed_result_checksum=digest(original),
            control_checksum=digest(control),
            state_checksum=digest(state),
            aggregate_key=aggregate_key,
            aggregate_checksum=digest(submission),
            request_checksums=tuple(digest(r) for r in records),
        )
        with documents.session():
            prior = documents.read(GRANT_KEY)
            grant = (
                validate(FinalizationGrant, prior[0])
                if prior
                else FinalizationGrant(
                    **expected, issued_at=now, deadline=now + timedelta(minutes=15)
                )
            )
            if any(getattr(grant, k) != v for k, v in expected.items()):
                raise ValueError("finalization_key_content_conflict")
            permission = cls(
                grant=grant, config=config, documents=documents, requests=requests, clock=clock
            )
            permission._check(state, role="review", persisted=False)
            documents.write(GRANT_KEY, (grant,))
        return permission, submission

    def _check(self, state, *, role, record=None, persisted=True):
        g, docs, scope = self.grant, self.documents, state.material.scope
        if self.clock() >= g.deadline:
            raise ValueError("finalization_deadline")
        if role != "review" or digest(state) != g.state_checksum:
            raise ValueError("finalization_scope")
        if persisted and docs.read(GRANT_KEY) != (g.model_dump(mode="json"),):
            raise ValueError("finalization_grant_changed")
        old = validate(ResearchLiveGrant, docs.read("research-live-grant")[0])
        control = docs.sessions.read(docs.run_id)
        if (
            control is None
            or control.phase != "stopped"
            or control.pending is not None
            or digest(control) != g.control_checksum
            or digest(original_failure(docs)) != g.failed_result_checksum
            or digest(old) != g.original_grant_checksum
            or docs.read(g.aggregate_key) is None
            or digest(docs.read(g.aggregate_key)[0]) != g.aggregate_checksum
        ):
            raise ValueError("finalization_origin_changed")
        if (
            (scope.run_id, scope.case_code) != (g.run_id, g.case_code)
            or g.run_id != docs.run_id
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
            raise ValueError("finalization_original_grant_binding")
        records = self.budget.records()
        if tuple(digest(r) for r in records) != g.request_checksums:
            raise ValueError("finalization_records_changed")
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
                raise ValueError("finalization_request_binding")
        if record is not None and digest(record) not in g.request_checksums:
            raise ValueError("finalization_request_not_confirmed")
        docs.validate_scope(scope)

    def check(self, state, *, role=None, record=None):
        self._check(state, role=role, record=record)
