"""Explicit cross-window reuse of confirmed, source-bound Review responses.

Original responses/requests retain their scope and bytes. Only the owner-derived
aggregate has the successor scope. A historical response never grants execution.
"""

from typing import Literal

from packages.grounding.models import Checksum, Token, Value, digest, validate

from .review_batches import (
    PartialReasonedSubmission,
    ReasonedSubmission,
    aggregate,
    collect_partial,
    validate_batch_receipts,
)


class HistoricalReview(Value):
    schema_version: Literal["confirmed-review-origin/1.0.0"] = "confirmed-review-origin/1.0.0"
    run_id: Token
    state_checksum: Checksum
    control_checksum: Checksum
    journal_checksum: Checksum
    grant_checksum: Checksum | None
    semantic_checksum: Checksum
    submission: PartialReasonedSubmission


class ContinuedReasonedSubmission(ReasonedSubmission):
    schema_version: Literal["research-batched-submission/5.0.0"] = (
        "research-batched-submission/5.0.0"
    )
    historical: HistoricalReview


def semantic_binding(state):
    """Only execution identity is excluded; medical inputs/config remain exact."""
    raw = state.model_dump(mode="json")
    scope = raw["material"]["scope"]
    for name in ("run_id", "graph", "material_checksum"):
        scope.pop(name)
    return digest(raw)


class ConfirmedReviewReader:
    """Installed by composition with ORIGINAL documents, not caller-supplied proof."""

    def __init__(
        self, *, documents, requests, state_reader, format_config=None, expected_format_origin=None
    ):
        self.documents, self.requests, self.state_reader = documents, requests, state_reader
        self.format_config = format_config
        self.expected_format_origin = expected_format_origin

    def current(self, state):
        docs = self.documents
        old = self.state_reader()
        control = docs.sessions.read(docs.run_id)
        if (
            control is None
            or control.phase != "stopped"
            or control.pending is not None
            or old.material.scope.run_id != docs.run_id
            or state.material.scope.run_id == docs.run_id
            or semantic_binding(old)
            != semantic_binding(
                state.model_copy(update={"config_checksum": old.config_checksum})
                if self.format_config is not None
                else state
            )
        ):
            raise ValueError("review_continuation_current_binding")
        docs.validate_scope(old.material.scope)
        self._check_consumption(old, state)
        intent = docs.read("live-stage-intent:review")
        if not intent:
            raise ValueError("review_continuation_origin_intent")
        key = intent[0]["event"]["code"] + ":A"
        sub = collect_partial(old, docs, key)
        if not isinstance(sub, PartialReasonedSubmission):
            raise ValueError("review_continuation_reasoned_required")
        records = validate_batch_receipts(old, sub, docs, self.requests)
        events = self.requests.journal.read()
        reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        all_records = self.requests.records()
        if self.format_config is None and (
            len(set(reserved)) != len(reserved)
            or set(reserved) != {r.request_id for r in all_records}
            or any(
                docs.read("review-batches:" + key + ":" + b + ":claim")
                for b in sub.pending_batch_ids
            )
        ):
            raise ValueError("review_continuation_unknown_or_unaccounted_request")
        if self.format_config is None:
            from .recorded_finalization import verified_corrected_records

            # An original confirmed schema correction consumes request budget,
            # but only its successful response is reusable medical evidence.
            # Reuse the same owner proof as recorded finalization; never infer
            # confirmation from a schema label or permit unknown extra requests.
            verified_corrected_records(old, sub, docs, self.requests, records)
        format_proof = None
        if self.format_config is not None:
            from .review_format_continuation import verify_format_origin

            format_proof = verify_format_origin(self, old, state, sub, records)
        grant = docs.read("research-live-grant")
        if any(not r.simulated for r in records):
            from .live_permission import ResearchLiveGrant
            from .review_batches import read_response, response_key

            g = validate(ResearchLiveGrant, grant[0]) if grant else None
            s = old.material.scope
            if (
                not self.requests.journal.durable
                or g is None
                or (
                    g.run_id,
                    g.case_code,
                    g.input_checksum,
                    g.visible_checksum,
                    g.policy_checksum,
                    g.config_checksum,
                    g.root_checksum,
                )
                != (
                    s.run_id,
                    s.case_code,
                    s.input_checksum,
                    s.visible_checksum,
                    s.policy_checksum,
                    old.config_checksum,
                    docs.root.checksum,
                )
                or "review" not in g.roles
                or any(r.deadline > g.deadline or r.api_requests != 1 for r in all_records)
            ):
                raise ValueError("review_continuation_original_permission")
            for record, batch in zip(records, sub.batches, strict=True):
                spec = next(b for b in sub.plan.batches if b.id == batch.batch_id)
                response = read_response(docs.read(response_key(sub.plan, spec))[0])
                config = response.provider_config
                if (record.provider, record.model, record.endpoint) != (
                    config.transport.provider,
                    config.transport.model,
                    config.transport.endpoint,
                ):
                    raise ValueError("review_continuation_original_provider")
        cls = HistoricalReview
        if format_proof is not None:
            from .review_format_continuation import FormatHistoricalReview

            cls = FormatHistoricalReview
        origin = cls(
            run_id=docs.run_id,
            state_checksum=digest(old),
            control_checksum=digest(control),
            journal_checksum=digest(events),
            grant_checksum=digest(grant) if grant else None,
            semantic_checksum=semantic_binding(state),
            submission=sub,
            **(format_proof or {}),
        )
        if (
            self.expected_format_origin is not None
            and digest(origin) != self.expected_format_origin
        ):
            raise ValueError("review_format_frozen_origin_changed")
        return origin

    def _check_consumption(self, old, state):
        if self.documents.consumed(old.material.scope.graph):
            raise ValueError("review_continuation_already_accepted")


class AcceptedPartialConfirmedReader(ConfirmedReviewReader):
    """Explicit method successor: original accepted partial receipt is mandatory.

    Reuses the unchanged response provenance, not an expired execution grant.
    Ordinary ConfirmedReviewReader still refuses consumed parents.
    """

    def _check_consumption(self, old, state):
        from .accepted_aggregation import AcceptedOriginReader
        from .partial_finalization import DEADLINE_ACCEPTANCE_KEY, RESULT_KEY, original_failure

        docs = self.documents
        saved = docs.read(RESULT_KEY)
        if not saved or len(saved) != 1 or self.format_config is not None:
            raise ValueError("review_continuation_partial_closure_required")
        local = saved[0]
        failure = original_failure(docs)
        phase = local.get("phases", [{}])[-1]
        if (
            local.get("schema_version") != "multimodal-partial-finalization-result/1.1.0"
            or local.get("complete") is not True
            or local.get("new_requests") != 0
            or local.get("original_result_checksum") != digest(failure)
            or phase.get("run_id") != docs.run_id
            or phase.get("target") is not None
            or phase.get("accepted") is not True
            or phase.get("acceptance_key") != DEADLINE_ACCEPTANCE_KEY
            or local.get("review_closure_scope", {}).get("local_confirmed_only") is not True
            or local.get("review_closure_scope", {}).get("historical_answers_reaccepted")
            is not False
        ):
            raise ValueError("review_continuation_partial_closure_binding")
        reader = AcceptedOriginReader(
            documents=docs,
            requests=self.requests,
            state_reader=lambda: old,
            acceptance_key=DEADLINE_ACCEPTANCE_KEY,
        )
        # Original read verifies request/raw/source bindings, accepted scores,
        # consumed graph and exact local grant. No new permission is inferred.
        reader.read(state)


def require_reader(owner):
    from .accepted_review_reuse import AcceptedReviewReader

    reader = getattr(owner, "confirmed_review_reader", None)
    if not isinstance(reader, (ConfirmedReviewReader, AcceptedReviewReader)):
        raise ValueError("review_continuation_original_reader_required")
    return reader


def reused_ids(origin):
    from .accepted_review_reuse import AcceptedReviewOrigin

    if isinstance(origin, AcceptedReviewOrigin):
        return set(origin.assessment_ids)
    return {i.assessment_id for i in origin.submission.candidate.items}


def aggregate_continued(state, plan, results, origin):
    from packages.llm_provider.models import ReasonedReviewProposal

    from .accepted_review_reuse import (
        AcceptedReviewOrigin,
        AcceptedReviewOriginV11,
        AcceptedReviewOriginV12,
        AcceptedReviewOriginV13,
        AcceptedReviewOriginV14,
        AcceptedReviewOriginV15,
        AcceptedReviewOriginV16,
        ReusedAcceptedSubmission,
        ReusedAcceptedSubmissionV7,
        ReusedAcceptedSubmissionV9,
        ReusedAcceptedSubmissionV10,
        ReusedAcceptedSubmissionV14,
        ReusedAcceptedSubmissionV15,
        ReusedAcceptedSubmissionV16,
    )

    old_ids = reused_ids(origin)
    from .review_batches import ReusePartitionPlan, check_plan

    if not results:
        check_plan(state, plan)
        if (
            not isinstance(origin, AcceptedReviewOrigin)
            or not old_ids
            or old_ids != set(plan.assessment_ids)
            or origin.semantic_checksum != semantic_binding(state)
        ):
            raise ValueError("all_reused_current_coverage_required")
        if isinstance(plan, ReusePartitionPlan) and (
            plan.origin_checksum != digest(origin) or set(plan.reused_assessment_ids) != old_ids
        ):
            raise ValueError("review_partition_origin_binding")
        return ReusedAcceptedSubmissionV16(
            scope=plan.scope,
            prompt_checksum=plan.prompt_checksum,
            config_checksum=plan.config_checksum,
            observation_checksums=tuple(o.reference for o in state.material.observations),
            source_checksums=tuple(s.reference for s in state.material.sources),
            plan=plan,
            batches=(),
            historical=origin,
            composition="all_current_items_reused",
            candidate=ReasonedReviewProposal(
                scope_checksum=digest(plan.scope),
                round=plan.round,
                items=tuple(
                    sorted(
                        (
                            i
                            for i in origin.submission.candidate.items
                            if i.assessment_id in old_ids
                        ),
                        key=lambda i: i.assessment_id,
                    )
                ),
            ),
        )
    current = aggregate(state, plan, results, partial=True)

    if isinstance(plan, ReusePartitionPlan) and (
        plan.origin_checksum != digest(origin) or set(plan.reused_assessment_ids) != old_ids
    ):
        raise ValueError("review_partition_origin_binding")
    new_ids = {i.assessment_id for i in current.candidate.items}
    if (
        origin.semantic_checksum != semantic_binding(state)
        or old_ids & new_ids
        or old_ids | new_ids != set(plan.assessment_ids)
        or any(
            set(b.assessment_ids) & old_ids and not set(b.assessment_ids) <= old_ids
            for b in plan.batches
        )
        or {b.id for b in plan.batches if set(b.assessment_ids) <= old_ids}
        != set(current.pending_batch_ids)
    ):
        raise ValueError("review_continuation_partition")
    items = tuple(
        sorted(
            (
                *(i for i in origin.submission.candidate.items if i.assessment_id in old_ids),
                *current.candidate.items,
            ),
            key=lambda i: i.assessment_id,
        )
    )
    raw = current.model_dump(mode="json")
    for name in ("completion", "pending_batch_ids", "pending_semantics"):
        raw.pop(name)
    cls = (
        ReusedAcceptedSubmissionV7
        if isinstance(origin, AcceptedReviewOriginV11)
        else ReusedAcceptedSubmission
        if isinstance(origin, AcceptedReviewOrigin)
        else ContinuedReasonedSubmission
    )
    from .review_format_continuation import FormatContinuedSubmission, FormatHistoricalReview

    if isinstance(origin, FormatHistoricalReview):
        cls = FormatContinuedSubmission
    elif isinstance(origin, AcceptedReviewOriginV12):
        cls = ReusedAcceptedSubmissionV9
    if isinstance(plan, ReusePartitionPlan):
        if not isinstance(origin, AcceptedReviewOrigin):
            raise ValueError("review_partition_accepted_origin_required")
        cls = ReusedAcceptedSubmissionV10
    if isinstance(origin, AcceptedReviewOriginV14):
        cls = ReusedAcceptedSubmissionV14
    if type(origin) is AcceptedReviewOriginV15 or (
        type(origin) is AcceptedReviewOriginV13 and not isinstance(plan, ReusePartitionPlan)
    ):
        cls = ReusedAcceptedSubmissionV15
    if isinstance(origin, AcceptedReviewOriginV16):
        cls = ReusedAcceptedSubmissionV16
        raw["composition"] = "accepted_origin_with_new_batches"
    raw.update(
        schema_version=cls.model_fields["schema_version"].default,
        historical=origin,
        candidate=ReasonedReviewProposal(
            scope_checksum=digest(plan.scope), round=plan.round, items=items
        ),
    )
    return validate(cls, raw)


def check_continued(state, submission, *, owner, documents, requests):
    reader = require_reader(owner)
    origin = reader.current(state)
    if (
        origin != submission.historical
        or aggregate_continued(state, submission.plan, submission.batches, origin) != submission
    ):
        raise ValueError("review_continuation_origin_changed")
    if not submission.batches:
        # No request (including UNKNOWN), local claim, response or batch result
        # may be hidden behind an all-reused composition.
        if (
            requests.records()
            or any(e.get("phase") in {"reserved", "completed"} for e in requests.journal.read())
            or any(
                documents.read("review-batches:" + submission.plan.key + ":" + b.id + suffix)
                for b in submission.plan.batches
                for suffix in (":claim", ":response", ":result")
            )
        ):
            raise ValueError("all_reused_unexpected_dispatch")
        return ()
    current = aggregate(state, submission.plan, submission.batches, partial=True)
    records = validate_batch_receipts(state, current, documents, requests)
    if len({r.simulated for r in (*records, *reader.requests.records())}) != 1:
        raise ValueError("review_continuation_execution_identity")
    return records


class ContinuationOwner:
    def __init__(self, owner, reader):
        self.owner, self.confirmed_review_reader = owner, reader

    def __getattr__(self, name):
        return getattr(self.owner, name)

    @property
    def role(self):
        return self.owner.role

    @role.setter
    def role(self, value):
        self.owner.role = value
