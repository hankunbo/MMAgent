"""Owner-derived finite review partitions and explicit multi-request evidence.

The complete parent Material is retained in each request. Only task definitions
are partitioned; fixed ratings and old interpretations never enter the payload.
"""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value, digest, validate
from packages.llm_provider.config import ReviewBatchPolicy
from packages.llm_provider.models import ReviewProposal, RolePayload, Scope
from packages.planner_control.models import Reference


class ReviewBatchSpec(Value):
    id: Token
    assessment_ids: tuple[Token, ...] = Field(min_length=1, max_length=5)


class ReviewBatchPlan(Value):
    schema_version: Literal["review-batch-plan/1.0.0"] = "review-batch-plan/1.0.0"
    key: Token
    scope: Scope
    round: Literal["A", "B", "C"]
    template_checksum: Checksum
    prompt_checksum: Checksum
    config_checksum: Checksum
    policy: ReviewBatchPolicy
    assessment_ids: tuple[Token, ...] = Field(min_length=1, max_length=128)
    fixed_assessment_ids: tuple[Token, ...] = Field(max_length=128)
    batches: tuple[ReviewBatchSpec, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def exact_partition(self):
        ids = self.assessment_ids
        if ids != tuple(sorted(set(ids))) or set(ids) & set(self.fixed_assessment_ids):
            raise ValueError("batch_inventory")
        expected = tuple(
            ReviewBatchSpec(
                id=f"batch:{n // self.policy.batch_size:03d}",
                assessment_ids=ids[n : n + self.policy.batch_size],
            )
            for n in range(0, len(ids), self.policy.batch_size)
        )
        if self.batches != expected or len(expected) > self.policy.max_batches:
            raise ValueError("batch_partition")
        return self


class BatchIdentity(Value):
    plan_checksum: Checksum
    batch_id: Token
    parent_template_checksum: Checksum


class ReusePartitionPlan(ReviewBatchPlan):
    """Owner-bound reuse and new-work partitions; never mix them in one request."""

    schema_version: Literal["review-batch-plan/1.1.0"] = "review-batch-plan/1.1.0"
    reused_assessment_ids: tuple[Token, ...] = Field(min_length=1, max_length=128)
    origin_checksum: Checksum

    @model_validator(mode="after")
    def exact_partition(self):
        ids, reused = self.assessment_ids, self.reused_assessment_ids
        if (
            ids != tuple(sorted(set(ids)))
            or reused != tuple(sorted(set(reused)))
            or not set(reused) < set(ids)
            or set(ids) & set(self.fixed_assessment_ids)
        ):
            raise ValueError("reuse_batch_inventory")
        expected = partition_specs(ids, reused, self.policy.batch_size)
        if self.batches != expected or len(expected) > self.policy.max_batches:
            raise ValueError("reuse_batch_partition")
        return self


def partition_specs(ids, reused, size):
    return tuple(
        ReviewBatchSpec(id=f"batch:{kind}:{n // size:03d}", assessment_ids=group[n : n + size])
        for kind, group in (
            ("reused", tuple(i for i in ids if i in reused)),
            ("new", tuple(i for i in ids if i not in reused)),
        )
        for n in range(0, len(group), size)
    )


def read_plan(value):
    raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    cls = {
        "review-batch-plan/1.0.0": ReviewBatchPlan,
        "review-batch-plan/1.1.0": ReusePartitionPlan,
    }.get(raw.get("schema_version"))
    if cls is None:
        raise ValueError("review_batch_plan_version")
    return validate(cls, raw)


class ReviewBatchPayload(RolePayload):
    schema_version: Literal["review-batch-input/1.0.0"] = "review-batch-input/1.0.0"
    role: Literal["review"] = "review"
    batch: BatchIdentity


class ReviewBatchProposal(ReviewProposal):
    schema_version: Literal["research-review-batch/1.0.0"] = "research-review-batch/1.0.0"
    plan_checksum: Checksum
    batch_id: Token


class ReviewBatchResult(Value):
    schema_version: Literal["review-batch-result/1.0.0"] = "review-batch-result/1.0.0"
    plan_checksum: Checksum
    batch_id: Token
    request_id: Token
    input_checksum: Checksum
    output_schema_checksum: Checksum
    response: Reference
    candidate: ReviewBatchProposal


class ReasonedReviewBatch(ReviewBatchProposal):
    from packages.evidence_scoring.review_protocol import ReasonedReviewItem as _Item

    schema_version: Literal["research-review-batch/2.0.0"] = "research-review-batch/2.0.0"
    items: tuple[_Item, ...] = Field(min_length=1, max_length=128)


class ReasonedBatchResult(ReviewBatchResult):
    schema_version: Literal["review-batch-result/2.0.0"] = "review-batch-result/2.0.0"
    candidate: ReasonedReviewBatch


def proposal_type(owner):
    return ReasonedReviewBatch if getattr(owner, "direction_policy", None) else ReviewBatchProposal


def read_batch_result(value):
    raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    cls = {
        "review-batch-result/1.0.0": ReviewBatchResult,
        "review-batch-result/2.0.0": ReasonedBatchResult,
    }.get(raw.get("schema_version"))
    if cls is None:
        raise ValueError("batch_result_version")
    return validate(cls, raw)


class ReviewBatchResponse(Value):
    schema_version: Literal["review-batch-response/1.0.0"] = "review-batch-response/1.0.0"
    request_id: Token
    record_checksum: Checksum
    raw_model_text: str = Field(max_length=262144)
    raw_checksum: Checksum
    error: Token | None
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "function_call"] | None
    untrusted_model_output: Literal[True] = True
    accepted_medical_evidence: Literal[False] = False


class ReviewBatchResponseV11(ReviewBatchResponse):
    from packages.llm_provider.config import ReviewShapeProviderConfig as _Config
    from packages.llm_provider.review_response import ReviewResponseAlignment as _Alignment

    schema_version: Literal["review-batch-response/1.1.0"] = "review-batch-response/1.1.0"
    provider_config: _Config
    alignment: _Alignment | None


class ReviewBatchResponseV12(ReviewBatchResponseV11):
    from packages.llm_provider.config import IntegratedReviewProviderConfig as _Config

    schema_version: Literal["review-batch-response/1.2.0"] = "review-batch-response/1.2.0"
    provider_config: _Config


class ReviewBatchResponseV13(ReviewBatchResponseV12):
    from packages.llm_provider.config import SourceQuoteProviderConfig as _Config

    schema_version: Literal["review-batch-response/1.3.0"] = "review-batch-response/1.3.0"
    provider_config: _Config


class ReviewBatchResponseV14(ReviewBatchResponseV13):
    from packages.llm_provider.config import DirectionalProviderConfig as _Config

    schema_version: Literal["review-batch-response/1.4.0"] = "review-batch-response/1.4.0"
    provider_config: _Config


class ReviewBatchResponseV15(ReviewBatchResponseV14):
    from packages.llm_provider.config import NullableReviewProviderConfig as _Config
    from packages.llm_provider.review_response import NullableReviewAlignment as _Alignment

    schema_version: Literal["review-batch-response/1.5.0"] = "review-batch-response/1.5.0"
    provider_config: _Config
    alignment: _Alignment | None


class ReviewBatchResponseV131(ReviewBatchResponseV13):
    from packages.llm_provider.config import RoutedSourceProviderConfig as _Config

    schema_version: Literal["review-batch-response/1.3.1"] = "review-batch-response/1.3.1"
    provider_config: _Config


class ReviewBatchResponseV151(ReviewBatchResponseV15):
    from packages.llm_provider.config import RoutedNullableProviderConfig as _Config

    schema_version: Literal["review-batch-response/1.5.1"] = "review-batch-response/1.5.1"
    provider_config: _Config


class ReviewBatchResponseV132(ReviewBatchResponseV13):
    from packages.llm_provider.config import CurlSourceProviderConfig as _Config

    schema_version: Literal["review-batch-response/1.3.2"] = "review-batch-response/1.3.2"
    provider_config: _Config


class ReviewBatchResponseV152(ReviewBatchResponseV15):
    from packages.llm_provider.config import CurlNullableProviderConfig as _Config

    schema_version: Literal["review-batch-response/1.5.2"] = "review-batch-response/1.5.2"
    provider_config: _Config


def read_response(value):
    cls = {
        "review-batch-response/1.0.0": ReviewBatchResponse,
        "review-batch-response/1.1.0": ReviewBatchResponseV11,
        "review-batch-response/1.2.0": ReviewBatchResponseV12,
        "review-batch-response/1.3.0": ReviewBatchResponseV13,
        "review-batch-response/1.3.1": ReviewBatchResponseV131,
        "review-batch-response/1.3.2": ReviewBatchResponseV132,
        "review-batch-response/1.4.0": ReviewBatchResponseV14,
        "review-batch-response/1.5.0": ReviewBatchResponseV15,
        "review-batch-response/1.5.1": ReviewBatchResponseV151,
        "review-batch-response/1.5.2": ReviewBatchResponseV152,
    }.get(value.get("schema_version"))
    if cls is None:
        raise ValueError("batch_response_version")
    return validate(cls, value)


def response_candidate(response, *, config_checksum, prompt_checksum):
    """Reconstruct only under the exact owner/journal-bound non-secret configuration."""
    from packages.llm_provider.provider import strict_json
    from packages.llm_provider.review_response import decode
    from packages.object_store import checksum

    if (
        response.error is not None
        or checksum(response.raw_model_text.encode()) != response.raw_checksum
    ):
        raise ValueError("batch_response_binding")
    if isinstance(response, ReviewBatchResponseV11):
        config = response.provider_config
        setting = next(r for r in config.roles if r.role == "review")
        if digest(config) != config_checksum or setting.prompt_checksum != prompt_checksum:
            raise ValueError("batch_response_configuration_binding")
        candidate, proof = decode(response.raw_model_text, config)
        if proof != response.alignment:
            raise ValueError("batch_response_alignment_binding")
        return candidate
    raw = strict_json(response.raw_model_text)
    candidate = validate(ReviewBatchProposal, raw)
    if raw != candidate.model_dump(mode="json"):
        raise ValueError("batch_original_response_binding")
    return candidate


class BatchedSubmission(Value):
    schema_version: Literal["research-batched-submission/1.0.0"] = (
        "research-batched-submission/1.0.0"
    )
    scope: Scope
    actor: Literal["PLANNER"] = "PLANNER"
    correlation: Literal["SOURCE:AGENT_RESEARCH"] = "SOURCE:AGENT_RESEARCH"
    prompt_checksum: Checksum
    config_checksum: Checksum
    observation_checksums: tuple[Reference, ...]
    source_checksums: tuple[Reference, ...]
    plan: ReviewBatchPlan
    batches: tuple[ReviewBatchResult, ...] = Field(min_length=1, max_length=20)
    candidate: ReviewProposal
    status: Literal["candidate_not_accepted"] = "candidate_not_accepted"


class PartialBatchedSubmission(BatchedSubmission):
    schema_version: Literal["research-batched-submission/2.0.0"] = (
        "research-batched-submission/2.0.0"
    )
    completion: Literal["partial"] = "partial"
    pending_batch_ids: tuple[Token, ...] = Field(min_length=1, max_length=20)
    pending_semantics: Literal["not_accepted_not_negative"] = "not_accepted_not_negative"


class ReasonedSubmission(BatchedSubmission):
    from packages.llm_provider.models import ReasonedReviewProposal as _Proposal

    schema_version: Literal["research-batched-submission/3.0.0"] = (
        "research-batched-submission/3.0.0"
    )
    batches: tuple[ReasonedBatchResult, ...] = Field(min_length=1, max_length=20)
    candidate: _Proposal


class PartialReasonedSubmission(PartialBatchedSubmission):
    from packages.llm_provider.models import ReasonedReviewProposal as _Proposal

    schema_version: Literal["research-batched-submission/4.0.0"] = (
        "research-batched-submission/4.0.0"
    )
    batches: tuple[ReasonedBatchResult, ...] = Field(min_length=1, max_length=20)
    candidate: _Proposal


class _PartitionPartial(PartialReasonedSubmission):
    """Internal aggregation carrier only, not a standalone accepted submission."""

    schema_version: Literal["research-batched-partition-intermediate/1.0.0"] = (
        "research-batched-partition-intermediate/1.0.0"
    )
    plan: ReusePartitionPlan


def make_plan(state, policy, *, key, round_id="A"):
    from .review_tasks import project_review_tasks

    template = project_review_tasks(state)
    policy = validate(ReviewBatchPolicy, policy)
    ids = template.assessment_ids
    return ReviewBatchPlan(
        key=key,
        scope=state.material.scope,
        round=round_id,
        template_checksum=digest(template),
        prompt_checksum=state.prompt_checksum,
        config_checksum=state.config_checksum,
        policy=policy,
        assessment_ids=ids,
        fixed_assessment_ids=template.fixed_assessment_ids,
        batches=tuple(
            ReviewBatchSpec(
                id=f"batch:{n // policy.batch_size:03d}",
                assessment_ids=ids[n : n + policy.batch_size],
            )
            for n in range(0, len(ids), policy.batch_size)
        ),
    )


def check_plan(state, plan):
    plan = read_plan(plan)
    expected = make_plan(state, plan.policy, key=plan.key, round_id=plan.round)
    if isinstance(plan, ReusePartitionPlan):
        expected = partition_plan(expected, plan.reused_assessment_ids, plan.origin_checksum)
    if plan != expected:
        raise ValueError("batch_parent_or_configuration_changed")
    return plan


def partition_plan(plan, reused_ids, origin_checksum):
    raw = plan.model_dump(mode="python")
    reused = tuple(sorted(reused_ids))
    return ReusePartitionPlan(
        **dict(
            raw,
            schema_version="review-batch-plan/1.1.0",
            reused_assessment_ids=reused,
            origin_checksum=origin_checksum,
            batches=partition_specs(plan.assessment_ids, reused, plan.policy.batch_size),
        )
    )


def batch_payload(state, plan, batch):
    from .review_tasks import project_review_tasks, review_payload

    plan = check_plan(state, plan)
    if batch not in plan.batches:
        raise ValueError("batch_not_in_plan")
    template = project_review_tasks(state)
    template = template.model_copy(
        update={
            "tasks": tuple(t for t in template.tasks if t.assessment_id in batch.assessment_ids)
        }
    )
    raw = review_payload(state.material, template, plan.round)
    return validate(
        ReviewBatchPayload,
        dict(
            **raw,
            batch=BatchIdentity(
                plan_checksum=digest(plan),
                batch_id=batch.id,
                parent_template_checksum=plan.template_checksum,
            ),
        ),
    ).model_dump(mode="json")


def check_batch(state, plan, batch, proposal):
    from packages.evidence_scoring.accepted import rows_from_candidate

    check_plan(state, plan)
    proposal = validate(proposal_type(state.policy), proposal)
    if (proposal.scope_checksum, proposal.round, proposal.plan_checksum, proposal.batch_id) != (
        digest(plan.scope),
        plan.round,
        digest(plan),
        batch.id,
    ) or {i.assessment_id for i in proposal.items} != set(batch.assessment_ids):
        raise ValueError("batch_candidate_binding")
    rows_from_candidate(state, proposal, batch_ids=batch.assessment_ids)
    return proposal


def aggregate(state, plan, results, *, partial=False):
    check_plan(state, plan)
    from .acceptance_models import TaskPolicyV3

    if partial and not isinstance(state.policy, TaskPolicyV3):
        raise ValueError("partial_review_requires_dynamic_policy")
    by_id = {r.batch_id: r for r in results}
    pending = tuple(b.id for b in plan.batches if b.id not in by_id)
    if (
        not results
        or len(by_id) != len(results)
        or not by_id.keys() <= {b.id for b in plan.batches}
        or (partial and not pending)
        or (not partial and pending)
        or len({r.request_id for r in results}) != len(results)
    ):
        raise ValueError("batch_incomplete_or_reused_request")
    items = []
    selected = tuple(b for b in plan.batches if b.id in by_id)
    if tuple(r.batch_id for r in results) != tuple(b.id for b in selected):
        raise ValueError("batch_result_order")
    for batch, result in zip(selected, results, strict=True):
        result = read_batch_result(result)
        if (
            result.plan_checksum,
            result.batch_id,
            result.input_checksum,
            result.output_schema_checksum,
        ) != (
            digest(plan),
            batch.id,
            digest(batch_payload(state, plan, batch)),
            digest(proposal_type(state.policy).model_json_schema()),
        ):
            raise ValueError("batch_result_binding")
        items.extend(check_batch(state, plan, batch, result.candidate).items)
    from packages.llm_provider.models import ReasonedReviewProposal

    reasoned = bool(getattr(state.policy, "direction_policy", None))
    candidate = (ReasonedReviewProposal if reasoned else ReviewProposal)(
        scope_checksum=digest(plan.scope),
        round=plan.round,
        items=tuple(sorted(items, key=lambda i: i.assessment_id)),
    )
    cls = PartialBatchedSubmission if partial else BatchedSubmission
    if reasoned:
        cls = PartialReasonedSubmission if partial else ReasonedSubmission
    if isinstance(plan, ReusePartitionPlan):
        if not partial or not reasoned:
            raise ValueError("reuse_partition_requires_owner_continuation")
        cls = _PartitionPartial
    return cls(
        **(dict(pending_batch_ids=pending) if partial else {}),
        scope=plan.scope,
        prompt_checksum=plan.prompt_checksum,
        config_checksum=plan.config_checksum,
        observation_checksums=tuple(o.reference for o in state.material.observations),
        source_checksums=tuple(s.reference for s in state.material.sources),
        plan=plan,
        batches=tuple(results),
        candidate=candidate,
    )


def read_submission(value):
    from .acceptance_models import Submission
    from .accepted_aggregation import (
        AggregatedSubmission,
        AliasAggregatedSubmission,
        LegacyAggregatedSubmission,
    )
    from .accepted_review_reuse import (
        ReusedAcceptedSubmission,
        ReusedAcceptedSubmissionV7,
        ReusedAcceptedSubmissionV9,
        ReusedAcceptedSubmissionV10,
        ReusedAcceptedSubmissionV14,
        ReusedAcceptedSubmissionV15,
        ReusedAcceptedSubmissionV16,
    )
    from .input_scope import ScopeSubmission, ScopeSubmissionV2, ScopeSubmissionV3
    from .recorded_explanation import ExplanationSubmission
    from .review_continuation import ContinuedReasonedSubmission
    from .review_format_continuation import FormatContinuedSubmission
    from .scope_finalization import RecordedScopeSubmission, RecordedScopeTextSubmission

    raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    cls = {
        "research-input-scope-submission/1.0.0": ScopeSubmission,
        "research-input-scope-submission/1.1.0": ScopeSubmissionV2,
        "research-input-scope-submission/1.4.0": ScopeSubmissionV3,
        "research-input-scope-submission/1.2.0": RecordedScopeSubmission,
        "research-input-scope-submission/1.3.0": RecordedScopeTextSubmission,
        "research-submission/1.0.0": Submission,
        "research-batched-submission/1.0.0": BatchedSubmission,
        "research-batched-submission/2.0.0": PartialBatchedSubmission,
        "research-batched-submission/3.0.0": ReasonedSubmission,
        "research-batched-submission/4.0.0": PartialReasonedSubmission,
        "research-batched-submission/5.0.0": ContinuedReasonedSubmission,
        "research-batched-submission/6.0.0": ReusedAcceptedSubmission,
        "research-batched-submission/7.0.0": ReusedAcceptedSubmissionV7,
        "research-batched-submission/9.0.0": ReusedAcceptedSubmissionV9,
        "research-batched-submission/10.0.0": ReusedAcceptedSubmissionV10,
        "research-batched-submission/14.0.0": ReusedAcceptedSubmissionV14,
        "research-batched-submission/15.0.0": ReusedAcceptedSubmissionV15,
        "research-batched-submission/16.0.0": ReusedAcceptedSubmissionV16,
        "research-batched-submission/8.0.0": FormatContinuedSubmission,
        "research-batched-submission/11.0.0": ExplanationSubmission,
        "research-batched-submission/12.0.0": AggregatedSubmission,
        "research-batched-submission/13.0.0": LegacyAggregatedSubmission,
        "research-batched-submission/17.0.0": AliasAggregatedSubmission,
    }.get(raw.get("schema_version"))
    if cls is None:
        raise ValueError("research_submission_version")
    return validate(cls, raw)


def response_key(plan, batch):
    return "review-batches:" + plan.key + ":" + batch.id + ":response"


def request_for_item(submission, item_id):
    from .accepted_aggregation import AggregatedSubmission, item_requests

    if isinstance(submission, AggregatedSubmission):
        return item_requests(submission.origins)[item_id]
    from .accepted_review_reuse import ReusedAcceptedSubmission, ReusedAcceptedSubmissionV7
    from .review_continuation import ContinuedReasonedSubmission, reused_ids

    if isinstance(
        submission,
        (ContinuedReasonedSubmission, ReusedAcceptedSubmission, ReusedAcceptedSubmissionV7),
    ) and item_id in reused_ids(submission.historical):
        return request_for_item(submission.historical.submission, item_id)
    if not isinstance(submission, BatchedSubmission):
        return submission.request_id
    return next(
        r.request_id
        for r in submission.batches
        if any(i.assessment_id == item_id for i in r.candidate.items)
    )


def validate_batch_receipts(state, submission, documents, requests):
    from packages.object_store import checksum

    if (
        aggregate(
            state,
            submission.plan,
            submission.batches,
            partial=isinstance(submission, PartialBatchedSubmission),
        )
        != submission
    ):
        raise ValueError("batch_aggregate_mismatch")
    if isinstance(submission, PartialBatchedSubmission):
        # A partial finalization cannot cherry-pick favorable completed batches.
        persisted = {
            b.id
            for b in submission.plan.batches
            if documents.read("review-batches:" + submission.plan.key + ":" + b.id + ":result")
            is not None
        }
        if persisted != {r.batch_id for r in submission.batches}:
            raise ValueError("partial_review_omits_confirmed_batch")
    records = requests.records()
    simulated = set()
    checked = []
    selected = {r.batch_id for r in submission.batches}
    for batch, result in zip(
        (b for b in submission.plan.batches if b.id in selected),
        submission.batches,
        strict=True,
    ):
        matches = [r for r in records if r.request_id == result.request_id]
        if len(matches) != 1:
            raise ValueError("batch_request_missing")
        record = matches[0]
        from packages.llm_provider.review_correction import verify_correction

        verify_correction(record, requests)
        if (
            record.scope,
            record.role,
            record.prompt_checksum,
            record.config_checksum,
            record.input_checksum,
            record.output_checksum,
            record.error,
        ) != (
            submission.scope,
            "review",
            submission.prompt_checksum,
            submission.config_checksum,
            result.input_checksum,
            digest(result.candidate),
            None,
        ):
            raise ValueError("batch_request_mismatch")
        saved = documents.read(response_key(submission.plan, batch))
        if saved is None:
            raise ValueError("batch_response_missing")
        response = read_response(saved[0])
        raw = response.model_dump(mode="json")
        candidate = response_candidate(
            response,
            config_checksum=submission.config_checksum,
            prompt_checksum=submission.prompt_checksum,
        )
        if (
            raw["request_id"] != record.request_id
            or raw["record_checksum"] != digest(record)
            or raw["error"] is not None
            or checksum(raw["raw_model_text"].encode()) != raw["raw_checksum"]
            or candidate != result.candidate
            or documents.reference(
                response_key(submission.plan, batch),
                0,
                raw,
                role="response",
                version=raw["schema_version"],
            )
            != result.response
        ):
            raise ValueError("batch_original_response_binding")
        simulated.add(record.simulated)
        checked.append(record)
    if len(simulated) != 1:
        raise ValueError("mixed_batch_execution_identity")
    return tuple(checked)


def collect_partial(state, documents, key):
    """Read confirmed results only; no request, retry, control change or acceptance.

    ResearchAcceptance still rechecks original requests/raw responses and current
    sources. Pending/unknown batches remain unresolved, never marked negative.
    """
    saved = documents.read("review-batches:" + key + ":plan")
    if saved is None:
        raise ValueError("partial_review_plan_missing")
    plan = check_plan(state, read_plan(saved[0]))
    results = []
    for batch in plan.batches:
        value = documents.read("review-batches:" + key + ":" + batch.id + ":result")
        if value:
            results.append(read_batch_result(value[0]))
    return aggregate(state, plan, tuple(results), partial=True)
