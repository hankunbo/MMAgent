"""Explicit zero-wire interpretation of identical repeated explanation fields.

Old JSON parsing, response bytes, request errors and grants stay immutable. This
port is not a retry, a score repair, or permission to choose conflicting values.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field

from packages.grounding.models import Checksum, Value, digest, validate
from packages.llm_provider.review_response import decode
from packages.object_store import checksum

from .acceptance_models import AcceptedRecordsV25
from .finalization import FinalizationGrant
from .live_permission import ResearchLiveGrant, ResearchLivePermission, validate_live_config
from .recorded_finalization import RecordedBatchRequests as RecordedBatchRequests
from .review_batches import (
    ReasonedBatchResult,
    ReasonedSubmission,
    aggregate,
    batch_payload,
    check_batch,
    check_plan,
    proposal_type,
    read_response,
)

POLICY = "identical-review-explanation-keys/1.0.0"
GRANT_KEY = "recorded-explanation:grant:1"
SUBMISSION_KEY = "recorded-explanation:submission:1"
ACCEPTANCE_KEY = "recorded-explanation:acceptance:1"
RESULT_KEY = "recorded-explanation:result:1"
REPORT_KEY = "recorded-explanation:report:1"
PARTITION_GRANT_KEY = "not_a_partition_grant"


class RepeatedExplanation(Value):
    item_index: int = Field(ge=0, le=4)
    field: Literal["quality_limitation", "resolvable_gap", "next_action"]


class ExplanationInterpretation(Value):
    schema_version: Literal["recorded-explanation-interpretation/1.0.0"] = (
        "recorded-explanation-interpretation/1.0.0"
    )
    policy: Literal["identical-review-explanation-keys/1.0.0"] = POLICY
    raw_checksum: Checksum
    normalized_checksum: Checksum
    candidate_checksum: Checksum
    repeated_fields: tuple[RepeatedExplanation, ...] = Field(min_length=1, max_length=64)
    scores_citations_status_unchanged: Literal[True] = True


class ExplanationSubmission(ReasonedSubmission):
    schema_version: Literal["research-batched-submission/11.0.0"] = (
        "research-batched-submission/11.0.0"
    )
    local_interpretation: ExplanationInterpretation
    original_record_checksum: Checksum


class ExplanationGrant(FinalizationGrant):
    schema_version: Literal["recorded-explanation-grant/1.0.0"] = "recorded-explanation-grant/1.0.0"
    purpose: Literal["save_identical_explanation_interpretation_only"] = (
        "save_identical_explanation_interpretation_only"
    )
    interpretation_checksum: Checksum


class ExplanationAcceptedRecords(AcceptedRecordsV25):
    schema_version: Literal["case-observation-interpretation/2.13.0"] = (
        "case-observation-interpretation/2.13.0"
    )
    local_interpretation: ExplanationInterpretation
    original_record_checksum: Checksum


def interpret(raw_text, config):
    if not isinstance(raw_text, str) or len(raw_text.encode()) > 262144:
        raise ValueError("explanation_response_size")

    class Pairs(list):
        pass

    repeated = []

    def walk(value, path=()):
        if len(path) > 64:
            raise ValueError("explanation_json_depth")
        if isinstance(value, Pairs):
            result = {}
            for key, child in value:
                child = walk(child, (*path, key))
                if key in result:
                    if not (
                        len(path) == 3
                        and path[0] == "items"
                        and type(path[1]) is int
                        and path[2] == "direction_assessment"
                        and key in {"quality_limitation", "resolvable_gap", "next_action"}
                        and type(child) in {str, type(None)}
                        and type(result[key]) is type(child)
                        and result[key] == child
                    ):
                        raise ValueError("explanation_conflicting_or_forbidden_duplicate")
                    repeated.append(RepeatedExplanation(item_index=path[1], field=key))
                result[key] = child
            return result
        if isinstance(value, list):
            return [walk(v, (*path, i)) for i, v in enumerate(value)]
        return value

    def invalid_constant(_):
        raise ValueError("explanation_nonfinite_json")

    normalized = walk(
        json.loads(raw_text, object_pairs_hook=Pairs, parse_constant=invalid_constant)
    )
    if not repeated:
        raise ValueError("explanation_no_redundant_fields")
    text = json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    candidate, _ = decode(text, config)  # Original shape, optional-null and medical contracts.
    return candidate, ExplanationInterpretation(
        raw_checksum=checksum(raw_text.encode()),
        normalized_checksum=checksum(text.encode()),
        candidate_checksum=digest(candidate),
        repeated_fields=tuple(repeated),
    )


def original_failure(documents):
    values = documents.read("multimodal-case-result")
    if not values or len(values) != 1:
        raise ValueError("explanation_original_result_missing")
    result = values[0]
    phases = result.get("phases", [])
    if (
        result.get("complete") is not False
        or not phases
        or phases[-1].get("target") is not None
        or phases[-1].get("accepted") is not False
        or phases[-1].get("run_id") != documents.run_id
        or phases[-1].get("error") != "candidate_failed_or_context_changed"
        or documents.read("dynamic-phase-result") != (phases[-1],)
    ):
        raise ValueError("explanation_original_failure_binding")
    return result


def closure_keys(_documents):
    return GRANT_KEY, SUBMISSION_KEY, ACCEPTANCE_KEY


def reconstruct(state, documents, requests, config):
    from agents.expert_agent.models import ReviewFrame
    from packages.llm_provider.http_recovery import observation_for
    from packages.planner_control.models import Reference

    from .acceptance_models import DomainState

    original_failure(documents)
    intent = documents.read("live-stage-intent:review")
    if not intent or len(intent) != 1:
        raise ValueError("explanation_review_intent_missing")
    stage = intent[0]
    event = validate(Reference, stage["event"])
    saved = documents.read("review:" + event.code + ":intent")
    if not saved or len(saved) != 1:
        raise ValueError("explanation_original_parent_missing")
    frame = validate(ReviewFrame, saved[0]["frame"])
    if (
        stage["role"] != "review"
        or stage["config_checksum"] != digest(config)
        or validate(DomainState, saved[0]["state"]) != state
        or frame.event != event
        or frame.scope != state.material.scope
        or not any(
            a.id == stage["proposed_action_id"]
            and a.kind == "request_review"
            and a.branch == "research_review"
            for a in frame.actions.actions
        )
        or documents.read("review:" + event.code + ":command")
        != (
            dict(
                event=stage["event"],
                proposed_action_id=stage["proposed_action_id"],
                role="review",
                abc=False,
            ),
        )
        or documents.read("accepted:" + event.code + ":domain")
        or documents.read("submission:" + event.code + ":domain")
    ):
        raise ValueError("explanation_intent_binding")
    key = event.code + ":A"
    saved = documents.read("review-batches:" + key + ":plan")
    if not saved:
        raise ValueError("explanation_plan_missing")
    plan = check_plan(state, saved[0])
    records = requests.records()
    if (
        len(plan.batches) != 1
        or len(records) != 1
        or plan.policy != config.review_batching
        or plan.key != key
        or len(saved) != 1
    ):
        raise ValueError("explanation_single_complete_batch_required")
    batch, record = plan.batches[0], records[0]
    slot = "review-batches:" + key + ":" + batch.id
    values = documents.read(slot + ":attempt:" + record.request_id)
    if not values or len(values) != 1:
        raise ValueError("explanation_original_response_missing")
    response = read_response(values[0])
    if (
        record.error != "schema"
        or record.retry_index != 0
        or record.api_requests != 1
        or record.role != "review"
        or record.scope != state.material.scope
        or record.config_checksum != digest(config)
        or record.prompt_checksum != state.prompt_checksum
        or record.input_checksum != digest(batch_payload(state, plan, batch))
        or response.error != "schema"
        or response.finish_reason != "stop"
        or response.record_checksum != digest(record)
        or response.request_id != record.request_id
        or response.provider_config != config
        or response.raw_checksum != checksum(response.raw_model_text.encode())
        or record.output_checksum != digest(response.raw_model_text.encode().hex())
        or documents.read(slot + ":claim")
        != (dict(plan_checksum=digest(plan), input_checksum=record.input_checksum),)
        or documents.read(slot + ":response")
        or documents.read(slot + ":result")
    ):
        raise ValueError("explanation_original_request_binding")
    if not record.simulated:
        http = observation_for(requests.journal, record.request_id)
        if (
            not http
            or http.get("http_status") != 200
            or http.get("origin") != "http_worker"
            or http.get("operation") != "chat_completions"
        ):
            raise ValueError("explanation_complete_response_required")
    candidate, proof = interpret(response.raw_model_text, config)
    check_batch(state, plan, batch, candidate)
    result = ReasonedBatchResult(
        plan_checksum=digest(plan),
        batch_id=batch.id,
        request_id=record.request_id,
        input_checksum=record.input_checksum,
        output_schema_checksum=digest(proposal_type(state.policy).model_json_schema()),
        response=documents.reference(
            slot + ":attempt:" + record.request_id,
            0,
            values[0],
            role="response",
            version=response.schema_version,
        ),
        candidate=candidate,
    )
    sub = aggregate(state, plan, (result,))
    return ExplanationSubmission(
        **dict(
            sub.model_dump(mode="python"),
            schema_version="research-batched-submission/11.0.0",
            local_interpretation=proof,
            original_record_checksum=digest(record),
        )
    ), records


def check_submission(state, submission, documents, requests):
    from packages.llm_provider.config import read_provider_config

    result = submission.batches[0]
    key = (
        "review-batches:"
        + submission.plan.key
        + ":"
        + result.batch_id
        + ":attempt:"
        + result.request_id
    )
    raw = documents.read(key)
    if not raw:
        raise ValueError("explanation_original_response_missing")
    expected, records = reconstruct(
        state, documents, requests, read_provider_config(raw[0]["provider_config"])
    )
    if expected != submission:
        raise ValueError("explanation_submission_changed")
    return records


class RecordedReviewPermission(ResearchLivePermission):
    def __init__(self, *, grant, config, documents, requests, clock=None):
        self.grant = validate(ExplanationGrant, grant)
        self.config = validate_live_config(config)
        self.documents, self.budget = documents, requests
        self.clock = clock or (lambda: datetime.now(UTC))
        if requests.journal.documents is not documents or not requests.journal.durable:
            raise ValueError("explanation_durable_owner_required")

    @classmethod
    def prepare(cls, *, owner, documents, requests, config, clock=None):
        state = owner.current()
        if documents.consumed(state.material.scope.graph):
            raise ValueError("explanation_parent_already_consumed")
        sub, records = reconstruct(state, documents, requests, config)
        control = documents.sessions.read(documents.run_id)
        original = validate(ResearchLiveGrant, documents.read("research-live-grant")[0])
        expected = dict(
            run_id=documents.run_id,
            case_code=state.material.scope.case_code,
            original_grant_checksum=digest(original),
            failed_result_checksum=digest(original_failure(documents)),
            control_checksum=digest(control),
            state_checksum=digest(state),
            aggregate_key=SUBMISSION_KEY,
            aggregate_checksum=digest(sub),
            request_checksums=tuple(digest(r) for r in records),
            interpretation_checksum=digest(sub.local_interpretation),
        )
        now = (clock or (lambda: datetime.now(UTC)))()
        with documents.session():
            prior = documents.read(GRANT_KEY)
            grant = (
                validate(ExplanationGrant, prior[0])
                if prior
                else ExplanationGrant(
                    **expected, issued_at=now, deadline=now + timedelta(minutes=15)
                )
            )
            if any(getattr(grant, k) != v for k, v in expected.items()):
                raise ValueError("explanation_key_content_conflict")
            permission = cls(
                grant=grant, config=config, documents=documents, requests=requests, clock=clock
            )
            permission._check(state, role="review", persisted=False)
            documents.write(SUBMISSION_KEY, (sub,))
            documents.write(GRANT_KEY, (grant,))
        return permission, sub

    def _check(self, state, *, role, record=None, persisted=True):
        g, docs = self.grant, self.documents
        control = docs.sessions.read(docs.run_id)
        old = validate(ResearchLiveGrant, docs.read("research-live-grant")[0])
        s = state.material.scope
        if (
            self.clock() >= g.deadline
            or role != "review"
            or digest(state) != g.state_checksum
            or control is None
            or control.phase != "stopped"
            or control.pending
            or control.cancelled
            or digest(control) != g.control_checksum
            or digest(old) != g.original_grant_checksum
            or digest(original_failure(docs)) != g.failed_result_checksum
            or (s.run_id, s.case_code) != (g.run_id, g.case_code)
            or (s.run_id, s.case_code, s.input_checksum, s.visible_checksum, s.policy_checksum)
            != (
                old.run_id,
                old.case_code,
                old.input_checksum,
                old.visible_checksum,
                old.policy_checksum,
            )
            or digest(self.config) != old.config_checksum
            or state.config_checksum != old.config_checksum
            or old.root_checksum != docs.root.checksum
            or "review" not in old.roles
            or s.graph.kind != "research_artifact"
            or any(x.purpose != "research_material" for x in state.material.sources)
        ):
            raise ValueError("explanation_local_permission_binding")
        sub, records = reconstruct(state, docs, self.budget, self.config)
        if (digest(sub), tuple(digest(r) for r in records), digest(sub.local_interpretation)) != (
            g.aggregate_checksum,
            g.request_checksums,
            g.interpretation_checksum,
        ):
            raise ValueError("explanation_records_changed")
        if persisted and (
            docs.read(GRANT_KEY) != (g.model_dump(mode="json"),)
            or docs.read(SUBMISSION_KEY) != (sub.model_dump(mode="json"),)
        ):
            raise ValueError("explanation_grant_changed")
        for r in records:
            if (
                r.simulated
                or r.deadline > old.deadline
                or (r.provider, r.model, r.endpoint)
                != (
                    self.config.transport.provider,
                    self.config.transport.model,
                    self.config.transport.endpoint,
                )
            ):
                raise ValueError("explanation_original_provider_binding")
        if record is not None and record not in records:
            raise ValueError("explanation_unknown_request")
        docs.validate_scope(s)
        if self.clock() >= g.deadline:
            raise ValueError("explanation_local_deadline")

    def check(self, state, *, role=None, record=None):
        self._check(state, role=role, record=record)
