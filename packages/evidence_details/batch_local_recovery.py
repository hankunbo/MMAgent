"""Recover confirmed raw batches after a stopped deadline, without dispatch.

Only missing local result envelopes are appended. This is not acceptance,
scoring, a renewed deadline, or permission to execute the unrequested batches.
"""

from packages.grounding.models import digest, validate
from packages.object_store import checksum

from .acceptance_models import DomainState
from .partial_finalization import original_failure
from .review_batches import (
    ReasonedBatchResult,
    ReasonedReviewBatch,
    ReviewBatchResult,
    batch_payload,
    check_batch,
    check_plan,
    read_batch_result,
    read_plan,
    read_response,
    response_candidate,
    response_key,
)

KEY = "confirmed-raw-batch-local-recovery:1"


def recover(*, owner, documents, requests):
    """Current owner and all original records must agree before any append."""
    if getattr(requests.journal, "documents", None) is not documents:
        raise ValueError("batch_local_recovery_journal_owner")
    state = validate(DomainState, owner.current())
    original = original_failure(documents)
    control = documents.sessions.read(documents.run_id)
    if (
        original["phases"][-1]["error"] != "deadline"
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
        or documents.read("dynamic-phase-input") != (state.model_dump(mode="json"),)
    ):
        raise ValueError("batch_local_recovery_stopped_parent")
    documents.validate_scope(state.material.scope)
    intent = documents.read("live-stage-intent:review")
    if not intent or len(intent) != 1:
        raise ValueError("batch_local_recovery_intent")
    prefix = "review-batches:" + intent[0]["event"]["code"] + ":A"
    raw_plan = documents.read(prefix + ":plan")
    if not raw_plan or len(raw_plan) != 1 or documents.read(prefix + ":historical"):
        raise ValueError("batch_local_recovery_local_plan_required")
    plan = check_plan(state, read_plan(raw_plan[0]))
    if "review-batches:" + plan.key != prefix or plan.round != "A":
        raise ValueError("batch_local_recovery_plan_identity")
    records = requests.records()
    events = requests.journal.read()
    if not records or any(r.error or r.retry_index for r in records):
        raise ValueError("batch_local_recovery_confirmed_only")
    indexed = {r.request_id: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("batch_local_recovery_duplicate_request")
    results, missing, used, bindings = [], [], set(), []
    originals = {"live-stage-intent:review": intent, prefix + ":plan": raw_plan}
    for batch in plan.batches:
        slot = prefix + ":" + batch.id
        raw_key = response_key(plan, batch)
        raw, saved, claim = (
            documents.read(k) for k in (raw_key, slot + ":result", slot + ":claim")
        )
        originals.update({raw_key: raw, slot + ":result": saved, slot + ":claim": claim})
        payload_checksum = digest(batch_payload(state, plan, batch))
        if raw is None:
            if saved or claim or any(r.input_checksum == payload_checksum for r in records):
                raise ValueError("batch_local_recovery_unknown_no_repeat")
            continue
        if len(raw) != 1 or claim != (
            dict(plan_checksum=digest(plan), input_checksum=payload_checksum),
        ):
            raise ValueError("batch_local_recovery_response_claim")
        response = read_response(raw[0])
        record = indexed.get(response.request_id)
        if (
            record is None
            or record.request_id in used
            or record.scope != plan.scope
            or record.role != "review"
            or record.input_checksum != payload_checksum
            or record.config_checksum != plan.config_checksum
            or record.prompt_checksum != plan.prompt_checksum
            or response.error is not None
            or response.record_checksum != digest(record)
            or response.raw_checksum != checksum(response.raw_model_text.encode())
        ):
            raise ValueError("batch_local_recovery_response_binding")
        candidate = check_batch(
            state,
            plan,
            batch,
            response_candidate(
                response, config_checksum=plan.config_checksum, prompt_checksum=plan.prompt_checksum
            ),
        )
        if record.output_checksum != digest(candidate):
            raise ValueError("batch_local_recovery_output_binding")
        result = (
            ReasonedBatchResult if isinstance(candidate, ReasonedReviewBatch) else ReviewBatchResult
        )(
            plan_checksum=digest(plan),
            batch_id=batch.id,
            request_id=record.request_id,
            input_checksum=record.input_checksum,
            output_schema_checksum=digest(type(candidate).model_json_schema()),
            response=documents.reference(
                raw_key,
                0,
                response.model_dump(mode="json"),
                role="response",
                version=response.schema_version,
            ),
            candidate=candidate,
        )
        if saved is not None:
            if len(saved) != 1 or read_batch_result(saved[0]) != result:
                raise ValueError("batch_local_recovery_result_conflict")
        else:
            missing.append((slot + ":result", result))
        results.append(result)
        used.add(record.request_id)
        bindings.append(
            dict(
                batch_id=batch.id,
                request_checksum=digest(record),
                response_checksum=digest(response),
                result_checksum=digest(result),
            )
        )
    if used != set(indexed):
        raise ValueError("batch_local_recovery_unaccounted_request")
    expected = dict(
        schema_version="confirmed-raw-batch-local-recovery/1.0.0",
        run_id=documents.run_id,
        state_checksum=digest(state),
        plan_checksum=digest(plan),
        control_checksum=digest(control),
        original_failure_checksum=digest(original),
        journal_checksum=digest(events),
        batches=bindings,
        new_api_requests=0,
        accepted_medical_evidence=False,
        new_score=False,
    )
    receipt = documents.read(KEY)
    if receipt is not None and receipt != (expected,):
        raise ValueError("batch_local_recovery_receipt_conflict")
    if missing and documents.consumed(state.material.scope.graph):
        raise ValueError("batch_local_recovery_parent_consumed")
    if validate(DomainState, owner.current()) != state:
        raise ValueError("batch_local_recovery_current_parent_changed")
    # No source/remote owner calls inside this short original Runtime transaction.
    with documents.session():
        if (
            documents.sessions.read(documents.run_id) != control
            or requests.journal.read() != events
            or original_failure(documents) != original
            or any(documents.read(k) != v for k, v in originals.items())
        ):
            raise ValueError("batch_local_recovery_concurrent_change")
        for key, result in missing:
            documents.write(key, (result,))
        documents.write(KEY, (expected,))
    return expected
