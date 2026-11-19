"""Confirmed format failures are accounted work, not unknown responses or evidence.

Only the response-format 8 -> 9 transition is compatible. Source, task, prompt,
model, partition and input semantics stay exact. Failed candidates are never reused.
"""

from typing import Literal

from pydantic import Field

from packages.grounding.models import Checksum, Token, Value, digest

from .review_continuation import ContinuedReasonedSubmission, HistoricalReview


class ConfirmedFormatFailure(Value):
    batch_id: Token
    request_id: Token
    record_checksum: Checksum
    response_checksum: Checksum
    raw_checksum: Checksum
    reason: Literal["confirmed_schema_failure_not_reused"] = "confirmed_schema_failure_not_reused"


class FormatHistoricalReview(HistoricalReview):
    schema_version: Literal["confirmed-review-origin/1.1.0"] = "confirmed-review-origin/1.1.0"
    format_transition: Literal["review-optional-null-upgrade/1.0.0"] = (
        "review-optional-null-upgrade/1.0.0"
    )
    original_config_checksum: Checksum
    current_config_checksum: Checksum
    confirmed_format_failures: tuple[ConfirmedFormatFailure, ...] = Field(
        min_length=1, max_length=20
    )


class FormatContinuedSubmission(ContinuedReasonedSubmission):
    schema_version: Literal["research-batched-submission/8.0.0"] = (
        "research-batched-submission/8.0.0"
    )
    historical: FormatHistoricalReview


def verify_format_origin(reader, old, state, sub, reused_records):
    from agents.expert_agent.committed_review import confirmed_transport
    from packages.llm_provider.config import read_provider_config, review_format_upgrade
    from packages.llm_provider.review_response import decode
    from packages.object_store import checksum

    from .review_batches import batch_payload, read_response, response_key

    docs, requests = reader.documents, reader.requests
    events, records = requests.journal.read(), requests.records()
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    if (
        len(set(reserved)) != len(reserved)
        or len({r.request_id for r in records}) != len(records)
        or set(reserved) != {r.request_id for r in records}
        or digest(reader.format_config) != state.config_checksum
    ):
        raise ValueError("review_format_unaccounted_request")
    confirmed_transport(requests)
    # Recover the original full public config from a validated successful response.
    batch = next(b for b in sub.plan.batches if b.id == sub.batches[0].batch_id)
    response = read_response(docs.read(response_key(sub.plan, batch))[0])
    original_config = read_provider_config(response.provider_config)
    if digest(original_config) != old.config_checksum or review_format_upgrade(
        original_config
    ) != read_provider_config(reader.format_config):
        raise ValueError("review_format_configuration_changed")
    reused = {r.request_id for r in reused_records}
    if tuple(r for r in records if r.request_id in reused) != tuple(reused_records):
        raise ValueError("review_format_record_order")
    pending = [b for b in sub.plan.batches if b.id in sub.pending_batch_ids]
    failed = [r for r in records if r.request_id not in reused]
    proofs, covered = [], set()
    for record in failed:
        matches = [
            b for b in pending if digest(batch_payload(old, sub.plan, b)) == record.input_checksum
        ]
        if (
            len(matches) != 1
            or record.error != "schema"
            or record.retry_index != 0
            or record.scope != old.material.scope
            or record.role != "review"
            or record.config_checksum != old.config_checksum
            or record.prompt_checksum != sub.prompt_checksum
        ):
            raise ValueError("review_format_not_confirmed_schema_failure")
        batch = matches[0]
        slot = "review-batches:" + sub.plan.key + ":" + batch.id
        if (
            batch.id in covered
            or docs.read(slot + ":claim")
            != (dict(plan_checksum=digest(sub.plan), input_checksum=record.input_checksum),)
            or docs.read(response_key(sub.plan, batch)) is not None
            or docs.read(slot + ":result")
        ):
            raise ValueError("review_format_failed_batch_binding")
        raw = docs.read(slot + ":attempt:" + record.request_id)
        if not raw or len(raw) != 1:
            raise ValueError("review_format_original_response_missing")
        response = read_response(raw[0])
        if (
            response.request_id != record.request_id
            or response.record_checksum != digest(record)
            or response.error != "schema"
            or response.finish_reason != "stop"
            or response.provider_config != original_config
            or checksum(response.raw_model_text.encode()) != response.raw_checksum
            or digest(response.raw_model_text.encode().hex()) != record.output_checksum
        ):
            raise ValueError("review_format_response_binding")
        if not record.simulated:
            from packages.llm_provider.http_recovery import observation_for

            http = observation_for(requests.journal, record.request_id)
            if (
                not http
                or http.get("http_status") != 200
                or (record.provider, record.model, record.endpoint)
                != (
                    original_config.transport.provider,
                    original_config.transport.model,
                    original_config.transport.endpoint,
                )
            ):
                raise ValueError("review_format_remote_response_unconfirmed")
        try:
            decode(response.raw_model_text, original_config)
        except ValueError:
            pass  # Confirm the old schema rejection, never accept the failed candidate.
        else:
            raise ValueError("review_format_failure_not_reproducible")
        covered.add(batch.id)
        proofs.append(
            ConfirmedFormatFailure(
                batch_id=batch.id,
                request_id=record.request_id,
                record_checksum=digest(record),
                response_checksum=digest(response),
                raw_checksum=response.raw_checksum,
            )
        )
    if not proofs or any(
        docs.read("review-batches:" + sub.plan.key + ":" + b.id + ":claim") and b.id not in covered
        for b in pending
    ):
        raise ValueError("review_format_pending_unknown")
    return dict(
        original_config_checksum=old.config_checksum,
        current_config_checksum=state.config_checksum,
        confirmed_format_failures=tuple(proofs),
    )
