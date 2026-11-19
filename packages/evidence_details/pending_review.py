"""Original accepted partial Review inventory, not automatic request replay.

An unknown response quarantines its exact batch, not unrelated unstarted batches.
No score, receipt or historical request is rewritten by this read-only owner port.
"""

from types import SimpleNamespace

from agents.expert_agent.reporting import read_research_report
from packages.grounding.models import digest, validate
from packages.llm_provider.models import read_request_record
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .acceptance import ResearchAcceptance
from .live_permission import ResearchLiveGrant
from .review_batches import (
    PartialBatchedSubmission,
    PartialReasonedSubmission,
    batch_payload,
    read_submission,
    validate_batch_receipts,
)


def unknown_response_proof(documents, request_id, journal_checksum):
    """Read-only proof for an explicit new attempt, never permission to replay.

    The pending owner must separately bind this request to its exact batch and
    current material. A known HTTP response or saved output needs its own local
    recovery/error path, not this unknown-response successor.
    """
    journal = RuntimeAgentJournal(documents)
    events = journal.read()
    records = [
        read_request_record(e["record"])
        for e in events
        if e.get("phase") == "completed" and e["record"].get("request_id") == request_id
    ]
    if digest(events) != journal_checksum or len(records) != 1:
        raise ValueError("pending_unknown_response_journal")
    record = records[0]
    confirmations = [
        e.get("confirmed")
        for e in events
        if e.get("phase") == "response_confirmation" and e.get("request_id") == request_id
    ]
    http = [
        e
        for e in events
        if e.get("schema_version") == "agent-http-observation/1.0.0"
        and e.get("request_id") == request_id
    ]
    control = documents.sessions.read(documents.run_id)
    if (
        record.role != "review"
        or record.error not in {"timeout", "transport"}
        or record.api_requests != 1
        or record.retry_index != 0
        or record.output_checksum is not None
        or any(v is True for v in confirmations)
        or any(v.get("http_status") is not None for v in http)
        or control is None
        or control.phase != "stopped"
        or control.pending
        or control.cancelled
    ):
        raise ValueError("pending_unknown_response_requires_different_recovery")
    documents.validate_scope(record.scope)
    return dict(
        schema_version="pending-unknown-response-proof/1.0.0",
        run_id=documents.run_id,
        request_id=request_id,
        record_checksum=digest(record),
        journal_checksum=digest(events),
        control_checksum=digest(control),
        original_unknown_preserved=True,
        duplicate_remote_cost_possible=True,
    )


def known_truncated_failure(documents, journal, slot, records, *, allow_empty_output=False):
    """Prove a confirmed output-limit rejection, without authorizing another call.

    Missing bytes/HTTP confirmation remain quarantined. A truncation is not a
    medical answer and is never repaired by parsing a prefix or filling fields.
    """
    from packages.llm_provider.http_recovery import observation_for
    from packages.object_store import checksum

    if not records or any(r.error != "truncated" for r in records):
        return None
    if (
        len(records) > 2
        or [r.retry_index for r in records] != list(range(len(records)))
        or len({r.invocation_id for r in records}) != 1
    ):
        raise ValueError("pending_truncated_attempt_inventory")
    values, empty_ids = [], []
    for r in records:
        saved = documents.read(slot + ":attempt:" + r.request_id)
        if not saved or len(saved) != 1:
            raise ValueError("pending_truncated_response_missing")
        raw = saved[0]
        text = raw.get("raw_model_text")
        empty = allow_empty_output and text == ""
        if (
            raw.get("record_checksum") != digest(r)
            or raw.get("error") != "truncated"
            or (not text and not empty)
            or raw.get("finish_reason") != "length"
            or checksum(raw["raw_model_text"].encode()) != raw.get("raw_checksum")
            or (
                r.output_checksum is not None
                if empty
                else digest(raw["raw_model_text"].encode().hex()) != r.output_checksum
            )
            or digest(raw.get("provider_config")) != r.config_checksum
            or (not r.simulated and r.api_requests != 1)
        ):
            raise ValueError("pending_truncated_response_binding")
        if empty:
            # Opt-in proof only: some reasoning responses exhaust the token cap
            # before emitting visible text. There is no medical answer to reuse.
            # Keep legacy nonempty-byte proofs unchanged; missing confirmation,
            # unknown usage or an ordinary empty response cannot use this branch.
            from packages.llm_provider.config import read_provider_config

            config = read_provider_config(raw["provider_config"])
            role = next(item for item in config.roles if item.role == "review")
            confirmations = [
                e.get("confirmed")
                for e in journal.read()
                if e.get("phase") == "response_confirmation" and e.get("request_id") == r.request_id
            ]
            if (
                r.role != "review"
                or r.endpoint != "chat_completions"
                or r.usage.status != "observed"
                or r.usage.output_tokens != role.max_output_tokens
                or confirmations != [True]
            ):
                raise ValueError("pending_empty_truncated_confirmation_or_usage")
            empty_ids.append(r.request_id)
        if not r.simulated:
            http = observation_for(journal, r.request_id)
            if (
                not http
                or http.get("origin") != "http_worker"
                or http.get("operation") != "chat_completions"
                or http.get("http_status") != 200
            ):
                raise ValueError("pending_truncated_not_confirmed_http")
        values.append(dict(request_id=r.request_id, record=digest(r), response=digest(raw)))
    return dict(
        schema_version="confirmed-review-truncation/1.1.0"
        if empty_ids
        else "confirmed-review-truncation/1.0.0",
        last_request_id=records[-1].request_id,
        attempts=values,
        original_answer_not_accepted=True,
        **(
            dict(empty_visible_output=True, empty_visible_request_ids=empty_ids)
            if empty_ids
            else {}
        ),
    )


def known_format_failure(documents, journal, slot, records):
    """Original, confirmed rejected bytes, never a repaired clinical answer."""
    from pydantic import ValidationError

    from packages.llm_provider.http_recovery import observation_for
    from packages.llm_provider.review_response import decode
    from packages.object_store import checksum

    if not records or any(r.error != "schema" for r in records):
        return None
    if (
        len(records) > 2
        or [r.retry_index for r in records] != list(range(len(records)))
        or len({r.invocation_id for r in records}) != 1
    ):
        raise ValueError("pending_format_attempt_inventory")
    values = []
    for r in records:
        saved = documents.read(slot + ":attempt:" + r.request_id)
        if not saved or len(saved) != 1:
            raise ValueError("pending_format_response_missing")
        raw = saved[0]
        if (
            raw["record_checksum"] != digest(r)
            or raw["error"] != "schema"
            or not raw["raw_model_text"]
            or raw["finish_reason"] != "stop"
            or checksum(raw["raw_model_text"].encode()) != raw["raw_checksum"]
            # Provider records failed output as digest(hex(raw bytes)); the
            # response artifact separately uses the raw-byte SHA256.
            or digest(raw["raw_model_text"].encode().hex()) != r.output_checksum
            or digest(raw.get("provider_config")) != r.config_checksum
        ):
            raise ValueError("pending_format_response_binding")
        if not r.simulated:
            http = observation_for(journal, r.request_id)
            if not http or http.get("origin") != "http_worker" or http.get("http_status") != 200:
                raise ValueError("pending_format_not_confirmed_http")
        try:
            decode(raw["raw_model_text"], raw["provider_config"])
        except (ValidationError, ValueError):
            pass
        else:
            raise ValueError("pending_format_valid_response_requires_local_recovery")
        values.append(dict(request_id=r.request_id, record=digest(r), response=digest(raw)))
    return dict(
        schema_version="confirmed-review-format-failure/1.0.0",
        last_request_id=records[-1].request_id,
        attempts=values,
        original_answer_not_accepted=True,
    )


def read_failed_review_attempt(
    *, documents, state, config, acknowledge_request=None, inspect_format_failures=False
):
    """Partition a stopped, unaccepted successor through its exact Review intent.

    Only an explicitly identified failed request may be reassessed in a NEW run.
    Confirmed responses/results must be locally recovered, never resent here.
    """
    from agents.expert_agent.models import ReviewFrame
    from packages.llm_provider.http_recovery import observation_for
    from packages.planner_control.models import Reference

    from .review_batches import check_plan, read_plan

    control = documents.sessions.read(documents.run_id)
    if control is None or control.phase != "stopped" or control.pending or control.cancelled:
        raise ValueError("pending_attempt_not_terminal")
    documents.validate_scope(state.material.scope)
    intent = documents.read("live-stage-intent:review")
    if not intent or len(intent) != 1:
        raise ValueError("pending_attempt_intent_missing")
    stage = intent[0]
    event = validate(Reference, stage["event"])
    prefix = "review:" + event.code
    original = documents.read(prefix + ":intent")
    command = documents.read(prefix + ":command")
    from .acceptance_models import DomainState

    if not original or len(original) != 1:
        raise ValueError("pending_attempt_parent_missing")
    frame = validate(ReviewFrame, original[0]["frame"])
    if (
        stage["role"] != "review"
        or stage["config_checksum"] != digest(config)
        or state.config_checksum != digest(config)
        or validate(DomainState, original[0]["state"]) != state
        or frame.event != event
        or frame.scope != state.material.scope
        or not any(
            a.id == stage["proposed_action_id"]
            and a.kind == "request_review"
            and a.branch == "research_review"
            for a in frame.actions.actions
        )
        or command
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
        raise ValueError("pending_attempt_owner_binding")
    key = event.code + ":A"
    saved = documents.read("review-batches:" + key + ":plan")
    if not saved:
        raise ValueError("pending_attempt_plan_missing")
    plan = check_plan(state, read_plan(saved[0]))
    if plan.key != key or plan.policy != config.review_batching or plan.round != "A":
        raise ValueError("pending_attempt_plan_binding")
    journal = RuntimeAgentJournal(documents)
    events = journal.read()
    records = [read_request_record(e["record"]) for e in events if e.get("phase") == "completed"]
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    if len(reserved) != len(set(reserved)) or sorted(reserved) != sorted(
        r.request_id for r in records
    ):
        raise ValueError("pending_attempt_unknown_reservation")
    grant = documents.read("research-live-grant")
    if any(not r.simulated for r in records):
        g = validate(ResearchLiveGrant, grant[0]) if grant else None
        s = state.material.scope
        if (
            g is None
            or not journal.durable
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
                digest(config),
                documents.root.checksum,
            )
            or "review" not in g.roles
        ):
            raise ValueError("pending_attempt_original_grant")
        if any(r.deadline > g.deadline for r in records):
            raise ValueError("pending_attempt_request_deadline")
    selected, quarantine, mapped, acknowledged = [], [], set(), False
    for batch in plan.batches:
        slot = "review-batches:" + key + ":" + batch.id
        payload = digest(batch_payload(state, plan, batch))
        matches = [r for r in records if r.input_checksum == payload]
        claim = documents.read(slot + ":claim")
        if documents.read(slot + ":result") or any(r.error is None for r in matches):
            raise ValueError("pending_attempt_confirmed_response_requires_local_recovery")
        if not claim and not matches:
            selected.extend(batch.assessment_ids)
            continue
        if claim != (dict(plan_checksum=digest(plan), input_checksum=payload),):
            raise ValueError("pending_attempt_claim_binding")
        # A confirmed rejected reply quarantines only its own batch. Validate
        # the original bytes even when the caller is NOT asking to reassess it;
        # inspection is not authorization and must not block unrelated unstarted work.
        format_proof = known_format_failure(documents, journal, slot, matches)
        for r in matches:
            attempt = documents.read(slot + ":attempt:" + r.request_id)
            if (
                r.scope != state.material.scope
                or r.role != "review"
                or r.config_checksum != digest(config)
                or r.prompt_checksum != state.prompt_checksum
                or not attempt
                or attempt[0]["record_checksum"] != digest(r)
                or attempt[0]["error"] != r.error
                or (attempt[0]["raw_model_text"] and not (format_proof and r.error == "schema"))
            ):
                raise ValueError("pending_attempt_request_binding")
            mapped.add(r.request_id)
        allow = False
        if len(matches) == 1 and matches[0].request_id == acknowledge_request:
            r = matches[0]
            http = observation_for(journal, r.request_id)
            rejected = (
                r.error == "rate_limit"
                and http
                and http.get("origin") == "http_worker"
                and http.get("operation") == "chat_completions"
                and http.get("http_status") == 429
                and http.get("category") in {"rate_limit", "unclassified_429"}
            )
            allow = bool(
                r.api_requests == 1
                and r.retry_index == 0
                and (r.error in {"timeout", "transport"} or rejected)
            )
        if allow:
            selected.extend(batch.assessment_ids)
            acknowledged = True
        quarantine.append(
            dict(
                run_id=documents.run_id,
                batch_id=batch.id,
                assessment_ids=list(batch.assessment_ids),
                request_ids=[r.request_id for r in matches],
                new_reassessment_authorized=allow,
                old_response_state_unchanged=True,
                **(dict(known_format_failure=format_proof) if inspect_format_failures else {}),
            )
        )
    if mapped != set(reserved) or (acknowledge_request and not acknowledged):
        raise ValueError("pending_attempt_unmapped_or_ack_mismatch")
    return dict(
        schema_version="pending-review-attempt-inventory/1.0.0",
        run_id=documents.run_id,
        selected_ids=sorted(selected),
        quarantined_batches=quarantine,
        plan_checksum=digest(plan),
        state_checksum=digest(state),
        journal_checksum=digest(events),
        control_checksum=digest(control),
        intent_checksum=digest(intent),
        grant_checksum=digest(grant),
        acknowledge_request=acknowledge_request,
        semantics="new_attempt_not_original_replay",
    )


def read_pending_review(
    *,
    documents,
    state,
    acceptance_key,
    acknowledge_unknown_request=None,
    allow_empty=False,
    inspect_format_failures=False,
    inspect_truncated_failures=False,
    allow_legacy_partial=False,
    inspect_empty_truncated_failures=False,
):
    """Read inventory, never permission; an ancestor may have no unstarted work.

    Caller must supply the freshly reconstructed original owner state. Empty
    inventory does not waive any source, receipt, request or acknowledgment gate.
    """
    if inspect_empty_truncated_failures and not inspect_truncated_failures:
        raise ValueError("pending_empty_truncated_explicit_inspection_required")
    control = documents.sessions.read(documents.run_id)
    if (
        control is None
        or control.phase != "stopped"
        or control.pending is not None
        or control.cancelled
        or state.material.scope.run_id != documents.run_id
    ):
        raise ValueError("pending_review_original_not_stopped")
    documents.validate_scope(state.material.scope)
    submitted = documents.read("submission:" + acceptance_key)
    if not submitted:
        raise ValueError("pending_review_submission_missing")
    sub = read_submission(submitted[0]["submission"])
    if not isinstance(sub, PartialReasonedSubmission) and not (
        allow_legacy_partial and type(sub) is PartialBatchedSubmission
    ):
        raise ValueError("pending_review_requires_partial_acceptance")
    acceptance = ResearchAcceptance(
        owner=SimpleNamespace(current=lambda: state), documents=documents, requests=None
    )
    report = read_research_report(acceptance, key=acceptance_key, parent=state)
    if (
        report["receipt"].get("status") != "accepted"
        or report["receipt"]["submission_checksum"] != digest(sub)
        or report["records"]["candidate"] != sub.candidate.model_dump(mode="json")
        or not documents.consumed(state.material.scope.graph)
    ):
        raise ValueError("pending_review_acceptance_binding")
    journal = RuntimeAgentJournal(documents)
    events = journal.read()
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    records = tuple(
        read_request_record(e["record"]) for e in events if e.get("phase") == "completed"
    )
    if (
        len(reserved) != len(set(reserved))
        or len(records) != len({r.request_id for r in records})
        or set(reserved) != {r.request_id for r in records}
    ):
        raise ValueError("pending_review_request_inventory")
    checked = validate_batch_receipts(
        state, sub, documents, SimpleNamespace(records=lambda: records, journal=journal)
    )
    if any(not r.simulated for r in records):
        grant = documents.read("research-live-grant")
        g = validate(ResearchLiveGrant, grant[0]) if grant else None
        s = state.material.scope
        if (
            g is None
            or not journal.durable
            or report["receipt"].get("permission_checksum") != digest(g)
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
                state.config_checksum,
                documents.root.checksum,
            )
            or "review" not in g.roles
            or any(r.deadline > g.deadline for r in records)
        ):
            raise ValueError("pending_review_original_permission")
    selected, quarantined, all_mapped = [], [], {r.request_id for r in checked}
    acknowledged = False
    for batch in sub.plan.batches:
        if batch.id not in sub.pending_batch_ids:
            continue
        slot = "review-batches:" + sub.plan.key + ":" + batch.id
        payload = digest(batch_payload(state, sub.plan, batch))
        claim = documents.read(slot + ":claim")
        matches = [r for r in records if r.input_checksum == payload]
        if any(
            r.scope != state.material.scope
            or r.role != "review"
            or r.config_checksum != state.config_checksum
            or r.prompt_checksum != state.prompt_checksum
            for r in matches
        ):
            raise ValueError("pending_review_request_binding")
        all_mapped.update(r.request_id for r in matches)
        if not claim and not matches:
            selected.extend(batch.assessment_ids)
            continue
        if claim != (dict(plan_checksum=digest(sub.plan), input_checksum=payload),):
            raise ValueError("pending_review_claim_binding")
        # Any sent/claimed batch is excluded by default, even if a local error is known.
        # A complete response with missing batch result requires local recovery instead.
        if any(r.error is None for r in matches):
            raise ValueError("pending_review_confirmed_response_requires_local_recovery")
        ids = [r.request_id for r in matches]
        can_reassess = (
            len(matches) == 1
            and matches[0].request_id == acknowledge_unknown_request
            and matches[0].error in {"timeout", "transport"}
            and matches[0].retry_index == 0
            and matches[0].api_requests == 1
        )
        if can_reassess:
            selected.extend(batch.assessment_ids)
            acknowledged = True
        quarantined.append(
            dict(
                batch_id=batch.id,
                assessment_ids=list(batch.assessment_ids),
                request_ids=ids,
                new_reassessment_authorized=can_reassess,
                old_response_state_unchanged=True,
                **(
                    dict(
                        known_format_failure=known_format_failure(documents, journal, slot, matches)
                    )
                    if inspect_format_failures
                    else {}
                ),
                **(
                    dict(
                        known_truncated_failure=known_truncated_failure(
                            documents,
                            journal,
                            slot,
                            matches,
                            **(
                                dict(allow_empty_output=True)
                                if inspect_empty_truncated_failures
                                else {}
                            ),
                        )
                    )
                    if inspect_truncated_failures
                    else {}
                ),
            )
        )
    if set(r.request_id for r in records) != all_mapped:
        raise ValueError("pending_review_unmapped_request")
    if acknowledge_unknown_request and not acknowledged:
        raise ValueError("pending_review_unknown_ack_mismatch")
    if not selected and not allow_empty:
        raise ValueError("pending_review_no_safe_unstarted_work")
    return dict(
        schema_version=(
            "pending-review-inventory/1.3.0"
            if inspect_empty_truncated_failures
            else "pending-review-inventory/1.2.0"
            if inspect_truncated_failures or allow_legacy_partial
            else "pending-review-inventory/1.0.0"
        ),
        run_id=documents.run_id,
        acceptance_key=acceptance_key,
        accepted_checksum=digest(acceptance.read(key=acceptance_key)),
        state_checksum=digest(state),
        journal_checksum=digest(events),
        control_checksum=digest(control),
        plan_checksum=digest(sub.plan),
        accepted_ids=sorted(i.assessment_id for i in sub.candidate.items),
        selected_ids=sorted(selected),
        quarantined_batches=quarantined,
        acknowledge_unknown_request=acknowledge_unknown_request,
        semantics="new_selected_review_only_not_whole_case_rescore",
    )


def read_local_unsent_review(*, documents, state, config):
    """Prove a fully unaccepted Review failed locally before any HTTP dispatch.

    This is an immutable failure inventory, not permission to retry an old key.
    A new bounded successor must independently bind this proof and current inputs.
    """
    value = read_failed_review_attempt(documents=documents, state=state, config=config)
    events = RuntimeAgentJournal(documents).read()
    records = [read_request_record(e["record"]) for e in events if e.get("phase") == "completed"]
    if not records or any(
        r.error != "configuration"
        or r.api_requests != 0
        or r.retry_index != 0
        or r.output_checksum is not None
        or r.remote_request_checksum is not None
        for r in records
    ):
        raise ValueError("local_review_dispatch_or_response_not_excluded")
    for r in records:
        confirmations = [
            e
            for e in events
            if e.get("phase") == "response_confirmation" and e.get("request_id") == r.request_id
        ]
        if (
            not r.simulated
            and confirmations
            != [dict(phase="response_confirmation", request_id=r.request_id, confirmed=False)]
        ) or any(
            e.get("schema_version") == "agent-http-observation/1.0.0"
            and e.get("request_id") == r.request_id
            for e in events
        ):
            raise ValueError("local_review_remote_confirmation_not_excluded")
    ids = list(value["selected_ids"])
    for batch in value["quarantined_batches"]:
        if batch["new_reassessment_authorized"] or not batch["request_ids"]:
            raise ValueError("local_review_batch_binding")
        ids.extend(batch["assessment_ids"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("local_review_target_partition")
    return dict(
        schema_version="local-unsent-review-proof/1.0.0",
        inventory=value,
        selected_ids=sorted(ids),
        request_ids=[r.request_id for r in records],
        charged_attempts=len(records),
        api_requests=0,
        unknown_requests=0,
        journal_checksum=digest(events),
        original_failure_unchanged=True,
    )
