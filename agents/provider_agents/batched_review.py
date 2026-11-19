"""Finite batch execution on the original Provider, request journal and documents."""

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_batches import (
    ReasonedBatchResult,
    ReviewBatchResponse,
    ReviewBatchResponseV11,
    ReviewBatchResponseV12,
    ReviewBatchResponseV13,
    ReviewBatchResponseV14,
    ReviewBatchResponseV15,
    ReviewBatchResponseV131,
    ReviewBatchResponseV132,
    ReviewBatchResponseV151,
    ReviewBatchResponseV152,
    ReviewBatchResult,
    aggregate,
    batch_payload,
    check_batch,
    make_plan,
    proposal_type,
    read_batch_result,
    read_response,
    response_candidate,
    response_key,
)
from packages.grounding.models import digest, validate
from packages.llm_provider.provider import ProviderError
from packages.object_store import checksum


def run_batches(agent, *, state, documents, key, round_id, before_request=None):
    state = validate(DomainState, state)
    config = agent.provider.config
    policy = getattr(config, "review_batching", None)
    if getattr(config, "direction_policy", None) != getattr(state.policy, "direction_policy", None):
        raise ProviderError("configuration", fields=("direction_policy",))
    if agent.role != "review" or policy is None:
        raise ProviderError("configuration")
    plan = make_plan(state, policy, key=key, round_id=round_id)
    prefix = "review-batches:" + key
    from packages.evidence_details.review_continuation import require_reader, reused_ids

    reuse = (
        require_reader(agent.materials)
        if getattr(agent.materials, "confirmed_review_reader", None) is not None
        else None
    )
    origin = reuse.current(state) if reuse else None
    reused = reused_ids(origin) if origin else set()
    if any(
        set(b.assessment_ids) & reused and not set(b.assessment_ids) <= reused for b in plan.batches
    ):
        from packages.evidence_details.accepted_review_reuse import AcceptedReviewOrigin
        from packages.evidence_details.review_batches import partition_plan

        if not isinstance(origin, AcceptedReviewOrigin):
            raise ValueError("review_continuation_partition")
        plan = partition_plan(plan, reused, digest(origin))

    def current():
        from agents.expert_agent.service import ReviewCurrentWindow

        agent.budget.check_control()
        if agent.budget.clock() >= agent.budget.deadline:
            raise ProviderError("deadline")
        if type(before_request) is ReviewCurrentWindow:
            domain = before_request.current(agent, documents, state)
        else:
            if before_request:
                before_request()
            domain = agent.materials.current()
        if validate(DomainState, domain) != state:
            raise ProviderError("scope", fields=("batch_parent",))
        if reuse and reuse.current(state) != origin:
            raise ProviderError("scope", fields=("confirmed_review_origin",))

    # Historic data has a separate read port; executing a plan always needs current authority.
    current()
    with documents.session():
        documents.write(prefix + ":plan", (plan,))
        if origin:
            documents.write(prefix + ":historical", (origin,))
    results = []
    for batch in plan.batches:
        if set(batch.assessment_ids) <= reused:
            continue
        current()
        slot = prefix + ":" + batch.id
        payload = batch_payload(state, plan, batch)
        saved = documents.read(slot + ":result")
        if saved:
            result = read_batch_result(saved[0])
            check_batch(state, plan, batch, result.candidate)
            results.append(result)
            continue
        raw_key = response_key(plan, batch)
        raw = documents.read(raw_key)
        if raw is None:
            # A committed attempt may have lost its acknowledgement before the
            # canonical response alias was written. Recover the original bytes.
            matches = []
            for record in agent.budget.records():
                if (
                    record.scope == plan.scope
                    and record.role == "review"
                    and record.input_checksum == digest(payload)
                    and record.error is None
                ):
                    attempt = documents.read(slot + ":attempt:" + record.request_id)
                    if attempt and attempt[0]["record_checksum"] == digest(record):
                        matches.append(attempt)
            if len(matches) > 1:
                raise ValueError("ambiguous_batch_response")
            if matches:
                with documents.session():
                    documents.write(raw_key, matches[0])
                raw = matches[0]
        if raw is None:
            with documents.session():
                current()
                agent.budget.before_reserve(digest(config))
                if documents.read(slot + ":claim"):
                    raise ProviderError("batch_response_unknown_no_repeat")
                documents.write(
                    slot + ":claim",
                    (dict(plan_checksum=digest(plan), input_checksum=digest(payload)),),
                )

            def capture(record, reply, *, slot=slot, raw_key=raw_key):
                # Durable response evidence precedes local result/aggregate confirmation.
                # A retry's error response is separate from the final successful response.
                from packages.llm_provider.config import (
                    CurlNullableProviderConfig,
                    CurlSourceProviderConfig,
                    DirectionalProviderConfig,
                    IntegratedReviewProviderConfig,
                    NullableReviewProviderConfig,
                    ReviewShapeProviderConfig,
                    RoutedNullableProviderConfig,
                    RoutedSourceProviderConfig,
                    SourceQuoteProviderConfig,
                )
                from packages.llm_provider.review_response import decode

                adapted = isinstance(config, ReviewShapeProviderConfig)
                cls = ReviewBatchResponseV11 if adapted else ReviewBatchResponse
                if isinstance(config, IntegratedReviewProviderConfig):
                    cls = ReviewBatchResponseV12
                if isinstance(config, SourceQuoteProviderConfig):
                    cls = ReviewBatchResponseV13
                if isinstance(config, DirectionalProviderConfig):
                    cls = ReviewBatchResponseV14
                if isinstance(config, NullableReviewProviderConfig):
                    cls = ReviewBatchResponseV15
                if isinstance(config, RoutedSourceProviderConfig):
                    cls = ReviewBatchResponseV131
                if isinstance(config, RoutedNullableProviderConfig):
                    cls = ReviewBatchResponseV151
                if isinstance(config, CurlSourceProviderConfig):
                    cls = ReviewBatchResponseV132
                if isinstance(config, CurlNullableProviderConfig):
                    cls = ReviewBatchResponseV152
                proof = None
                if adapted and record.error is None:
                    _, proof = decode(reply.content, config)
                value = cls(
                    request_id=record.request_id,
                    record_checksum=digest(record),
                    raw_model_text=reply.content.decode("utf-8", errors="replace"),
                    raw_checksum=checksum(reply.content),
                    error=record.error,
                    finish_reason=reply.finish_reason,
                    untrusted_model_output=True,
                    accepted_medical_evidence=False,
                    **(dict(provider_config=config, alignment=proof) if adapted else {}),
                )
                with documents.session():
                    documents.write(slot + ":attempt:" + record.request_id, (value,))
                    if record.error is None:
                        documents.write(raw_key, (value,))

            def admit(candidate, *, batch=batch):
                try:
                    check_batch(state, plan, batch, candidate)
                except ValueError as error:
                    from packages.evidence_details.review_tasks import project_review_tasks
                    from packages.llm_provider.review_citations import failure_paths

                    fields = (
                        failure_paths(project_review_tasks(state), candidate)
                        if str(error) in {"review_citation_scope", "review_duplicate_citation"}
                        else ()
                    )
                    if not fields and str(error) in {
                        "review_ungradable_status",
                        "review_missing_status",
                        "review_acquisition_mismatch",
                        "review_applicability",
                        "review_observation_method_unit_eye",
                        "review_direction_protocol_binding",
                    }:
                        fields = (str(error),)
                    raise ProviderError("scope", fields=fields or ("batch_candidate",)) from error

            agent.provider.generate_review_batch(
                input_payload=payload,
                scope=plan.scope,
                budget=agent.budget,
                deadline=agent.budget.deadline,
                admit=admit,
                before_request=current,
                capture=capture,
            )
            raw = documents.read(raw_key)
        if raw is None:
            raise ProviderError("batch_response_unknown_no_repeat")
        response_object = read_response(raw[0])
        response = response_object.model_dump(mode="json")
        records = [r for r in agent.budget.records() if r.request_id == response["request_id"]]
        if len(records) != 1 or digest(records[0]) != response["record_checksum"]:
            raise ValueError("batch_original_record_missing")
        record = records[0]
        if record.error or record.input_checksum != digest(payload):
            raise ProviderError(record.error or "scope")
        candidate = check_batch(
            state,
            plan,
            batch,
            response_candidate(
                response_object,
                config_checksum=plan.config_checksum,
                prompt_checksum=plan.prompt_checksum,
            ),
        )
        if checksum(response["raw_model_text"].encode()) != response[
            "raw_checksum"
        ] or record.output_checksum != digest(candidate):
            raise ValueError("batch_response_binding")
        current()
        result = (
            ReasonedBatchResult if getattr(config, "direction_policy", None) else ReviewBatchResult
        )(
            plan_checksum=digest(plan),
            batch_id=batch.id,
            request_id=record.request_id,
            input_checksum=record.input_checksum,
            output_schema_checksum=digest(proposal_type(config).model_json_schema()),
            response=documents.reference(
                raw_key, 0, response, role="response", version=response["schema_version"]
            ),
            candidate=candidate,
        )
        with documents.session():
            documents.write(slot + ":result", (result,))
        results.append(result)
    current()
    if origin:
        from packages.evidence_details.review_continuation import aggregate_continued

        submission = aggregate_continued(state, plan, results, origin)
    else:
        submission = aggregate(state, plan, results)
    with documents.session():
        documents.write(prefix + ":aggregate", (submission,))
    return submission


def read_progress(documents, key):
    """Exact historical read, including stopped/expired runs; never generates requests."""
    from packages.evidence_details.review_batches import read_plan

    prefix = "review-batches:" + key
    saved = documents.read(prefix + ":plan")
    if saved is None:
        return None
    plan = read_plan(saved[0])
    return dict(
        plan=plan.model_dump(mode="json"),
        batches=[
            dict(
                batch_id=b.id,
                claim=documents.read(prefix + ":" + b.id + ":claim"),
                response=documents.read(response_key(plan, b)),
                result=documents.read(prefix + ":" + b.id + ":result"),
            )
            for b in plan.batches
        ],
        aggregate=documents.read(prefix + ":aggregate"),
    )
