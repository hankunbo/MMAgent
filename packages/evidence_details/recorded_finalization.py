"""Save a complete recorded Review after deadline, including verified format correction.

No dispatch or Control transition is exposed. Old grants/results stay immutable.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import model_validator

from packages.grounding.models import Checksum, digest, validate
from packages.llm_provider.models import read_request_record
from packages.object_store import checksum

from .finalization import FinalizationGrant
from .live_permission import ResearchLiveGrant, ResearchLivePermission, validate_live_config
from .review_batches import (
    aggregate,
    read_batch_result,
    read_plan,
    read_response,
    validate_batch_receipts,
)

GRANT_KEY = "recorded-review-finalization:grant:1"
SUBMISSION_KEY = "recorded-review-finalization:submission:1"
RESULT_KEY = "multimodal-recorded-finalization-result"
REPORT_KEY = "multimodal-recorded-finalization-report"
ACCEPTANCE_KEY = "recorded-review-finalization:1"
PARTITION_GRANT_KEY = "recorded-partition-finalization:grant:1"
PARTITION_SUBMISSION_KEY = "recorded-partition-finalization:submission:1"
PARTITION_ACCEPTANCE_KEY = "recorded-partition-finalization:1"
PARTITION_POLICY = "recorded-partition-local-window/1.1.0"


class RecordedReviewGrant(FinalizationGrant):
    schema_version: Literal["recorded-review-finalization/1.0.0"] = (
        "recorded-review-finalization/1.0.0"
    )
    purpose: Literal["save_complete_recorded_review_only"] = "save_complete_recorded_review_only"
    rejected_response_checksums: tuple[Checksum, ...]


class RecordedPartitionGrant(RecordedReviewGrant):
    """New zero-wire authorization; old 15-minute grants remain immutable."""

    schema_version: Literal["recorded-review-finalization/1.1.0"] = (
        "recorded-review-finalization/1.1.0"
    )
    previous_local_grant_checksum: Checksum | None
    previous_local_receipt_checksum: Checksum | None

    @model_validator(mode="after")
    def bounded(self):
        if (
            self.issued_at.tzinfo is None
            or self.deadline.tzinfo is None
            or not timedelta(0) < self.deadline - self.issued_at <= timedelta(minutes=45)
            or len(set(self.request_checksums)) != len(self.request_checksums)
        ):
            raise ValueError("finalization_bound")
        return self


class RecordedOriginCompatibilityGrant(RecordedPartitionGrant):
    """Only an owner-verified v1.3 origin / ordinary-plan aggregation failure."""

    schema_version: Literal["recorded-review-finalization/1.2.0"] = (
        "recorded-review-finalization/1.2.0"
    )
    origin_compatibility_checksum: Checksum


class RecordedAllReusedGrant(RecordedPartitionGrant):
    """Local closure of an original all-reused failure, not a new model Review."""

    schema_version: Literal["recorded-review-finalization/1.3.0"] = (
        "recorded-review-finalization/1.3.0"
    )
    request_checksums: tuple[()] = ()
    all_reused_origin_checksum: Checksum


class RecordedLineageCapacityGrant(RecordedPartitionGrant):
    """Confirmed aggregate rejected only by the legacy provenance depth bound."""

    schema_version: Literal["recorded-review-finalization/1.4.0"] = (
        "recorded-review-finalization/1.4.0"
    )
    lineage_capacity_checksum: Checksum


def lineage_capacity_proof(documents):
    """Bind the saved technical failure and complete aggregate; not acceptance.

    Current inputs, sources, ancestor receipts and every request are independently
    reconstructed by the original owner before a local grant can be created.
    """
    from .accepted_review_reuse import (
        MAX_ACCEPTED_REVIEW_DEPTH,
        PROVENANCE_POLICY,
        provenance_batches,
    )
    from .review_batches import read_submission

    value = documents.read("multimodal-case-result")
    if not value or len(value) != 1 or not value[0].get("phases"):
        return None
    result, phase = value[0], value[0]["phases"][-1]
    if (
        result.get("complete") is not False
        or phase.get("run_id") != documents.run_id
        or phase.get("target") is not None
        or phase.get("accepted") is not False
        or phase.get("error") != "accepted_review_reuse_depth"
        or documents.read("dynamic-phase-result") != (phase,)
    ):
        return None
    intent = documents.read("live-stage-intent:review")
    if not intent or len(intent) != 1:
        return None
    prefix = "review-batches:" + intent[0]["event"]["code"] + ":A"
    plans = documents.read(prefix + ":plan")
    origins = documents.read(prefix + ":historical")
    aggregates = documents.read(prefix + ":aggregate")
    if any(not v or len(v) != 1 for v in (plans, origins, aggregates)):
        return None
    plan, sub = read_plan(plans[0]), read_submission(aggregates[0])
    if (
        sub.plan != plan
        or sub.scope.run_id != documents.run_id
        or not hasattr(sub, "historical")
        or (sub.historical.model_dump(mode="json"),) != origins
        or not sub.batches
    ):
        return None
    # Validate the new finite traversal, then prove this actually exceeded the
    # old bound. A shallow unrelated failure cannot acquire the new permission.
    provenance_batches(sub)
    depth, node = 0, sub
    while hasattr(node, "historical"):
        depth += 1
        if depth > MAX_ACCEPTED_REVIEW_DEPTH:
            return None
        node = node.historical.submission
    if depth <= 4:
        return None
    batches = []
    for batch in sub.batches:
        slot = prefix + ":" + batch.batch_id
        saved, response = documents.read(slot + ":result"), documents.read(slot + ":response")
        if (
            not saved
            or len(saved) != 1
            or read_batch_result(saved[0]) != batch
            or not response
            or len(response) != 1
        ):
            return None
        parsed = read_response(response[0])
        if parsed.error is not None or parsed.request_id != batch.request_id:
            return None
        batches.append(dict(result=saved, response=response))
    return digest(
        dict(
            policy=PROVENANCE_POLICY,
            failure=result,
            intent=intent,
            plan=plan,
            historical=origins,
            aggregate=sub,
            batches=batches,
        )
    )


def recorded_grant_type(raw):
    """Shared strict version dispatch for live permission and historical archive readers."""
    cls = {
        "recorded-review-finalization/1.0.0": RecordedReviewGrant,
        "recorded-review-finalization/1.1.0": RecordedPartitionGrant,
        "recorded-review-finalization/1.2.0": RecordedOriginCompatibilityGrant,
        "recorded-review-finalization/1.3.0": RecordedAllReusedGrant,
        "recorded-review-finalization/1.4.0": RecordedLineageCapacityGrant,
    }.get(raw.get("schema_version"))
    if cls is None:
        raise ValueError("recorded_finalization_grant_version")
    return cls


def all_reused_failure_proof(documents):
    """Identify the exact old technical failure; current owner still authorizes reuse."""
    value = documents.read("multimodal-case-result")
    if not value or not value[0].get("phases"):
        return None
    result, phase = value[0], value[0]["phases"][-1]
    if (
        result.get("complete") is not False
        or phase.get("run_id") != documents.run_id
        or phase.get("target") is not None
        or phase.get("accepted") is not False
        or phase.get("error") != "batch_incomplete_or_reused_request"
        or documents.read("dynamic-phase-result") != (phase,)
    ):
        return None
    intent = documents.read("live-stage-intent:review")
    if not intent:
        return None
    prefix = "review-batches:" + intent[0]["event"]["code"] + ":A"
    plans, origins = documents.read(prefix + ":plan"), documents.read(prefix + ":historical")
    if not plans or not origins or len(origins) != 1:
        return None
    plan = read_plan(plans[0])
    origin = origins[0]
    if (
        not origin.get("assessment_ids")
        or set(origin["assessment_ids"]) != set(plan.assessment_ids)
        or documents.read(prefix + ":aggregate") is not None
        or any(
            documents.read(prefix + ":" + b.id + suffix)
            for b in plan.batches
            for suffix in (":claim", ":response", ":result")
        )
    ):
        return None
    return digest(dict(failure=result, intent=intent, plan=plan, historical=origin))


def origin_compatibility_proof(documents):
    """Recognize the exact saved failure shape, not permission based on error text.

    Reconstruction still revalidates the accepted ancestor, every batch/request,
    current sources and inputs before any new local acceptance is permitted.
    """
    from .accepted_review_reuse import AcceptedReviewOriginV13
    from .review_batches import ReviewBatchPlan

    value = documents.read("multimodal-case-result")
    if not value or not value[0].get("phases"):
        return None
    result, phase = value[0], value[0]["phases"][-1]
    if (
        result.get("complete") is not False
        or phase.get("run_id") != documents.run_id
        or phase.get("target") is not None
        or phase.get("accepted") is not False
        or not str(phase.get("error", "")).startswith(
            "2 validation errors for ReusedAcceptedSubmissionV9\nhistorical.schema_version"
        )
        or documents.read("dynamic-phase-result") != (phase,)
    ):
        return None
    intent = documents.read("live-stage-intent:review")
    if not intent:
        return None
    prefix = "review-batches:" + intent[0]["event"]["code"] + ":A"
    plan, origin = documents.read(prefix + ":plan"), documents.read(prefix + ":historical")
    if (
        not plan
        or not origin
        or plan[0].get("schema_version") != "review-batch-plan/1.0.0"
        or origin[0].get("schema_version") != "accepted-review-origin/1.3.0"
        or documents.read(prefix + ":aggregate") is not None
    ):
        return None
    validated_plan = validate(ReviewBatchPlan, plan[0])
    validated_origin = validate(AcceptedReviewOriginV13, origin[0])
    return digest(
        dict(
            failure=result,
            intent=intent,
            plan=validated_plan,
            origin=validated_origin,
            semantics="saved_batches_require_current_owner_reconstruction",
        )
    )


def closure_keys(documents):
    """Select by original plan/reuse records, never caller-chosen storage keys.

    Single-item batches do not require ReusePartitionPlan: their original v1
    plan already separates old and new items. Reconstruction below still needs
    the original accepted-review owner; the stored marker is not authorization.
    """
    from .review_batches import ReusePartitionPlan

    intent = documents.read("live-stage-intent:review")
    if intent:
        prefix = "review-batches:" + intent[0]["event"]["code"] + ":A"
        saved = documents.read(prefix + ":plan")
        if saved and (
            isinstance(read_plan(saved[0]), ReusePartitionPlan)
            or documents.read(prefix + ":historical") is not None
        ):
            return PARTITION_GRANT_KEY, PARTITION_SUBMISSION_KEY, PARTITION_ACCEPTANCE_KEY
    return GRANT_KEY, SUBMISSION_KEY, ACCEPTANCE_KEY


def prior_local_proof(documents):
    grant, receipt = documents.read(GRANT_KEY), documents.read("accepted:" + ACCEPTANCE_KEY)
    if receipt and receipt[-1].get("status") == "accepted":
        raise ValueError("recorded_finalization_prior_acceptance_must_be_reused")
    return dict(
        previous_local_grant_checksum=digest(grant[0]) if grant else None,
        previous_local_receipt_checksum=digest(receipt) if receipt else None,
    )


class RecordedBatchRequests:
    def __init__(self, journal):
        if not journal.durable:
            raise ValueError("recorded_finalization_durable_required")
        self.journal = journal

    def records(self):
        events = self.journal.read()
        reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        records = tuple(
            read_request_record(e["record"]) for e in events if e.get("phase") == "completed"
        )
        if (
            (not records and all_reused_failure_proof(self.journal.documents) is None)
            or len(set(reserved)) != len(reserved)
            or sorted(reserved) != sorted(r.request_id for r in records)
            or any(
                r.simulated
                or r.api_requests != 1
                or r.role != "review"
                or r.error not in (None, "schema")
                or r.retry_index not in (0, 1)
                for r in records
            )
        ):
            raise ValueError("recorded_finalization_unknown_or_unsupported")
        return records


def original_failure(documents):
    value = documents.read("multimodal-case-result")
    if not value:
        raise ValueError("recorded_finalization_original_missing")
    result = value[0]
    phases = result.get("phases", [])
    if (
        result.get("complete") is not False
        or not phases
        or phases[-1].get("run_id") != documents.run_id
        or phases[-1].get("target") is not None
        or phases[-1].get("accepted") is not False
        or (
            phases[-1].get("error") != "deadline"
            and origin_compatibility_proof(documents) is None
            and all_reused_failure_proof(documents) is None
            and lineage_capacity_proof(documents) is None
        )
        or documents.read("dynamic-phase-result") != (phases[-1],)
    ):
        raise ValueError("recorded_finalization_only_closed_deadline")
    return result


def reconstruct(state, documents, requests, *, owner=None):
    intent = documents.read("live-stage-intent:review")
    if not intent:
        raise ValueError("recorded_finalization_intent")
    key = intent[0]["event"]["code"] + ":A"
    saved = documents.read("review-batches:" + key + ":plan")
    if not saved:
        raise ValueError("recorded_finalization_plan_missing")
    plan = read_plan(saved[0])
    from .review_batches import ReusePartitionPlan

    partition = isinstance(plan, ReusePartitionPlan)
    historical = documents.read("review-batches:" + key + ":historical")
    reuse = partition or historical is not None
    origin = None
    reused = set()
    if reuse:
        from .accepted_review_reuse import AcceptedReviewOrigin
        from .review_continuation import require_reader, reused_ids

        origin = require_reader(owner).current(state)
        if not isinstance(origin, AcceptedReviewOrigin):
            raise ValueError("recorded_finalization_accepted_origin_required")
        if historical != (origin.model_dump(mode="json"),):
            raise ValueError("recorded_finalization_partition_origin_changed")
        reused = reused_ids(origin)
        if (
            not reused
            or not reused <= set(plan.assessment_ids)
            or any(
                set(b.assessment_ids) & reused and not set(b.assessment_ids) <= reused
                for b in plan.batches
            )
        ):
            raise ValueError("recorded_finalization_reuse_partition")
    results = []
    for batch in plan.batches:
        if reuse and set(batch.assessment_ids) <= reused:
            continue  # Original accepted owner above validates these, not fabricated local rows.
        raw = documents.read("review-batches:" + key + ":" + batch.id + ":result")
        if not raw:
            raise ValueError("recorded_finalization_complete_batches_required")
        results.append(read_batch_result(raw[0]))
    if reuse:
        from .review_continuation import aggregate_continued, check_continued

        submission = aggregate_continued(state, plan, tuple(results), origin)
        confirmed = check_continued(
            state, submission, owner=owner, documents=documents, requests=requests
        )
    else:
        submission = aggregate(state, plan, tuple(results))
        confirmed = validate_batch_receipts(state, submission, documents, requests)
    saved_aggregate = documents.read("review-batches:" + key + ":aggregate")
    # A partition may finish its last batch just before the final guard expires.
    # Rebuild a NEW finalization submission, never populate or rewrite the old aggregate.
    compatible_missing = saved_aggregate is None and (
        origin_compatibility_proof(documents) is not None
        or all_reused_failure_proof(documents) is not None
    )
    if (
        saved_aggregate is not None or not partition and not compatible_missing
    ) and saved_aggregate != (submission.model_dump(mode="json"),):
        raise ValueError("recorded_finalization_original_aggregate_required")
    records, rejected = verified_corrected_records(
        state, submission, documents, requests, confirmed
    )
    return submission, records, rejected


def verified_corrected_records(state, submission, documents, requests, confirmed):
    """Original confirmed format correction accounting, never replay authorization.

    Also used for an already accepted ordinary Review: failed attempts belong
    to its request ledger, but only the separate successful candidate is evidence.
    """
    from .pending_review import known_format_failure

    key = submission.plan.key
    records = tuple(requests.records())
    confirmed_ids = {r.request_id for r in confirmed}
    if tuple(r for r in records if r.request_id in confirmed_ids) != tuple(confirmed):
        raise ValueError("recorded_finalization_confirmed_order")
    rejected = []
    events = requests.journal.read()
    for record in records:
        if record in confirmed:
            continue
        following = [r for r in confirmed if r.invocation_id == record.invocation_id]
        corrections = [
            e
            for e in events
            if e.get("schema_version") == "review-schema-correction/1.0.0"
            and e.get("failed_request_id") == record.request_id
        ]
        if (
            record.error != "schema"
            or record.retry_index != 0
            or len(following) != 1
            or following[0].retry_index != 1
            or len(corrections) != 1
            or corrections[0]["failed_record_checksum"] != digest(record)
        ):
            raise ValueError("recorded_finalization_unaccounted_request")
        batch = next(b for b in submission.batches if b.request_id == following[0].request_id)
        slot = "review-batches:" + key + ":" + batch.batch_id
        # This also binds original raw bytes to the failed output checksum and,
        # for real requests, the original HTTP 200 confirmation. No schema-only
        # label or caller-provided 'confirmed' boolean is enough.
        if known_format_failure(documents, requests.journal, slot, [record]) is None:
            raise ValueError("recorded_finalization_error_unconfirmed")
        raw = documents.read(
            "review-batches:" + key + ":" + batch.batch_id + ":attempt:" + record.request_id
        )
        if not raw:
            raise ValueError("recorded_finalization_original_error_missing")
        response = read_response(raw[0])
        if (
            response.error != "schema"
            or response.finish_reason != "stop"
            or response.request_id != record.request_id
            or response.record_checksum != digest(record)
            or checksum(response.raw_model_text.encode()) != response.raw_checksum
        ):
            raise ValueError("recorded_finalization_error_binding")
        from packages.llm_provider.review_response import decode

        if (
            not hasattr(response, "provider_config")
            or digest(response.provider_config) != state.config_checksum
        ):
            raise ValueError("recorded_finalization_error_config")
        try:
            decode(response.raw_model_text, response.provider_config)
        except (ValueError, TypeError):
            pass
        else:
            raise ValueError("recorded_finalization_error_not_reproduced")
        # The original malformed bytes stay rejected; only the independently returned
        # successful record, checked by validate_batch_receipts/verify_correction, is used.
        rejected.append(digest(response))
    return records, tuple(rejected)


class RecordedReviewPermission(ResearchLivePermission):
    def __init__(self, *, grant, config, documents, requests, clock=None, owner=None):
        raw = grant.model_dump(mode="json") if hasattr(grant, "model_dump") else grant
        cls = recorded_grant_type(raw)
        self.grant = validate(cls, grant)
        self.grant_key, self.submission_key, self.acceptance_key = closure_keys(documents)
        if isinstance(self.grant, RecordedPartitionGrant) != (
            self.grant_key == PARTITION_GRANT_KEY
        ):
            raise ValueError("recorded_finalization_plan_grant_version")
        self.config = validate_live_config(config)
        self.documents, self.budget = documents, requests
        self.owner = owner
        self.clock = clock or (lambda: datetime.now(UTC))
        if requests.journal.documents is not documents:
            raise ValueError("recorded_finalization_owner")

    @classmethod
    def prepare(cls, *, owner, documents, requests, config, clock=None):
        state = owner.current()
        original = original_failure(documents)
        control = documents.sessions.read(documents.run_id)
        if control is None or control.phase != "stopped" or control.pending is not None:
            raise ValueError("recorded_finalization_stopped_required")
        if documents.consumed(state.material.scope.graph):
            raise ValueError("recorded_finalization_parent_consumed")
        sub, records, rejected = reconstruct(state, documents, requests, owner=owner)
        grant_key, submission_key, _ = closure_keys(documents)
        partition = grant_key == PARTITION_GRANT_KEY
        compatibility = origin_compatibility_proof(documents)
        all_reused = all_reused_failure_proof(documents)
        lineage_capacity = lineage_capacity_proof(documents)
        grant_type = (
            RecordedLineageCapacityGrant
            if lineage_capacity is not None
            else RecordedAllReusedGrant
            if all_reused is not None
            else RecordedOriginCompatibilityGrant
            if compatibility is not None
            else RecordedPartitionGrant
            if partition
            else RecordedReviewGrant
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
            **(prior_local_proof(documents) if partition else {}),
            **(dict(origin_compatibility_checksum=compatibility) if compatibility else {}),
            **(dict(all_reused_origin_checksum=all_reused) if all_reused else {}),
            **(dict(lineage_capacity_checksum=lineage_capacity) if lineage_capacity else {}),
        )
        # Time spent reconstructing historical owners is preparation, not a
        # retroactive extension of the stopped execution or of an existing grant.
        now = (clock or (lambda: datetime.now(UTC)))()
        with documents.session():
            saved = documents.read(grant_key)
            grant = (
                validate(grant_type, saved[0])
                if saved
                else grant_type(
                    **expected,
                    issued_at=now,
                    deadline=now + timedelta(minutes=45 if partition else 15),
                )
            )
            if any(getattr(grant, k) != v for k, v in expected.items()):
                raise ValueError("recorded_finalization_key_content_conflict")
            permission = cls(
                grant=grant,
                config=config,
                documents=documents,
                requests=requests,
                clock=clock,
                owner=owner,
            )
            permission._check(state, role="review", persisted=False)
            documents.write(submission_key, (sub,))
            documents.write(grant_key, (grant,))
        return permission, sub

    def _check(self, state, *, role, record=None, persisted=True):
        if self.clock() >= self.grant.deadline:
            raise ValueError("recorded_finalization_deadline")
        self._check_bindings(state, role=role, record=record, persisted=persisted)
        if self.clock() >= self.grant.deadline:
            raise ValueError("recorded_finalization_deadline")

    def _check_bindings(self, state, *, role, record=None, persisted=True):
        """Current owner bindings; not permission to perform a new acceptance."""
        g, docs, scope = self.grant, self.documents, state.material.scope
        compatibility = origin_compatibility_proof(docs)
        all_reused = all_reused_failure_proof(docs)
        lineage_capacity = lineage_capacity_proof(docs)
        if (lineage_capacity is not None) != isinstance(g, RecordedLineageCapacityGrant) or (
            lineage_capacity is not None and g.lineage_capacity_checksum != lineage_capacity
        ):
            raise ValueError("recorded_finalization_lineage_capacity_binding")
        if (all_reused is not None) != isinstance(g, RecordedAllReusedGrant) or (
            all_reused is not None and g.all_reused_origin_checksum != all_reused
        ):
            raise ValueError("recorded_finalization_all_reused_binding")
        if (compatibility is not None) != isinstance(g, RecordedOriginCompatibilityGrant) or (
            compatibility is not None and g.origin_compatibility_checksum != compatibility
        ):
            raise ValueError("recorded_finalization_origin_compatibility_binding")
        if role != "review" or digest(state) != g.state_checksum:
            raise ValueError("recorded_finalization_scope")
        if persisted and (
            docs.read(self.grant_key) != (g.model_dump(mode="json"),)
            or digest(docs.read(self.submission_key)[0]) != g.aggregate_checksum
        ):
            raise ValueError("recorded_finalization_grant_changed")
        if isinstance(g, RecordedPartitionGrant) and any(
            getattr(g, name) != value for name, value in prior_local_proof(docs).items()
        ):
            raise ValueError("recorded_finalization_previous_local_attempt_changed")
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
            raise ValueError("recorded_finalization_origin_changed")
        sub, records, rejected = reconstruct(state, docs, self.budget, owner=self.owner)
        if (digest(sub), tuple(digest(r) for r in records), rejected) != (
            g.aggregate_checksum,
            g.request_checksums,
            g.rejected_response_checksums,
        ):
            raise ValueError("recorded_finalization_records_changed")
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
                raise ValueError("recorded_finalization_request_binding")
        if record is not None and (record not in records or record.error is not None):
            raise ValueError("recorded_finalization_request_not_accepted")
        docs.validate_scope(scope)

    def check(self, state, *, role=None, record=None):
        self._check(state, role=role, record=record)


def read_saved_acceptance(*, owner, documents, requests, config):
    """Prove an already committed local closure, never renew its grant or run.

    Expiry prevents new writes; it does not erase an accepted historical fact.
    Current inputs/sources still come from the original owner. This read-only
    proof is not a ResearchLivePermission and exposes no dispatch/write method.
    """
    from agents.expert_agent.reporting import read_research_report

    from .acceptance import ResearchAcceptance
    from .acceptance_models import read_receipt

    state = owner.current()
    grant_key, submission_key, key = closure_keys(documents)
    grant = documents.read(grant_key)
    bundle = documents.read("accepted:" + key)
    submitted = documents.read("submission:" + key)
    local = documents.read(submission_key)
    if (
        not grant
        or len(grant) != 1
        or not bundle
        or len(bundle) != 4
        or not submitted
        or len(submitted) != 1
        or not local
        or len(local) != 1
    ):
        raise ValueError("recorded_acceptance_proof_missing")
    permission = RecordedReviewPermission(
        grant=grant[0], config=config, documents=documents, requests=requests, owner=owner
    )
    permission._check_bindings(state, role="review")
    receipt = read_receipt(bundle[3])
    if (
        receipt.status != "accepted"
        or getattr(receipt, "permission_checksum", None) != digest(permission.grant)
        or receipt.submission_checksum != permission.grant.aggregate_checksum
        or submitted[0].get("submission") != local[0]
        or not documents.consumed(state.material.scope.graph)
    ):
        raise ValueError("recorded_acceptance_proof_binding")
    # Original reporting owner revalidates graph, records, scoring and source
    # quotations. Merely finding an accepted flag or a model response is not proof.
    report = read_research_report(
        ResearchAcceptance(owner=owner, documents=documents, requests=requests),
        key=key,
        parent=state,
    )
    if report["records"]["candidate"] != local[0]["candidate"]:
        raise ValueError("recorded_acceptance_proof_candidate")
    if owner.current() != state:
        raise ValueError("recorded_acceptance_proof_current_state_changed")
    return dict(
        schema_version="recorded-review-acceptance-proof/1.0.0",
        run_id=documents.run_id,
        acceptance_key=key,
        acceptance_checksum=digest(bundle),
        grant_checksum=digest(permission.grant),
        original_result_checksum=permission.grant.failed_result_checksum,
        state_checksum=digest(state),
        control_checksum=permission.grant.control_checksum,
        submission_checksum=permission.grant.aggregate_checksum,
        journal_checksum=digest(requests.journal.read()),
        semantics="saved_acceptance_not_new_dispatch_authorization",
    )
