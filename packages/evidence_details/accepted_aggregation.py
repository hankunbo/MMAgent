"""Current-owner reaggregation of accepted local judgments, with zero new requests.

Historical partial/complete/local-interpretation receipts remain immutable. The
original scoring owner is used; this is not a new LLM integrated assessment.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value, digest, validate
from packages.llm_provider.models import ReasonedReviewProposal, ReviewProposal, read_request_record

from .acceptance_models import AcceptanceReceiptV2, AcceptedInterpretation, AcceptedRecordsV25
from .accepted_review_reuse import compatible_local_ids
from .accepted_source_aliases import AliasProjection, project_aliases
from .live_permission import ResearchLiveGrant, ResearchLivePermission
from .partial_finalization import (
    CITATION_ACCEPTANCE_KEY,
    DEADLINE_ACCEPTANCE_KEY,
    accepted_citation_permission,
    accepted_deadline_permission,
)
from .recorded_explanation import ExplanationGrant, ExplanationSubmission, check_submission
from .review_batches import (
    BatchedSubmission,
    PartialBatchedSubmission,
    PartialReasonedSubmission,
    ReasonedSubmission,
    make_plan,
    read_submission,
    request_for_item,
    validate_batch_receipts,
)

POLICY = "accepted-local-review-aggregation/1.0.0"
GRANT_KEY = "accepted-aggregation:grant:1"
ACCEPTANCE_KEY = "accepted-aggregation:acceptance:1"


class AcceptedOrigin(Value):
    run_id: Token
    acceptance_key: Token
    acceptance_checksum: Checksum
    submission_checksum: Checksum
    state_checksum: Checksum
    control_checksum: Checksum
    journal_checksum: Checksum
    permission_checksum: Checksum | None
    item_requests: dict[Token, Token] = Field(min_length=1, max_length=64)
    unused_request_ids: tuple[Token, ...]


class AggregatedSubmission(ReasonedSubmission):
    schema_version: Literal["research-batched-submission/12.0.0"] = (
        "research-batched-submission/12.0.0"
    )
    batches: tuple = Field(default=(), max_length=0)
    origins: tuple[AcceptedOrigin, ...] = Field(min_length=1, max_length=8)
    aggregation_policy: Literal["accepted-local-review-aggregation/1.0.0"] = POLICY
    new_model_requests: Literal[0] = 0
    plan_semantics: Literal["validation_only_not_dispatched"] = "validation_only_not_dispatched"
    review_semantics: Literal["unchanged_local_judgments_not_new_joint_review"] = (
        "unchanged_local_judgments_not_new_joint_review"
    )


class AggregatedRecords(AcceptedRecordsV25):
    schema_version: Literal["case-observation-interpretation/2.14.0"] = (
        "case-observation-interpretation/2.14.0"
    )
    batch_results: tuple = Field(default=(), max_length=0)
    origins: tuple[AcceptedOrigin, ...] = Field(min_length=1, max_length=8)
    status: Literal["accepted_research_reaggregation"] = "accepted_research_reaggregation"
    execution: Literal["local_reaggregation_of_accepted_reviews"] = (
        "local_reaggregation_of_accepted_reviews"
    )
    new_model_requests: Literal[0] = 0

    @model_validator(mode="after")
    def batch_identity(self):
        mapping = item_requests(self.origins)
        if (
            self.scope != self.batch_plan.scope
            or set(mapping) != {i.assessment_id for i in self.candidate.items}
            or self.request_ids != tuple(dict.fromkeys(mapping.values()))
            or any(i.request_id != mapping[i.review.assessment_id] for i in self.interpretations)
        ):
            raise ValueError("accepted_aggregation_record_binding")
        return self


class LegacyAggregatedSubmission(AggregatedSubmission):
    """Explicit legacy item family; never invent the newer direction explanation."""

    schema_version: Literal["research-batched-submission/13.0.0"] = (
        "research-batched-submission/13.0.0"
    )
    candidate: ReviewProposal
    compatibility_policy: Literal["identical_or_legacy_review_batch_three_to_one/1.0.0"] = (
        "identical_or_legacy_review_batch_three_to_one/1.0.0"
    )


class LegacyAggregatedRecords(AggregatedRecords):
    schema_version: Literal["case-observation-interpretation/2.15.0"] = (
        "case-observation-interpretation/2.15.0"
    )
    candidate: ReviewProposal
    interpretations: tuple[AcceptedInterpretation, ...]
    compatibility_policy: Literal["identical_or_legacy_review_batch_three_to_one/1.0.0"] = (
        "identical_or_legacy_review_batch_three_to_one/1.0.0"
    )


class AliasAcceptedOrigin(AcceptedOrigin):
    alias_projection: AliasProjection


class AliasAggregatedSubmission(AggregatedSubmission):
    schema_version: Literal["research-batched-submission/17.0.0"] = (
        "research-batched-submission/17.0.0"
    )
    origins: tuple[AliasAcceptedOrigin, ...] = Field(min_length=1, max_length=8)
    compatibility_policy: Literal["original_owner_exact_source_aliases/1.0.0"] = (
        "original_owner_exact_source_aliases/1.0.0"
    )


class AliasAggregatedRecords(AggregatedRecords):
    schema_version: Literal["case-observation-interpretation/2.19.0"] = (
        "case-observation-interpretation/2.19.0"
    )
    origins: tuple[AliasAcceptedOrigin, ...] = Field(min_length=1, max_length=8)
    compatibility_policy: Literal["original_owner_exact_source_aliases/1.0.0"] = (
        "original_owner_exact_source_aliases/1.0.0"
    )


class AggregationReceipt(AcceptanceReceiptV2):
    schema_version: Literal["research-acceptance/3.0.0"] = "research-acceptance/3.0.0"
    execution: Literal["local_reaggregation_of_accepted_reviews"] = (
        "local_reaggregation_of_accepted_reviews"
    )
    new_medical_review: Literal[False] = False
    reviewer_kind: Literal["original_accepted_review_reuse"] = "original_accepted_review_reuse"


class AggregationGrant(Value):
    schema_version: Literal["accepted-aggregation-grant/1.0.0"] = "accepted-aggregation-grant/1.0.0"
    policy: Literal["accepted-local-review-aggregation/1.0.0"] = POLICY
    run_id: Token
    state_checksum: Checksum
    origins_checksum: Checksum
    control_checksum: Checksum
    submission_checksum: Checksum
    issued_at: datetime
    deadline: datetime
    new_requests: Literal[0] = 0

    @model_validator(mode="after")
    def bounded(self):
        if (
            self.issued_at.tzinfo is None
            or self.deadline.tzinfo is None
            or not timedelta(0) < self.deadline - self.issued_at <= timedelta(minutes=15)
        ):
            raise ValueError("accepted_aggregation_window")
        return self


def item_requests(origins):
    result = {}
    runs = set()
    for origin in origins:
        if origin.run_id in runs or set(result) & set(origin.item_requests):
            raise ValueError("accepted_aggregation_duplicate_origin_or_item")
        runs.add(origin.run_id)
        result.update(origin.item_requests)
    return result


class HistoricalRequests:
    """Read all original reservations, including unused failures; never dispatch."""

    def __init__(self, journal):
        self.journal = journal

    def records(self):
        events = self.journal.read()
        ids = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        records = tuple(
            read_request_record(e["record"]) for e in events if e.get("phase") == "completed"
        )
        if (
            len(ids) != len(set(ids))
            or len(records) != len({r.request_id for r in records})
            or set(ids) != {r.request_id for r in records}
        ):
            raise ValueError("accepted_aggregation_unconfirmed_request_inventory")
        return records


class AcceptedOriginReader:
    def __init__(
        self,
        *,
        documents,
        requests,
        state_reader,
        acceptance_key,
        current_config=None,
        reasoned_batch_resize=False,
        client_transition=False,
        network_transition=False,
        confirmed_review_reader=None,
        reused_origin_new_items_only=False,
        alias_manifests=None,
        retained_identity_aliases=False,
    ):
        self.documents, self.requests = documents, requests
        self.state_reader, self.acceptance_key = state_reader, acceptance_key
        self.current_config = current_config
        if network_transition and (
            client_transition or reasoned_batch_resize or alias_manifests is not None
        ):
            raise ValueError("accepted_aggregation_network_combined_transition")
        self.network_transition = network_transition
        if client_transition and (reasoned_batch_resize or alias_manifests is not None):
            raise ValueError("accepted_aggregation_client_combined_transition")
        self.client_transition = client_transition
        self.reasoned_batch_resize = reasoned_batch_resize
        self.confirmed_review_reader = confirmed_review_reader
        self.reused_origin_new_items_only = reused_origin_new_items_only
        self.alias_manifests = alias_manifests
        self.retained_identity_aliases = retained_identity_aliases

    def _committed_cancellation(self, old, control):
        # Default/historical readers never gain this explicit closure capability.
        return False

    def check_compatible_ids(self, old, current, candidate, ids):
        """Ordinary origins must retain every accepted item unchanged."""
        if set(ids) != {i.assessment_id for i in candidate.items}:
            raise ValueError("accepted_aggregation_changed_item")

    def read(self, current):
        if self.network_transition and (
            self.client_transition or self.reasoned_batch_resize or self.alias_manifests is not None
        ):
            raise ValueError("accepted_aggregation_network_combined_transition")
        if self.client_transition and (
            self.reasoned_batch_resize or self.alias_manifests is not None
        ):
            raise ValueError("accepted_aggregation_client_combined_transition")
        from types import SimpleNamespace

        from agents.expert_agent.reporting import read_research_report

        from .acceptance import ResearchAcceptance

        old, docs, key = self.state_reader(), self.documents, self.acceptance_key
        control = docs.sessions.read(docs.run_id)
        if (
            control is None
            or control.phase != "stopped"
            or control.pending
            or old.material.scope.run_id != docs.run_id
            or current.material.scope.run_id == docs.run_id
        ):
            raise ValueError("accepted_aggregation_origin_not_terminal")
        committed_cancel = (
            self._committed_cancellation(old, control) if control.cancelled else False
        )
        if (
            control.cancelled
            and not committed_cancel
            and key not in {DEADLINE_ACCEPTANCE_KEY, CITATION_ACCEPTANCE_KEY}
        ):
            raise ValueError("accepted_aggregation_origin_not_terminal")
        docs.validate_scope(old.material.scope)
        saved, bundle = docs.read("submission:" + key), docs.read("accepted:" + key)
        if not saved or not bundle or len(bundle) != 4:
            raise ValueError("accepted_aggregation_origin_missing")
        sub = read_submission(saved[0]["submission"])
        from .accepted_review_reuse import ReusedAcceptedSubmission, ReusedAcceptedSubmissionV7

        reused = isinstance(sub, (ReusedAcceptedSubmission, ReusedAcceptedSubmissionV7))
        if control.cancelled and not (
            committed_cancel
            or type(sub) is PartialReasonedSubmission
            or key == CITATION_ACCEPTANCE_KEY
            and type(sub) is PartialBatchedSubmission
        ):
            raise ValueError("accepted_aggregation_origin_not_terminal")
        if not (reused and self.reused_origin_new_items_only) and type(sub) not in {
            BatchedSubmission,
            PartialBatchedSubmission,
            ReasonedSubmission,
            PartialReasonedSubmission,
            ExplanationSubmission,
        }:
            raise ValueError("accepted_aggregation_origin_version")
        report = read_research_report(
            ResearchAcceptance(
                owner=SimpleNamespace(current=lambda: old), documents=docs, requests=None
            ),
            key=key,
            parent=old,
        )
        if (
            report["receipt"]["status"] != "accepted"
            or report["receipt"]["submission_checksum"] != digest(sub)
            or report["records"]["candidate"] != sub.candidate.model_dump(mode="json")
            or not docs.consumed(old.material.scope.graph)
        ):
            raise ValueError("accepted_aggregation_origin_receipt")
        if reused:
            from .review_continuation import check_continued

            # Recheck the installed original ancestor owner; saved reuse metadata
            # alone is not authority. Only this run's fresh batches enter the union.
            if self.confirmed_review_reader is None:
                raise ValueError("accepted_aggregation_ancestor_owner_required")
            checked = check_continued(
                old,
                sub,
                owner=SimpleNamespace(confirmed_review_reader=self.confirmed_review_reader),
                documents=docs,
                requests=self.requests,
            )
        else:
            checked = (
                check_submission(old, sub, docs, self.requests)
                if isinstance(sub, ExplanationSubmission)
                else validate_batch_receipts(old, sub, docs, self.requests)
            )
        records = tuple(self.requests.records())
        permission_sha = report["receipt"].get("permission_checksum")
        if any(not r.simulated for r in checked):
            raw = docs.read("research-live-grant")
            grant = validate(ResearchLiveGrant, raw[0]) if raw else None
            s = old.material.scope
            if (
                grant is None
                or not self.requests.journal.durable
                or (
                    grant.run_id,
                    grant.case_code,
                    grant.input_checksum,
                    grant.visible_checksum,
                    grant.policy_checksum,
                    grant.config_checksum,
                    grant.root_checksum,
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
                or "review" not in grant.roles
                or any(r.simulated or r.deadline > grant.deadline for r in checked)
            ):
                raise ValueError("accepted_aggregation_original_permission")
            expected_permission = digest(grant)
            if isinstance(sub, PartialReasonedSubmission) and key == DEADLINE_ACCEPTANCE_KEY:
                local = accepted_deadline_permission(old, docs, self.requests, sub, grant)
                expected_permission = digest(local)
            if (
                isinstance(sub, (PartialBatchedSubmission, PartialReasonedSubmission))
                and key == CITATION_ACCEPTANCE_KEY
            ):
                # Exact committed local fact provenance, never a renewed live grant.
                # The original rejected response remains excluded by its owner.
                local = accepted_citation_permission(old, docs, self.requests)
                expected_permission = digest(local)
            if isinstance(sub, ExplanationSubmission):
                from .recorded_explanation import GRANT_KEY as KEY
                from .recorded_explanation import original_failure

                local = validate(ExplanationGrant, docs.read(KEY)[0])
                if (
                    local.run_id != docs.run_id
                    or local.original_grant_checksum != digest(grant)
                    or local.control_checksum != digest(control)
                    or local.state_checksum != digest(old)
                    or local.failed_result_checksum != digest(original_failure(docs))
                    or local.aggregate_checksum != digest(sub)
                    or local.request_checksums != tuple(digest(r) for r in checked)
                    or local.interpretation_checksum != digest(sub.local_interpretation)
                ):
                    raise ValueError("accepted_aggregation_local_origin_permission")
                expected_permission = digest(local)
            if permission_sha != expected_permission:
                raise ValueError("accepted_aggregation_original_permission")
        config_pair = None
        if self.current_config is not None and old.config_checksum != current.config_checksum:
            allowed = (
                {
                    ReasonedSubmission,
                    PartialReasonedSubmission,
                    BatchedSubmission,
                    PartialBatchedSubmission,
                }
                if self.client_transition or self.network_transition
                else {ReasonedSubmission, PartialReasonedSubmission}
                if self.reasoned_batch_resize
                else {BatchedSubmission, PartialBatchedSubmission}
            )
            if type(sub) not in allowed:
                raise ValueError("accepted_aggregation_batch_resize_legacy_only")
            configs = []
            for batch in sub.batches:
                response = docs.read(
                    "review-batches:" + sub.plan.key + ":" + batch.batch_id + ":response"
                )
                if not response or len(response) != 1 or "provider_config" not in response[0]:
                    raise ValueError("accepted_aggregation_original_batch_config_missing")
                configs.append(response[0]["provider_config"])
            if not configs or any(digest(c) != old.config_checksum for c in configs):
                raise ValueError("accepted_aggregation_original_batch_config_binding")
            config_pair = (configs[0], self.current_config)
        projected, candidate, alias_proof = old, sub.candidate, None
        if self.alias_manifests is not None:
            if (
                config_pair is not None
                or (reused and not self.retained_identity_aliases)
                or not isinstance(sub.candidate, ReasonedReviewProposal)
            ):
                raise ValueError("accepted_aggregation_alias_combined_transition")
            projected, candidate, alias_proof = project_aliases(
                old, current, sub.candidate, *self.alias_manifests()
            )
            if reused and (
                projected != old
                or candidate != sub.candidate
                or alias_proof.original_manifest_checksum != alias_proof.current_manifest_checksum
                or any(t.before != t.after for t in alias_proof.transfers)
            ):
                # Its ancestor was already revalidated by check_continued. Do not
                # rename or rebind a retained context after that owner check.
                raise ValueError("accepted_aggregation_retained_alias_identity_required")
        ids = compatible_local_ids(
            projected,
            current,
            candidate,
            include_unassessed=True,
            **(
                dict(network_configs=config_pair)
                if self.network_transition
                else dict(client_configs=config_pair)
                if self.client_transition
                else dict(reasoned_batching_configs=config_pair)
                if self.reasoned_batch_resize
                else dict(batching_configs=config_pair)
            ),
        )
        self.check_compatible_ids(projected, current, candidate, ids)
        selected_items = tuple(i for i in candidate.items if i.assessment_id in ids)
        if reused:
            from .review_continuation import reused_ids

            fresh = {i.assessment_id for batch in sub.batches for i in batch.candidate.items}
            if not fresh or fresh != set(ids) - set(reused_ids(sub.historical)):
                raise ValueError("accepted_aggregation_new_batch_partition")
            ids = tuple(i for i in ids if i in fresh)
            selected_items = tuple(i for i in selected_items if i.assessment_id in fresh)
        mapped = {i: request_for_item(sub, i) for i in ids}
        proof_type = AliasAcceptedOrigin if alias_proof is not None else AcceptedOrigin
        proof = proof_type(
            **(dict(alias_projection=alias_proof) if alias_proof is not None else {}),
            run_id=docs.run_id,
            acceptance_key=key,
            acceptance_checksum=digest(bundle),
            submission_checksum=digest(sub),
            state_checksum=digest(old),
            control_checksum=digest(control),
            journal_checksum=digest(self.requests.journal.read()),
            permission_checksum=permission_sha,
            item_requests=mapped,
            unused_request_ids=tuple(
                r.request_id for r in records if r.request_id not in mapped.values()
            ),
        )
        return proof, selected_items, checked


class CommittedClosureOriginReader(AcceptedOriginReader):
    """Cancelled technical closeout may retain an exact previously committed Review.

    This does not reopen the old run. Current source/material, accepted receipt,
    request confirmation and new aggregation permission are still required by
    the ordinary reader. Only the original recorded-closure owner grants this
    narrow local reuse route, never an exported score or caller authorization flag.
    """

    def __init__(self, *, campaign, closure_checksum, **kwargs):
        super().__init__(**kwargs)
        self.campaign, self.closure_checksum = campaign, closure_checksum

    def _committed_cancellation(self, old, control):
        from packages.multimodal_research.campaign_renewal import recorded_acquisition_origin

        if control.phase != "stopped" or control.pending is not None:
            raise ValueError("accepted_aggregation_committed_control_not_closed")
        docs = self.documents
        proof = recorded_acquisition_origin(self.campaign, docs.run_id)
        confirmation = proof.get("confirmation", {})
        if (
            proof.get("schema_version") != "renewal-recorded-origin/1.0.0"
            or proof.get("closure_checksum") != self.closure_checksum
            or confirmation.get("run_id") != docs.run_id
            or confirmation.get("acceptance_key") != self.acceptance_key
            or confirmation.get("parent_checksum") != digest(old)
            or confirmation.get("accepted_checksum")
            != digest(docs.read("accepted:" + self.acceptance_key))
            or confirmation.get("journal_checksum") != digest(self.requests.journal.read())
            or docs.sessions.read(docs.run_id) != control
        ):
            raise ValueError("accepted_aggregation_committed_closure_binding")
        return True


class NullReviewRevision(Value):
    """Exact accepted-record selection, not caller-supplied replacement scores."""

    assessment_id: Token
    previous_run_id: Token
    replacement_run_id: Token
    previous_submission_checksum: Checksum
    replacement_submission_checksum: Checksum
    previous_item_checksum: Checksum
    replacement_item_checksum: Checksum
    new_context_checksum: Checksum
    reason: Literal["new_owner_bound_method_context"] = "new_owner_bound_method_context"


def _target_context(state, target):
    """Identify saved unscored context; original state_reader reprojects its bytes.

    This only verifies structural attribution and new material presence. It does
    not prove the medical explanation or interpret fragmented JSON as evidence.
    """
    import json

    values = []
    for observation in state.material.observations:
        if observation.method != "owner_display_context_not_disease_evidence":
            continue
        try:
            envelope = json.loads(observation.value)
        except (ValueError, TypeError):
            raise ValueError("null_revision_context_format") from None
        if not isinstance(envelope, dict):
            raise ValueError("null_revision_context_format")
        if envelope.get("assessment_id") != target:
            continue
        if observation.status != "obtained" or observation.unit != "context_only":
            raise ValueError("null_revision_context_status")
        values.append(observation.reference)
    return tuple(sorted(values, key=lambda v: v.code))


def select_null_revisions(values, states, revisions):
    """Run only after every original AcceptedOriginReader has validated its origin.

    Changed medical facts/definitions still fail the ordinary compatibility
    checks. Here only a duplicated, previously null judgment can be superseded
    by an already accepted new Review with added task-bound method context.
    The old acceptance/submission checksum remains on its retained origin;
    the exact revision and both item hashes belong to the versioned saved plan.
    """
    from collections import Counter

    if (
        not revisions
        or len(revisions) > 6
        or any(type(r) is not NullReviewRevision for r in revisions)
    ):
        raise ValueError("null_revision_explicit_bounded_plan")
    by_run = {value[0].run_id: value for value in values}
    if len(by_run) != len(values) or set(states) != set(by_run):
        raise ValueError("null_revision_origin_inventory")
    for run_id, (origin, _, _) in by_run.items():
        if type(origin) is not AcceptedOrigin or digest(states[run_id]) != origin.state_checksum:
            raise ValueError("null_revision_original_state_binding")
    occurrences = Counter(i.assessment_id for _, items, _ in values for i in items)
    selected = {r.assessment_id for r in revisions}
    if len(selected) != len(revisions) or {i for i, n in occurrences.items() if n > 1} != selected:
        raise ValueError("null_revision_duplicate_selection")
    omitted = {}
    for revision in revisions:
        target = revision.assessment_id
        if revision.previous_run_id == revision.replacement_run_id or occurrences[target] != 2:
            raise ValueError("null_revision_requires_distinct_accepted_runs")
        previous = by_run.get(revision.previous_run_id)
        replacement = by_run.get(revision.replacement_run_id)
        if previous is None or replacement is None:
            raise ValueError("null_revision_unknown_origin")
        old_origin, old_items, _ = previous
        new_origin, new_items, _ = replacement
        old = next((i for i in old_items if i.assessment_id == target), None)
        new = next((i for i in new_items if i.assessment_id == target), None)
        if (
            old is None
            or new is None
            or old.judgment.value is not None
            or old.status == "assessed"
            or not old.observation_available
            or not new.observation_available
            or (old_origin.submission_checksum, new_origin.submission_checksum)
            != (revision.previous_submission_checksum, revision.replacement_submission_checksum)
            or (digest(old), digest(new))
            != (revision.previous_item_checksum, revision.replacement_item_checksum)
            or old_origin.item_requests.get(target) is None
            or new_origin.item_requests.get(target) is None
            or old_origin.item_requests[target] == new_origin.item_requests[target]
        ):
            raise ValueError("null_revision_accepted_item_binding")
        old_state, new_state = states[old_origin.run_id], states[new_origin.run_id]
        # No vote can be manufactured from missing patient observations, and no
        # changed primary observation is smuggled through a metadata revision.
        old_policy = next((i for i in old_state.policy.items if i.id == target), None)
        new_policy = next((i for i in new_state.policy.items if i.id == target), None)
        if old_policy is None or old_policy != new_policy:
            raise ValueError("null_revision_task_changed")
        code = old_policy.observation_code
        old_observation = next(
            (o for o in old_state.material.observations if o.reference.code == code), None
        )
        new_observation = next(
            (o for o in new_state.material.observations if o.reference.code == code), None
        )
        if (
            old_observation is None
            or old_observation != new_observation
            or old_observation.status != "obtained"
        ):
            raise ValueError("null_revision_patient_observation_changed_or_missing")
        before, after = _target_context(old_state, target), _target_context(new_state, target)
        if not after or before == after or digest(after) != revision.new_context_checksum:
            raise ValueError("null_revision_new_material_required")
        omitted.setdefault(old_origin.run_id, set()).add(target)
    output = []
    for origin, items, records in values:
        remove = omitted.get(origin.run_id, set())
        if not remove:
            output.append((origin, items, records))
            continue
        retained = tuple(i for i in items if i.assessment_id not in remove)
        # Keep a nonempty original origin in this initial bounded protocol. A
        # fully superseded origin needs a separate explicit historical carrier.
        if not retained:
            raise ValueError("null_revision_empty_historical_origin")
        mapping = {k: v for k, v in origin.item_requests.items() if k not in remove}
        unused = tuple(
            dict.fromkeys(
                (*origin.unused_request_ids, *(origin.item_requests[k] for k in sorted(remove)))
            )
        )
        unused = tuple(r for r in unused if r not in mapping.values())
        proof = origin.model_copy(update=dict(item_requests=mapping, unused_request_ids=unused))
        output.append((proof, retained, records))
    return output


class AcceptedReviewSet:
    """Installed original-owner readers, not caller-provided candidates or scores."""

    def __init__(self, readers, config, *, null_revisions=()):
        if not 1 <= len(readers) <= 8 or not all(
            isinstance(r, AcceptedOriginReader) for r in readers
        ):
            raise ValueError("accepted_aggregation_readers_required")
        self.readers, self.config = tuple(readers), config
        self.null_revisions = tuple(null_revisions)

    def current(self, state):
        values = [r.read(state) for r in self.readers]
        if self.null_revisions:
            states = {r.documents.run_id: r.state_reader() for r in self.readers}
            values = select_null_revisions(values, states, self.null_revisions)
        origins = tuple(v[0] for v in values)
        mapping = item_requests(origins)
        items = sorted((i for v in values for i in v[1]), key=lambda i: i.assessment_id)
        if set(mapping) != {p.id for p in state.policy.items}:
            raise ValueError("accepted_aggregation_incomplete_current_observations")
        plan = make_plan(
            state, self.config.review_batching, key="accepted-aggregation:A", round_id="A"
        )
        from .acceptance_models import TaskPolicyV7

        reasoned = isinstance(state.policy, TaskPolicyV7)
        candidate_type = ReasonedReviewProposal if reasoned else ReviewProposal
        submission_type = AggregatedSubmission if reasoned else LegacyAggregatedSubmission
        if any(isinstance(o, AliasAcceptedOrigin) for o in origins):
            if not reasoned or not all(isinstance(o, AliasAcceptedOrigin) for o in origins):
                raise ValueError("accepted_aggregation_alias_origin_family")
            submission_type = AliasAggregatedSubmission
        candidate = candidate_type(
            scope_checksum=digest(state.material.scope), round="A", items=tuple(items)
        )
        sub = submission_type(
            scope=state.material.scope,
            config_checksum=state.config_checksum,
            prompt_checksum=state.prompt_checksum,
            observation_checksums=tuple(o.reference for o in state.material.observations),
            source_checksums=tuple(s.reference for s in state.material.sources),
            plan=plan,
            batches=(),
            candidate=candidate,
            origins=origins,
        )
        records = tuple(r for v in values for r in v[2])
        if len(records) != len({r.request_id for r in records}):
            raise ValueError("accepted_aggregation_duplicate_request")
        return sub, records


class AggregationRequests:
    def __init__(self, source, owner, journal):
        self.source, self.owner, self.journal = source, owner, journal

    def records(self):
        return self.source.current(self.owner.current())[1]


def rejected_aggregation(documents, *, manifest, plan, original_result_checksum):
    """Read an exact rejected local attempt. This grants no current consumption.

    The successor must independently reconstruct current origins and obtain its
    own bounded local grant. An old HTTP failure or unconfirmed request is not
    a rejected aggregation and cannot enter this recovery route.
    """
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    from .acceptance_models import DomainState

    docs = documents
    saved = docs.read("submission:" + ACCEPTANCE_KEY)
    rejected = docs.read("accepted:" + ACCEPTANCE_KEY)
    raw_state = docs.read("dynamic-phase-input")
    raw_grant = docs.read(GRANT_KEY)
    control = docs.sessions.read(docs.run_id)
    journal = RuntimeAgentJournal(docs)
    if (
        manifest.get("schema_version")
        not in {
            "accepted-review-aggregation-manifest/1.0.0",
            "accepted-review-aggregation-manifest/1.1.0",
        }
        or manifest.get("run_id") != docs.run_id
        or manifest.get("plan") != plan
        or manifest.get("original_result_checksum") != original_result_checksum
        or docs.read("accepted-aggregation-manifest") != (manifest,)
        or not saved
        or len(saved) != 1
        or saved[0].get("status") != "submitted_not_accepted"
        or not rejected
        or len(rejected) != 1
        or not raw_state
        or len(raw_state) != 1
        or not raw_grant
        or len(raw_grant) != 1
        or control is None
        or control.phase != "stopped"
        or control.pending
        or control.cancelled
        or not journal.durable
        or journal.read()
        or docs.read("multimodal-case-result") is not None
        or docs.read("dynamic-phase-result") is not None
    ):
        raise ValueError("accepted_aggregation_recovery_requires_rejected_local_attempt")
    state = validate(DomainState, raw_state[0])
    sub = read_submission(saved[0]["submission"])
    grant = validate(AggregationGrant, raw_grant[0])
    receipt = validate(AggregationReceipt, rejected[0])
    if (
        not isinstance(sub, AggregatedSubmission)
        or receipt.status != "rejected"
        or receipt.reason != "domain_acceptance_rejected"
        or receipt.graph is not None
        or receipt.records_checksum is not None
        or receipt.scores_checksum is not None
        or receipt.parent_graph.model_dump(mode="json")
        != state.material.scope.graph.model_dump(
            mode="json", include={"role", "code", "version", "checksum"}
        )
        or receipt.permission_checksum != digest(grant)
        or receipt.submission_checksum != digest(sub)
        or saved[0]["submission_checksum"] != digest(sub)
        or (
            grant.run_id,
            grant.state_checksum,
            grant.control_checksum,
            grant.submission_checksum,
            grant.origins_checksum,
        )
        != (docs.run_id, digest(state), digest(control), digest(sub), digest(sub.origins))
        or state.material.scope.run_id != docs.run_id
        or control.binding.run_id != docs.run_id
        or sub.scope != state.material.scope
        or manifest.get("state_checksum") != digest(state)
        or manifest.get("submission_checksum") != digest(sub)
        or manifest.get("origins_checksum") != digest(sub.origins)
    ):
        raise ValueError("accepted_aggregation_recovery_original_binding")
    docs.validate_scope(state.material.scope)
    if docs.sessions.read(docs.run_id) != control or journal.read():
        raise ValueError("accepted_aggregation_recovery_control_changed")
    proof = dict(
        schema_version="rejected-aggregation-origin/1.0.0",
        run_id=docs.run_id,
        manifest_checksum=digest(manifest),
        state_checksum=digest(state),
        submission_checksum=digest(sub),
        rejection_checksum=digest(rejected),
        grant_checksum=digest(grant),
        control_checksum=digest(control),
        journal_checksum=digest(journal.read()),
        request_history_checksum=manifest.get("request_history_checksum"),
        new_model_requests=0,
    )
    return proof, state, sub


def check_recovery_content(old_state, old_submission, state, submission):
    """Only the local run/scope digest may change; no new judgment is inferred."""

    def material_identity(value):
        raw = value.model_dump(mode="json")
        raw["material"]["scope"].pop("run_id")
        raw["material"]["scope"].pop("material_checksum")
        return raw

    if (
        old_state.material.scope.run_id == state.material.scope.run_id
        or material_identity(old_state) != material_identity(state)
        or old_submission.origins != submission.origins
        or old_submission.candidate.items != submission.candidate.items
        or old_submission.observation_checksums != submission.observation_checksums
        or old_submission.source_checksums != submission.source_checksums
        or old_submission.schema_version != submission.schema_version
    ):
        raise ValueError("accepted_aggregation_recovery_current_content_changed")


def check_aggregation(state, submission, owner):
    source = getattr(owner, "accepted_review_set", None)
    if not isinstance(source, AcceptedReviewSet):
        raise ValueError("accepted_aggregation_owner_required")
    expected, records = source.current(state)
    if expected != submission:
        raise ValueError("accepted_aggregation_content_changed")
    return records


class AggregationPermission(ResearchLivePermission):
    def __init__(self, *, grant, config, documents, requests, owner, clock=None):
        self.grant = validate(AggregationGrant, grant)
        self.config, self.documents, self.budget, self.owner = config, documents, requests, owner
        self.clock = clock or (lambda: datetime.now(UTC))

    @classmethod
    def prepare(cls, *, owner, documents, requests, config, clock=None):
        state = owner.current()
        sub, _ = owner.accepted_review_set.current(state)
        now = (clock or (lambda: datetime.now(UTC)))()
        expected = dict(
            run_id=documents.run_id,
            state_checksum=digest(state),
            origins_checksum=digest(sub.origins),
            control_checksum=digest(documents.sessions.read(documents.run_id)),
            submission_checksum=digest(sub),
        )
        prior = documents.read(GRANT_KEY)
        g = (
            validate(AggregationGrant, prior[0])
            if prior
            else AggregationGrant(**expected, issued_at=now, deadline=now + timedelta(minutes=15))
        )
        if any(getattr(g, k) != v for k, v in expected.items()):
            raise ValueError("accepted_aggregation_grant_conflict")
        p = cls(
            grant=g, config=config, documents=documents, requests=requests, owner=owner, clock=clock
        )
        p.check(state, role="review", persisted=False)
        with documents.session():
            documents.write(GRANT_KEY, (g,))
        return p, sub

    def check_submission(self, state, submission):
        """Combine the two original checks without repeating source reconstruction."""
        self._check(state, role="review", submission=submission)

    def check(self, state, *, role=None, record=None, persisted=True):
        self._check(state, role=role, record=record, persisted=persisted)

    def _local(self, state, *, role=None):
        g, docs = self.grant, self.documents
        control = docs.sessions.read(docs.run_id)
        if (
            self.clock() >= g.deadline
            or role not in {None, "review"}
            or g.run_id != docs.run_id
            or state.material.scope.run_id != docs.run_id
            or digest(state) != g.state_checksum
            or control is None
            or control.phase != "stopped"
            or control.pending
            or control.cancelled
            or digest(control) != g.control_checksum
            or control.binding.run_id != docs.run_id
            or (
                control.binding.case_code,
                control.binding.input_checksum,
                control.binding.visible_checksum,
            )
            != (
                state.material.scope.case_code,
                state.material.scope.input_checksum,
                state.material.scope.visible_checksum,
            )
            or self.budget.journal.documents is not docs
            or not self.budget.journal.durable
            or self.budget.journal.read()
            or state.config_checksum != digest(self.config)
        ):
            raise ValueError("accepted_aggregation_local_permission")
        return control

    def check_commit(self, state, submission):
        """Final local guard after provenance reads, without a second origin traversal."""
        control = self._local(state, role="review")
        g, docs = self.grant, self.documents
        if (
            digest(submission) != g.submission_checksum
            or digest(submission.origins) != g.origins_checksum
            or docs.read(GRANT_KEY) != (g.model_dump(mode="json"),)
        ):
            raise ValueError("accepted_aggregation_origin_changed")
        docs.validate_scope(state.material.scope)
        if docs.sessions.read(docs.run_id) != control:
            raise ValueError("accepted_aggregation_control_changed")
        if self.clock() >= g.deadline:
            raise ValueError("accepted_aggregation_deadline")

    def _check(self, state, *, role=None, record=None, persisted=True, submission=None):
        control = self._local(state, role=role)
        g, docs = self.grant, self.documents
        source = getattr(self.owner, "accepted_review_set", None)
        if not isinstance(source, AcceptedReviewSet):
            raise ValueError("accepted_aggregation_owner_required")
        sub, records = source.current(state)
        if submission is not None and sub != submission:
            raise ValueError("accepted_aggregation_content_changed")
        if (
            digest(sub) != g.submission_checksum
            or digest(sub.origins) != g.origins_checksum
            or record is not None
            and record not in records
            or persisted
            and docs.read(GRANT_KEY) != (g.model_dump(mode="json"),)
        ):
            raise ValueError("accepted_aggregation_origin_changed")
        docs.validate_scope(state.material.scope)
        # Origin verification may do I/O. Re-read the original Control after it;
        # no earlier snapshot grants a late, cancelled or changed composition.
        if docs.sessions.read(docs.run_id) != control:
            raise ValueError("accepted_aggregation_control_changed")
        if self.clock() >= g.deadline:
            raise ValueError("accepted_aggregation_deadline")
