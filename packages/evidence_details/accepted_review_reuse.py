"""Reuse accepted local judgments, not permission or a fresh integrated review.

The original full context remains attached. Newly obtained observations may be
added, but existing obtained context, definitions and their source bindings may
not change. This is deliberately stricter than copying scores from an export.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

from pydantic import Field

from packages.grounding.models import Checksum, Token, Value, digest, validate

from .review_batches import (
    ReasonedBatchResult,
    ReasonedSubmission,
    ReusePartitionPlan,
    ReviewBatchPlan,
    read_submission,
    validate_batch_receipts,
)
from .review_continuation import ContinuedReasonedSubmission
from .review_format_continuation import FormatContinuedSubmission

PROVENANCE_POLICY = "accepted-review-provenance-bound/2.0.0"
MAX_ACCEPTED_REVIEW_DEPTH = 16


class AcceptedReviewOrigin(Value):
    schema_version: Literal["accepted-review-origin/1.0.0"] = "accepted-review-origin/1.0.0"
    run_id: Token
    acceptance_key: Token
    acceptance_checksum: Checksum
    state_checksum: Checksum
    control_checksum: Checksum
    journal_checksum: Checksum
    semantic_checksum: Checksum
    assessment_ids: tuple[Token, ...]
    submission: ReasonedSubmission
    semantics: Literal["unchanged_local_judgments_not_new_joint_review"] = (
        "unchanged_local_judgments_not_new_joint_review"
    )


class ReusedAcceptedSubmission(ReasonedSubmission):
    schema_version: Literal["research-batched-submission/6.0.0"] = (
        "research-batched-submission/6.0.0"
    )
    historical: AcceptedReviewOrigin


class AcceptedReviewOriginV11(AcceptedReviewOrigin):
    schema_version: Literal["accepted-review-origin/1.1.0"] = "accepted-review-origin/1.1.0"
    submission: ReasonedSubmission | ReusedAcceptedSubmission | ReusedAcceptedSubmissionV7 = Field(
        discriminator="schema_version"
    )


class ReusedAcceptedSubmissionV7(ReasonedSubmission):
    schema_version: Literal["research-batched-submission/7.0.0"] = (
        "research-batched-submission/7.0.0"
    )
    historical: AcceptedReviewOriginV11


AcceptedReviewOriginV11.model_rebuild()
ReusedAcceptedSubmissionV7.model_rebuild()


class AcceptedReviewOriginV12(AcceptedReviewOriginV11):
    """Retain the exact confirmed-failure provenance across accepted successors."""

    schema_version: Literal["accepted-review-origin/1.2.0"] = "accepted-review-origin/1.2.0"
    submission: (
        ReasonedSubmission
        | ReusedAcceptedSubmission
        | ReusedAcceptedSubmissionV7
        | FormatContinuedSubmission
        | ReusedAcceptedSubmissionV9
    ) = Field(discriminator="schema_version")


class ReusedAcceptedSubmissionV9(ReusedAcceptedSubmissionV7):
    schema_version: Literal["research-batched-submission/9.0.0"] = (
        "research-batched-submission/9.0.0"
    )
    historical: AcceptedReviewOriginV12


AcceptedReviewOriginV12.model_rebuild()
ReusedAcceptedSubmissionV9.model_rebuild()


class AcceptedReviewOriginV13(AcceptedReviewOriginV12):
    schema_version: Literal["accepted-review-origin/1.3.0"] = "accepted-review-origin/1.3.0"
    submission: (
        ReasonedSubmission
        | ReusedAcceptedSubmission
        | ReusedAcceptedSubmissionV7
        | FormatContinuedSubmission
        | ReusedAcceptedSubmissionV9
        | ReusedAcceptedSubmissionV10
    ) = Field(discriminator="schema_version")


class ReusedAcceptedSubmissionV10(ReusedAcceptedSubmissionV9):
    schema_version: Literal["research-batched-submission/10.0.0"] = (
        "research-batched-submission/10.0.0"
    )
    plan: ReusePartitionPlan
    historical: (
        AcceptedReviewOrigin
        | AcceptedReviewOriginV11
        | AcceptedReviewOriginV12
        | AcceptedReviewOriginV13
    ) = Field(discriminator="schema_version")


AcceptedReviewOriginV13.model_rebuild()
ReusedAcceptedSubmissionV10.model_rebuild()


class AcceptedReviewOriginV14(AcceptedReviewOriginV13):
    """Accepted same-contract continuation, retaining its confirmed ancestor."""

    schema_version: Literal["accepted-review-origin/1.4.0"] = "accepted-review-origin/1.4.0"
    submission: ContinuedReasonedSubmission | ReusedAcceptedSubmissionV14 = Field(
        discriminator="schema_version"
    )


class ReusedAcceptedSubmissionV14(ReusedAcceptedSubmissionV10):
    schema_version: Literal["research-batched-submission/14.0.0"] = (
        "research-batched-submission/14.0.0"
    )
    plan: ReviewBatchPlan | ReusePartitionPlan = Field(discriminator="schema_version")
    historical: AcceptedReviewOriginV14


AcceptedReviewOriginV14.model_rebuild()
ReusedAcceptedSubmissionV14.model_rebuild()


class AcceptedReviewOriginV15(AcceptedReviewOriginV13):
    """An accepted ordinary successor of a partitioned origin."""

    schema_version: Literal["accepted-review-origin/1.5.0"] = "accepted-review-origin/1.5.0"
    submission: ReusedAcceptedSubmissionV15


class ReusedAcceptedSubmissionV15(ReusedAcceptedSubmissionV10):
    schema_version: Literal["research-batched-submission/15.0.0"] = (
        "research-batched-submission/15.0.0"
    )
    plan: ReviewBatchPlan | ReusePartitionPlan = Field(discriminator="schema_version")
    historical: AcceptedReviewOriginV13 | AcceptedReviewOriginV15 = Field(
        discriminator="schema_version"
    )


AcceptedReviewOriginV15.model_rebuild()
ReusedAcceptedSubmissionV15.model_rebuild()


class AcceptedReviewOriginV16(AcceptedReviewOrigin):
    schema_version: Literal["accepted-review-origin/1.6.0"] = "accepted-review-origin/1.6.0"
    submission: ReusedAcceptedSubmissionV16


class ReusedAcceptedSubmissionV16(ReusedAcceptedSubmissionV7):
    """Explicit zero-new-batch composition and its subsequent ordinary extensions.

    Empty fresh Review remains invalid. Only the original accepted owner can
    establish full coverage; no model response or direction is fabricated.
    """

    schema_version: Literal["research-batched-submission/16.0.0"] = (
        "research-batched-submission/16.0.0"
    )
    batches: tuple[ReasonedBatchResult, ...] = Field(max_length=20)
    plan: ReviewBatchPlan | ReusePartitionPlan = Field(discriminator="schema_version")
    historical: (
        AcceptedReviewOrigin
        | AcceptedReviewOriginV11
        | AcceptedReviewOriginV12
        | AcceptedReviewOriginV13
        | AcceptedReviewOriginV14
        | AcceptedReviewOriginV15
        | AcceptedReviewOriginV16
    ) = Field(discriminator="schema_version")
    composition: Literal["all_current_items_reused", "accepted_origin_with_new_batches"]


AcceptedReviewOriginV16.model_rebuild()
ReusedAcceptedSubmissionV16.model_rebuild()


def all_reused_live(owner, state):
    """Execution identity comes from the verified accepted bundle, not caller flags."""
    reader = owner.confirmed_review_reader
    origin = reader.current(state)
    if not isinstance(origin, AcceptedReviewOrigin):
        raise ValueError("all_reused_accepted_origin_required")
    from .acceptance_models import read_records

    records = read_records(reader.documents.read("accepted:" + origin.acceptance_key)[0])
    return records.execution == "bounded_live_research"


def provenance_batches(submission, *, depth=0):
    """Oldest-first exact batches, bounded to the campaign's 16 successor levels.

    This is traversal policy, not a new score or acceptance grant. Every ancestor
    is still qualified through AcceptedReviewReader and the original current owners.
    Iteration prevents recursive stack growth; both object and run cycles fail.
    """
    if type(depth) is not int or not 0 <= depth <= MAX_ACCEPTED_REVIEW_DEPTH:
        raise ValueError("accepted_review_reuse_depth")
    groups, seen_objects, seen_runs = [], set(), set()
    current = submission
    while True:
        if depth > MAX_ACCEPTED_REVIEW_DEPTH:
            raise ValueError("accepted_review_reuse_depth")
        if id(current) in seen_objects:
            raise ValueError("accepted_review_reuse_cycle_or_depth")
        seen_objects.add(id(current))
        groups.append(current.batches)
        if not isinstance(
            current,
            (ReusedAcceptedSubmission, ReusedAcceptedSubmissionV7, ContinuedReasonedSubmission),
        ):
            break
        origin = current.historical
        if origin.run_id in seen_runs:
            raise ValueError("accepted_review_reuse_cycle_or_depth")
        seen_runs.add(origin.run_id)
        current = origin.submission
        depth += 1
    return tuple(batch for group in reversed(groups) for batch in group)


def compatible_local_ids(
    old,
    current,
    candidate,
    *,
    include_unassessed=False,
    batching_configs=None,
    reasoned_batching_configs=None,
    network_configs=None,
    capacity_configs=None,
    client_configs=None,
):
    """Owner state, never caller flags, defines the reusable subset.

    The previous request could see all obtained observations: preserve that context,
    not just each item's primary observation. Missing inputs may become obtained.
    Any changed existing fact requires explicit re-review, not silent reuse.
    """
    config_matches = old.config_checksum == current.config_checksum
    if client_configs is not None:
        from packages.llm_provider.config import read_provider_config, with_http_client

        if any(
            c is not None
            for c in (
                batching_configs,
                reasoned_batching_configs,
                network_configs,
                capacity_configs,
            )
        ):
            raise ValueError("accepted_review_client_combined_transition")
        previous, following = [read_provider_config(c) for c in client_configs]
        if (digest(previous), digest(following)) != (old.config_checksum, current.config_checksum):
            raise ValueError("accepted_review_client_config_binding")
        client = getattr(following, "http_client", None)
        if (
            (previous.schema_version, following.schema_version)
            not in {
                ("agent-provider-config/7.1.0", "agent-provider-config/7.2.0"),
                ("agent-provider-config/9.1.0", "agent-provider-config/9.2.0"),
            }
            or client is None
            or with_http_client(previous, client) != following
        ):
            raise ValueError("accepted_review_client_config_delta")
        # Only the HTTP implementation changed. Original observations, relation,
        # task, prompts, model, route, limits and exact request proofs still hold.
        config_matches = True
    if capacity_configs is not None:
        from packages.llm_provider.config import read_provider_config

        if (
            batching_configs is not None
            or reasoned_batching_configs is not None
            or network_configs is not None
        ):
            raise ValueError("accepted_review_capacity_combined_transition")
        previous, following = [read_provider_config(c) for c in capacity_configs]
        if (digest(previous), digest(following)) != (old.config_checksum, current.config_checksum):
            raise ValueError("accepted_review_capacity_config_binding")
        before, after = previous.model_dump(mode="json"), following.model_dump(mode="json")
        if not previous.review_batching.max_batches < following.review_batching.max_batches <= 20:
            raise ValueError("accepted_review_capacity_config_delta")
        before["review_batching"]["max_batches"] = following.review_batching.max_batches
        if before != after:
            raise ValueError("accepted_review_capacity_config_delta")
        config_matches = True
    if network_configs is not None:
        from packages.llm_provider.config import read_provider_config, with_network_route

        if batching_configs is not None or reasoned_batching_configs is not None:
            raise ValueError("accepted_review_network_combined_transition")
        previous, following = [read_provider_config(c) for c in network_configs]
        if (digest(previous), digest(following)) != (old.config_checksum, current.config_checksum):
            raise ValueError("accepted_review_network_config_binding")
        if not config_matches:
            route = getattr(following, "network_route", None)
            if previous.schema_version == following.schema_version and previous.schema_version in {
                "agent-provider-config/7.1.0",
                "agent-provider-config/7.2.0",
                "agent-provider-config/9.1.0",
                "agent-provider-config/9.2.0",
            }:
                from packages.llm_provider.config import with_replacement_network_route

                expected = with_replacement_network_route(previous, route)
            else:
                expected = with_network_route(previous, route)
            if route is None or expected != following:
                raise ValueError("accepted_review_network_config_delta")
            # Explicit original/current checksums. A routed 9.1/9.2 successor may
            # replace only the route; legacy direct-to-routed behavior remains.
            # Original context, prompts, source bindings, policy and accepted
            # request provenance are still checked below and by the reader.
            config_matches = True
    if batching_configs is not None or reasoned_batching_configs is not None:
        # Explicit legacy-local-judgment reuse, not a fresh integrated Review.
        # Full parent context and every medical field are checked below. Only
        # the declared 3 -> 1 partition change can differ; no model/prompt/route
        # or request limit is erased from either historical/current checksum.
        from packages.llm_provider.config import read_provider_config

        if batching_configs is not None and reasoned_batching_configs is not None:
            raise ValueError("accepted_review_batch_combined_transition")
        reasoned = reasoned_batching_configs is not None
        previous, following = [
            read_provider_config(c)
            for c in (reasoned_batching_configs if reasoned else batching_configs)
        ]
        if (digest(previous), digest(following)) != (old.config_checksum, current.config_checksum):
            raise ValueError("accepted_review_batch_config_binding")
        if not config_matches:
            before, after = previous.model_dump(mode="json"), following.model_dump(mode="json")
            if (
                previous.schema_version
                not in (
                    {"agent-provider-config/9.2.0"}
                    if reasoned
                    else {"agent-provider-config/7.0.0", "agent-provider-config/7.1.0"}
                )
                or following.schema_version != previous.schema_version
                or previous.review_batching.batch_size != 3
                or following.review_batching.batch_size != 1
            ):
                raise ValueError("accepted_review_batch_config_delta")
            before["review_batching"]["batch_size"] = 1
            if before != after:
                raise ValueError("accepted_review_batch_config_delta")
            config_matches = True
    a, b = old.material, current.material
    if (
        (a.scope.case_code, a.scope.input_checksum, a.scope.visible_checksum)
        != (b.scope.case_code, b.scope.input_checksum, b.scope.visible_checksum)
        or (a.goal_code, a.goal_summary, a.visibility, old.laterality)
        != (b.goal_code, b.goal_summary, b.visibility, current.laterality)
        or (old.prompt_checksum, old.policy.scale)
        != (current.prompt_checksum, current.policy.scale)
        or not config_matches
        or old.policy.aggregation_checksum != current.policy.aggregation_checksum
    ):
        raise ValueError("accepted_review_reuse_context_changed")
    observations = {o.reference.code: o for o in b.observations}
    sources = {s.reference.code: s for s in b.sources}
    if any(
        o.status == "obtained" and observations.get(o.reference.code) != o for o in a.observations
    ) or any(sources.get(s.reference.code) != s for s in a.sources):
        raise ValueError("accepted_review_reuse_visible_context_changed")
    previous = {p.id: p for p in old.policy.items}
    policies = {p.id: p for p in current.policy.items}
    resources = {r.code: r for r in current.resources}
    old_resources = {r.code: r for r in old.resources}
    ids = []
    for item in candidate.items:
        p = previous[item.assessment_id]
        if (item.status != "assessed" and not include_unassessed) or policies.get(p.id) != p:
            continue
        if any(
            resources.get(k) != old_resources.get(k) for path in p.dependency_paths for k in path
        ):
            raise ValueError("accepted_review_reuse_dependency_changed")
        ids.append(p.id)
    if not ids:
        raise ValueError("accepted_review_reuse_no_unchanged_judgment")
    return tuple(sorted(ids))


class AcceptedReviewReader:
    """Original accepted bundle and requests plus current source-bound owner state."""

    def __init__(
        self,
        *,
        documents,
        requests,
        state_reader,
        acceptance_key,
        ancestor_reader=None,
        lineage=(),
        network_configs=None,
        capacity_configs=None,
        client_configs=None,
    ):
        self.documents, self.requests = documents, requests
        self.state_reader, self.acceptance_key = state_reader, acceptance_key
        self.ancestor_reader, self.lineage = ancestor_reader, lineage
        self.network_configs = network_configs
        self.capacity_configs = capacity_configs
        self.client_configs = client_configs

    def current(self, state):
        from .accepted_read_scope import current

        return current(self, state)

    def _current(self, state):
        from .acceptance_models import read_receipt, read_records
        from .review_continuation import semantic_binding

        old = self.state_reader()
        docs, key = self.documents, self.acceptance_key
        if docs.run_id in self.lineage or len(self.lineage) > MAX_ACCEPTED_REVIEW_DEPTH:
            raise ValueError("accepted_review_reuse_cycle_or_depth")
        control = docs.sessions.read(docs.run_id)
        if (
            control is None
            or control.phase != "stopped"
            or control.pending is not None
            or old.material.scope.run_id != docs.run_id
            or state.material.scope.run_id == docs.run_id
        ):
            raise ValueError("accepted_review_reuse_origin_not_terminal")
        docs.validate_scope(old.material.scope)
        submitted, bundle = docs.read("submission:" + key), docs.read("accepted:" + key)
        if not submitted or not bundle or len(bundle) != 4:
            raise ValueError("accepted_review_reuse_origin_missing")
        sub = read_submission(submitted[0]["submission"])
        if type(sub) not in {
            ReasonedSubmission,
            ReusedAcceptedSubmission,
            ReusedAcceptedSubmissionV7,
            FormatContinuedSubmission,
            ReusedAcceptedSubmissionV9,
            ReusedAcceptedSubmissionV10,
            ContinuedReasonedSubmission,
            ReusedAcceptedSubmissionV14,
            ReusedAcceptedSubmissionV15,
            ReusedAcceptedSubmissionV16,
        }:
            raise ValueError("accepted_review_reuse_origin_version")
        records, receipt = read_records(bundle[0]), read_receipt(bundle[3])
        from packages.evidence_scoring.accepted import build_accepted

        if (
            receipt.status != "accepted"
            or receipt.submission_checksum != digest(sub)
            or receipt.records_checksum != digest(records)
            or receipt.scores_checksum != digest(bundle[2]["scores"])
            or receipt.graph
            != docs.reference(
                "accepted:" + key, 1, bundle[1], role="graph", version=receipt.graph.version
            )
            or records.scope != old.material.scope
            or records.candidate != sub.candidate
            or records.batch_results != sub.batches
            or bundle[2]["scores"] != build_accepted(old, sub.candidate)
            or not docs.consumed(old.material.scope.graph)
        ):
            raise ValueError("accepted_review_reuse_receipt_binding")
        if type(sub) is ReasonedSubmission:
            confirmed = validate_batch_receipts(old, sub, docs, self.requests)
        else:
            from .review_continuation import ConfirmedReviewReader, check_continued

            if self.ancestor_reader is None:
                raise ValueError("accepted_review_reuse_ancestor_owner_required")
            from .accepted_read_scope import ancestor as original_ancestor

            ancestor = original_ancestor(self, sub.historical.run_id, (*self.lineage, docs.run_id))
            expected = (
                ConfirmedReviewReader
                if isinstance(sub, ContinuedReasonedSubmission)
                else AcceptedReviewReader
            )
            if not isinstance(ancestor, expected):
                raise ValueError("accepted_review_reuse_ancestor_owner_required")
            confirmed = check_continued(
                old,
                sub,
                owner=SimpleNamespace(confirmed_review_reader=ancestor),
                documents=docs,
                requests=self.requests,
            )
        expected_records = self._validated_requests(old, sub, receipt, confirmed)
        events = self.requests.journal.read()
        reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
        if (
            len(reserved) != len(set(reserved))
            or set(reserved) != {r.request_id for r in expected_records}
            or tuple(self.requests.records()) != expected_records
        ):
            raise ValueError("accepted_review_reuse_unknown_request")
        ids = compatible_local_ids(
            old,
            state,
            sub.candidate,
            network_configs=self.network_configs,
            capacity_configs=self.capacity_configs,
            client_configs=self.client_configs,
        )
        cls = (
            AcceptedReviewOriginV16
            if type(sub) is ReusedAcceptedSubmissionV16
            else AcceptedReviewOriginV15
            if type(sub) is ReusedAcceptedSubmissionV15
            else AcceptedReviewOriginV14
            if type(sub) in {ContinuedReasonedSubmission, ReusedAcceptedSubmissionV14}
            else AcceptedReviewOriginV13
            if type(sub) is ReusedAcceptedSubmissionV10
            else AcceptedReviewOriginV12
            if type(sub) in {FormatContinuedSubmission, ReusedAcceptedSubmissionV9}
            else AcceptedReviewOrigin
            if type(sub) is ReasonedSubmission
            else AcceptedReviewOriginV11
        )
        return cls(
            run_id=docs.run_id,
            acceptance_key=key,
            acceptance_checksum=digest(bundle),
            state_checksum=digest(old),
            control_checksum=digest(control),
            journal_checksum=digest(events),
            semantic_checksum=semantic_binding(state),
            assessment_ids=ids,
            submission=sub,
        )

    def _validated_requests(self, old, sub, receipt, confirmed):
        docs = self.documents
        if tuple(self.requests.records()) != tuple(confirmed):
            from .recorded_finalization import verified_corrected_records

            confirmed, _ = verified_corrected_records(old, sub, docs, self.requests, confirmed)
        if any(not r.simulated for r in confirmed) or (
            isinstance(sub, ReusedAcceptedSubmissionV16)
            and not sub.batches
            and getattr(receipt, "permission_checksum", None) is not None
        ):
            from .live_permission import ResearchLiveGrant

            grant = docs.read("research-live-grant")
            g = validate(ResearchLiveGrant, grant[0]) if grant else None
            s = old.material.scope
            if (
                g is None
                or not self.requests.journal.durable
                or getattr(receipt, "permission_checksum", None) != digest(g)
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
                or any(r.deadline > g.deadline or r.api_requests != 1 for r in confirmed)
            ):
                raise ValueError("accepted_review_reuse_original_permission")
        return confirmed


class RecordedAcceptedReviewReader(AcceptedReviewReader):
    """Explicit historical local-grant route; normal live-grant reader unchanged."""

    def __init__(self, *, config, saved_proof, **kwargs):
        super().__init__(**kwargs)
        self.config, self.saved_proof = config, saved_proof

    def _validated_requests(self, old, sub, receipt, confirmed):
        from .recorded_finalization import RecordedBatchRequests, read_saved_acceptance

        ancestor = None
        if hasattr(sub, "historical"):
            if self.ancestor_reader is None:
                raise ValueError("accepted_review_reuse_ancestor_owner_required")
            from .accepted_read_scope import ancestor as original_ancestor

            ancestor = original_ancestor(
                self, sub.historical.run_id, (*self.lineage, self.documents.run_id)
            )
        requests = RecordedBatchRequests(self.requests.journal)
        # super().current() already obtained `old` from the original current-state
        # owner for this single call. Reuse that immutable value while checking
        # the historical receipt, then freshly re-read at the return boundary.
        # No cross-call cache, dispatch authority, or source/tool gate is removed.
        proof = read_saved_acceptance(
            owner=SimpleNamespace(current=lambda: old, confirmed_review_reader=ancestor),
            documents=self.documents,
            requests=requests,
            config=self.config,
        )
        from .accepted_read_scope import has_return_recheck

        if not has_return_recheck() and self.state_reader() != old:
            raise ValueError("accepted_review_reuse_current_state_changed")
        if (
            proof != self.saved_proof
            or proof["acceptance_key"] != self.acceptance_key
            or proof["grant_checksum"] != receipt.permission_checksum
            or proof["submission_checksum"] != digest(sub)
        ):
            raise ValueError("accepted_review_reuse_recorded_permission")
        return requests.records()
