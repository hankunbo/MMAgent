"""07 controlled submission/acceptance. Installed owners re-read every dependency.

The persistence port is RuntimeDocuments: all submissions, rejections, conflicts and
accepted successor bundles use the existing Runtime artifact transaction and Audit.
"""

from packages.grounding.models import digest, validate
from packages.llm_provider.models import ReviewProposal, SourceProposal, VisionProposal
from packages.planner_control.models import Reference

from .acceptance_models import (
    AcceptanceReceipt,
    AcceptanceReceiptV2,
    AcceptedInterpretation,
    AcceptedRecordsV21,
    AcceptedRecordsV22,
    AcceptedRecordsV23,
    AcceptedRecordsV24,
    DomainState,
    ResearchGraphValue,
    read_receipt,
    read_records,
)
from .review_batches import (
    BatchedSubmission,
    PartialBatchedSubmission,
    read_submission,
    request_for_item,
)


class ResearchAcceptance:
    def __init__(self, *, owner, documents, requests, live_permission=None):
        self.owner, self.documents, self.requests = owner, documents, requests
        self.live_permission = live_permission
        if live_permission is not None and (
            live_permission.documents is not documents or live_permission.budget is not requests
        ):
            raise ValueError("research_live_owner_binding")

    def _state(self):
        # Reader must validate source state/withdrawal, input bytes and the current graph.
        state = validate(DomainState, self.owner.current())
        self.documents.validate_scope(state.material.scope)
        return state

    def _check(self, submission, state):
        if self.documents.consumed(submission.scope.graph):
            raise ValueError("parent_graph_already_consumed")
        if submission.scope != state.material.scope:
            raise ValueError("stale_or_foreign_scope")
        if submission.candidate.scope_checksum != digest(submission.scope):
            raise ValueError("candidate_scope")
        if submission.observation_checksums != tuple(
            o.reference for o in state.material.observations
        ):
            raise ValueError("observation_version")
        if submission.source_checksums != tuple(s.reference for s in state.material.sources):
            raise ValueError("source_version")
        if (submission.prompt_checksum, submission.config_checksum) != (
            state.prompt_checksum,
            state.config_checksum,
        ):
            raise ValueError("prompt_or_config_version")
        if isinstance(submission, BatchedSubmission):
            return self._check_batches(submission, state)
        role = (
            "review"
            if isinstance(submission.candidate, ReviewProposal)
            else "source"
            if isinstance(submission.candidate, SourceProposal)
            else "vision"
        )
        matches = [r for r in self.requests.records() if r.request_id == submission.request_id]
        if len(matches) != 1:
            raise ValueError("request_receipt_missing")
        r = matches[0]
        from .scope_finalization import RecordedScopeSubmission, check_request

        if isinstance(submission, RecordedScopeSubmission):
            r = check_request(submission, state, self.owner, self.documents, self.requests)
        elif (
            r.scope,
            r.role,
            r.prompt_checksum,
            r.config_checksum,
            r.output_checksum,
            r.error,
        ) != (
            submission.scope,
            role,
            submission.prompt_checksum,
            submission.config_checksum,
            digest(submission.candidate),
            None,
        ):
            raise ValueError("request_receipt_mismatch")
        if not r.simulated:
            from .live_permission import require_live_permission

            require_live_permission(self.live_permission, state, role=role, record=r)
        c = submission.candidate
        if isinstance(c, ReviewProposal):
            from packages.evidence_scoring.accepted import build_accepted

            from .review_tasks import project_review_tasks, review_payload

            expected_input = review_payload(state.material, project_review_tasks(state), c.round)
            if r.input_checksum != digest(expected_input):
                raise ValueError("review_request_task_binding")
            return build_accepted(state, c)
        if isinstance(c, SourceProposal):
            from packages.graph_engine.research_candidates import validate_research_relation
            from packages.knowledge_claims.research_candidates import validate_research_source
            from packages.knowledge_claims.research_quote import quote_policy

            validate_research_source(c, state.material, quote_policy=quote_policy(self.owner))
            validate_research_relation(c, scope=submission.scope, node_codes=state.nodes)
        if isinstance(c, VisionProposal):
            from packages.llm_provider.input_scope import ScopeVisionProposal
            from packages.llm_provider.models import PartialVisionProposal

            if isinstance(c, ScopeVisionProposal):
                from .input_scope import check_submission

                check_submission(submission, state, self.owner, r)

            if (
                isinstance(c, PartialVisionProposal)
                and getattr(self.owner, "partial_observation_contract", None) != "1.0.0"
            ):
                raise ValueError("partial_vision_owner_not_installed")
            images = [o.image for o in state.material.observations if o.image is not None]
            if len(images) != 1 or (
                images[0].reference.checksum,
                images[0].preprocessing.checksum,
                images[0].location,
            ) != (c.image_checksum, c.preprocessing_checksum, c.location):
                raise ValueError("vision_binding")
            self.owner.verify_image(images[0])
        return None

    def _check_batches(self, submission, state):
        from packages.evidence_scoring.accepted import build_accepted

        from .accepted_aggregation import AggregatedSubmission, check_aggregation

        if isinstance(submission, AggregatedSubmission):
            from .accepted_aggregation import AggregationPermission
            from .live_permission import require_live_permission

            if type(self.live_permission) is AggregationPermission:
                # One current original-owner proof serves content and permission.
                # No cached result survives this submit/accept call.
                self.live_permission.check_submission(state, submission)
            else:
                check_aggregation(state, submission, self.owner)
                require_live_permission(self.live_permission, state, role="review")
            return build_accepted(state, submission.candidate)

        from .accepted_review_reuse import ReusedAcceptedSubmission, ReusedAcceptedSubmissionV7
        from .recorded_explanation import ExplanationSubmission, check_submission
        from .review_batches import validate_batch_receipts
        from .review_continuation import ContinuedReasonedSubmission, check_continued

        if self.live_permission is not None and submission.plan.policy != getattr(
            self.live_permission.config, "review_batching", None
        ):
            raise ValueError("batch_policy_not_installed")
        if isinstance(submission, ExplanationSubmission):
            records = check_submission(state, submission, self.documents, self.requests)
        elif isinstance(
            submission,
            (ContinuedReasonedSubmission, ReusedAcceptedSubmission, ReusedAcceptedSubmissionV7),
        ):
            records = check_continued(
                state,
                submission,
                owner=self.owner,
                documents=self.documents,
                requests=self.requests,
            )
        else:
            records = validate_batch_receipts(state, submission, self.documents, self.requests)
        for record in records:
            if not record.simulated:
                from .live_permission import require_live_permission

                require_live_permission(self.live_permission, state, role="review", record=record)
        if not submission.batches:
            from .accepted_review_reuse import all_reused_live
            from .live_permission import require_live_permission

            if hasattr(self.requests, "check_control"):
                self.requests.check_control()
            if all_reused_live(self.owner, state):
                require_live_permission(self.live_permission, state, role="review")
        return build_accepted(
            state,
            submission.candidate,
            reviewed_ids=tuple(i.assessment_id for i in submission.candidate.items)
            if isinstance(submission, PartialBatchedSubmission)
            else None,
        )

    def _receipt(self, sha, state, reason):
        from .accepted_aggregation import AggregationGrant, AggregationReceipt

        ref = state.material.scope.graph
        live = self.live_permission is not None
        local = isinstance(getattr(self.live_permission, "grant", None), AggregationGrant)
        return (
            AggregationReceipt if local else AcceptanceReceiptV2 if live else AcceptanceReceipt
        )(
            **(
                dict(
                    permission_checksum=digest(self.live_permission.grant), new_medical_review=False
                )
                if live
                else {}
            ),
            submission_checksum=sha,
            status="rejected",
            reason=reason,
            parent_graph=Reference(
                role=ref.role, code=ref.code, version=ref.version, checksum=ref.checksum
            ),
            graph=None,
            records_checksum=None,
            scores_checksum=None,
        )

    def _rejection(self, key, sha, state, reason):
        receipt = self._receipt(sha, state, reason)
        self.documents.write(key, (receipt,))
        return receipt

    def submit(self, value, *, key):
        try:
            return self._submit(value, key=key)
        except Exception:
            self.documents.failure(key, digest(value), "object_or_transaction_failure")
            raise

    def _submit(self, value, *, key):
        """Submission is recorded separately; successful submission is not acceptance."""
        with self.documents.session():
            state = self._state()
            sha = digest(value)
            prior = self.documents.read("submission:" + key)
            if prior is not None:
                if prior[0]["submission_checksum"] != sha:
                    return self._rejection(
                        "conflict:" + key + sha, sha, state, "idempotency_conflict"
                    )
                return (
                    read_receipt(prior[0]["receipt"])
                    if prior[0]["status"] == "rejected"
                    else prior[0]
                )
            try:
                submission = read_submission(value)
                self._check(submission, state)
                from .source_quotes import installed_source_proof

                proof = installed_source_proof(submission, state.material, self.owner)
            except (ValueError, TypeError):
                receipt = self._receipt(sha, state, "domain_submission_rejected")
                # A rejected key still binds the original content.
                self.documents.write(
                    "submission:" + key,
                    (
                        {
                            "submission_checksum": sha,
                            "status": "rejected",
                            "receipt": receipt.model_dump(mode="json"),
                        },
                    ),
                )
                return receipt
            result = dict(
                schema_version="research-submission-receipt/1.0.0",
                submission_checksum=sha,
                status="submitted_not_accepted",
                submission=submission.model_dump(mode="json"),
            )
            if proof is not None:
                result.update(
                    schema_version="research-submission-receipt/1.1.0",
                    source_quote_validation=proof,
                )
            self.documents.write("submission:" + key, (result,))
            return result

    def accept(self, *, key):
        try:
            return self._accept(key=key)
        except Exception:
            self.documents.failure(key, digest(key), "object_or_transaction_failure")
            raise

    def _accept(self, *, key):
        with self.documents.session():
            prior = self.documents.read("accepted:" + key)
            if prior is not None:
                return read_receipt(prior[-1])
            saved = self.documents.read("submission:" + key)
            if saved is None or saved[0]["status"] != "submitted_not_accepted":
                raise ValueError("submitted_candidate_required")
            submission = read_submission(saved[0]["submission"])
            state = self._state()
            try:
                scores = self._check(submission, state)
                from .source_quotes import installed_source_proof

                proof = installed_source_proof(submission, state.material, self.owner)
                expected_version = (
                    "research-submission-receipt/1.1.0"
                    if proof is not None
                    else "research-submission-receipt/1.0.0"
                )
                if (
                    saved[0].get("source_quote_validation") != proof
                    or saved[0].get("schema_version") != expected_version
                ):
                    raise ValueError("research_quote_proof_changed")
            except (ValueError, TypeError):
                return self._rejection(
                    "accepted:" + key, digest(submission), state, "domain_acceptance_rejected"
                )
            from .review_tasks import project_review_tasks

            c = submission.candidate
            from .accepted_aggregation import (
                AggregatedRecords,
                AggregatedSubmission,
                AggregationReceipt,
                LegacyAggregatedRecords,
                LegacyAggregatedSubmission,
                item_requests,
            )

            local_aggregation = isinstance(submission, AggregatedSubmission)
            invalidated = []
            if isinstance(c, VisionProposal):
                changed = {
                    o.reference.code
                    for o in state.material.observations
                    if o.image and o.image.reference.checksum == c.image_checksum
                }
                invalidated = [i.id for i in state.policy.items if i.observation_code in changed]
                from packages.llm_provider.input_scope import ScopeVisionProposal

                if isinstance(c, ScopeVisionProposal):
                    # A dedicated survey has no prior disease interpretation to
                    # invalidate and must never manufacture an assessment slot.
                    invalidated = []
            observations = ()
            interpretations = ()
            if isinstance(c, VisionProposal):
                original = next(
                    o
                    for o in state.material.observations
                    if o.image and o.image.reference.checksum == c.image_checksum
                )
                newref = Reference(
                    role="observation",
                    code=original.reference.code,
                    version="2.0.0",
                    checksum=digest(
                        dict(original=original, proposal=c, request_id=submission.request_id)
                    ),
                )
                observations = (
                    original.model_copy(
                        update={
                            "reference": newref,
                            "status": "obtained" if c.status == "observed" else c.status,
                            "value": c.observation_value()
                            if hasattr(c, "observation_value")
                            else c.finding,
                        }
                    ),
                )
            if isinstance(c, ReviewProposal):
                from .acceptance_models import ReasonedInterpretation, TaskPolicyV7

                interpretation_type = (
                    ReasonedInterpretation
                    if isinstance(state.policy, TaskPolicyV7)
                    else AcceptedInterpretation
                )
                policies = {p.id: p for p in state.policy.items}
                obs = {o.reference.code: o.reference for o in state.material.observations}
                sources = {s.reference.code: s.reference for s in state.material.sources}
                interpretations = tuple(
                    interpretation_type(
                        id="interpretation:"
                        + digest([request_for_item(submission, i.assessment_id), i])[7:],
                        observation_ref=obs[policies[i.assessment_id].observation_code],
                        source_refs=tuple(
                            sources[k] for k in policies[i.assessment_id].source_codes
                        ),
                        policy_checksum=submission.scope.policy_checksum,
                        review=i,
                        correlation_group=policies[i.assessment_id].correlation_group,
                        dependence_block=policies[i.assessment_id].dependence_block,
                        dependency_paths=policies[i.assessment_id].dependency_paths,
                        request_id=request_for_item(submission, i.assessment_id),
                    )
                    for i in c.items
                )
            records = dict(
                schema_version="case-observation-interpretation/2.1.0",
                observation_facts=state.material.observations,
                review_tasks=project_review_tasks(state) if isinstance(c, ReviewProposal) else None,
                scope=submission.scope.model_dump(mode="json"),
                parent_records="preserved_by_parent_graph",
                candidate=c.model_dump(mode="json"),
                observations=observations,
                interpretations=interpretations,
                invalidated_assessment_ids=tuple(invalidated),
                request_id=None
                if isinstance(submission, BatchedSubmission)
                else submission.request_id,
                status="accepted_research_fixture",
                clinical_acceptance=False,
            )
            all_reused = isinstance(submission, BatchedSubmission) and not submission.batches
            from .accepted_review_reuse import all_reused_live

            request = (
                None
                if all_reused and not local_aggregation
                else next(
                    r
                    for r in self.requests.records()
                    if r.request_id
                    == (
                        request_for_item(submission, c.items[0].assessment_id)
                        if local_aggregation
                        else submission.batches[0].request_id
                        if isinstance(submission, BatchedSubmission)
                        else submission.request_id
                    )
                )
            )
            live = local_aggregation or (
                all_reused_live(self.owner, state) if all_reused else not request.simulated
            )
            if live:
                records.update(
                    schema_version="case-observation-interpretation/2.2.0",
                    status="accepted_live_research",
                    permission_checksum=digest(self.live_permission.grant),
                )
            if isinstance(submission, BatchedSubmission):
                records.update(
                    schema_version="case-observation-interpretation/2.3.0",
                    request_ids=tuple(r.request_id for r in submission.batches),
                    batch_plan=submission.plan,
                    batch_results=submission.batches,
                    execution="bounded_live_research"
                    if live
                    else "offline_fixture_or_candidate_only",
                    permission_checksum=digest(self.live_permission.grant) if live else None,
                )
                if isinstance(submission, PartialBatchedSubmission):
                    records.update(
                        schema_version="case-observation-interpretation/2.4.0",
                        completion="partial",
                        pending_batch_ids=submission.pending_batch_ids,
                    )
                cls = (
                    AcceptedRecordsV24
                    if isinstance(submission, PartialBatchedSubmission)
                    else AcceptedRecordsV23
                )
                if getattr(state.policy, "direction_policy", None):
                    from .acceptance_models import AcceptedRecordsV25, AcceptedRecordsV26

                    cls = (
                        AcceptedRecordsV26
                        if isinstance(submission, PartialBatchedSubmission)
                        else AcceptedRecordsV25
                    )
                    records["schema_version"] = cls.model_fields["schema_version"].default
                from .accepted_review_reuse import (
                    ReusedAcceptedSubmission,
                    ReusedAcceptedSubmissionV7,
                    ReusedAcceptedSubmissionV9,
                    ReusedAcceptedSubmissionV10,
                    ReusedAcceptedSubmissionV14,
                    ReusedAcceptedSubmissionV15,
                    ReusedAcceptedSubmissionV16,
                    provenance_batches,
                )
                from .review_continuation import ContinuedReasonedSubmission
                from .review_format_continuation import FormatContinuedSubmission

                if isinstance(
                    submission,
                    (
                        ContinuedReasonedSubmission,
                        ReusedAcceptedSubmission,
                        ReusedAcceptedSubmissionV7,
                    ),
                ):
                    from .acceptance_models import (
                        AcceptedRecordsV27,
                        AcceptedRecordsV28,
                        AcceptedRecordsV29,
                        AcceptedRecordsV210,
                        AcceptedRecordsV211,
                        AcceptedRecordsV212,
                        AcceptedRecordsV216,
                        AcceptedRecordsV217,
                        AcceptedRecordsV218,
                    )

                    cls = (
                        AcceptedRecordsV218
                        if isinstance(submission, ReusedAcceptedSubmissionV16)
                        else AcceptedRecordsV217
                        if isinstance(submission, ReusedAcceptedSubmissionV15)
                        else AcceptedRecordsV216
                        if isinstance(submission, ReusedAcceptedSubmissionV14)
                        else AcceptedRecordsV212
                        if isinstance(submission, ReusedAcceptedSubmissionV10)
                        else AcceptedRecordsV211
                        if isinstance(submission, ReusedAcceptedSubmissionV9)
                        else AcceptedRecordsV210
                        if isinstance(submission, FormatContinuedSubmission)
                        else AcceptedRecordsV29
                        if isinstance(submission, ReusedAcceptedSubmissionV7)
                        else AcceptedRecordsV28
                        if isinstance(submission, ReusedAcceptedSubmission)
                        else AcceptedRecordsV27
                    )
                    records.update(
                        schema_version=cls.model_fields["schema_version"].default,
                        historical_review=submission.historical,
                        request_ids=tuple(
                            r.request_id
                            for r in (
                                provenance_batches(submission)
                                if isinstance(submission, ReusedAcceptedSubmissionV7)
                                else (
                                    *submission.historical.submission.batches,
                                    *submission.batches,
                                )
                            )
                        ),
                    )
                    if isinstance(submission, ReusedAcceptedSubmissionV16):
                        records["composition"] = submission.composition
                from .recorded_explanation import ExplanationAcceptedRecords, ExplanationSubmission

                if isinstance(submission, ExplanationSubmission):
                    cls = ExplanationAcceptedRecords
                    records.update(
                        schema_version=cls.model_fields["schema_version"].default,
                        local_interpretation=submission.local_interpretation,
                        original_record_checksum=submission.original_record_checksum,
                    )
                if local_aggregation:
                    from .accepted_aggregation import (
                        AliasAggregatedRecords,
                        AliasAggregatedSubmission,
                    )

                    cls = (
                        AliasAggregatedRecords
                        if isinstance(submission, AliasAggregatedSubmission)
                        else LegacyAggregatedRecords
                        if isinstance(submission, LegacyAggregatedSubmission)
                        else AggregatedRecords
                    )
                    records.update(
                        schema_version=cls.model_fields["schema_version"].default,
                        status="accepted_research_reaggregation",
                        execution="local_reaggregation_of_accepted_reviews",
                        origins=submission.origins,
                        request_ids=tuple(
                            dict.fromkeys(item_requests(submission.origins).values())
                        ),
                        new_model_requests=0,
                    )
                records = validate(cls, records)
            else:
                from packages.llm_provider.input_scope import ScopeVisionProposal

                if isinstance(c, ScopeVisionProposal):
                    from .input_scope import payload, records_type

                    cls = records_type(c, live=live)
                    from .scope_finalization import RecordedScopeSubmission, recorded_records_type

                    if isinstance(submission, RecordedScopeSubmission):
                        cls = recorded_records_type(submission)
                        records.update(
                            storage_interpretation=submission.storage_interpretation,
                            original_request_checksum=submission.original_request_checksum,
                        )
                    records.update(
                        schema_version=cls.model_fields["schema_version"].default,
                        scope_task=payload(self.owner, state).scope_task,
                    )
                else:
                    cls = AcceptedRecordsV22 if live else AcceptedRecordsV21
                records = validate(cls, records)
            graph = dict(
                schema_version="case-evidence-graph/2.0.0",
                scope=submission.scope.model_dump(mode="json"),
                parent_graph={
                    k: v
                    for k, v in submission.scope.graph.model_dump(mode="json").items()
                    if k not in {"kind", "parent"}
                },
                records_checksum=digest(records),
                scores_checksum=digest(scores) if scores is not None else None,
                nodes=state.nodes,
                invalidated_assessment_ids=tuple(invalidated),
                accepted_validations=[],
                clinical_patient_graph=False,
            )
            graph = validate(ResearchGraphValue, graph)
            gref = self.documents.reference(
                "accepted:" + key, 1, graph, role="graph", version=graph.schema_version
            )
            ref = submission.scope.graph
            receipt = (
                AggregationReceipt
                if local_aggregation
                else AcceptanceReceiptV2
                if live
                else AcceptanceReceipt
            )(
                **(
                    dict(
                        permission_checksum=digest(self.live_permission.grant),
                        new_medical_review=isinstance(c, ReviewProposal) and not local_aggregation,
                    )
                    if live
                    else {}
                ),
                submission_checksum=digest(submission),
                status="accepted",
                reason="accepted_local_reviews_reaggregated"
                if local_aggregation
                else "bounded_live_research_accepted"
                if live
                else "research_fixture_accepted",
                parent_graph=Reference(
                    role=ref.role, code=ref.code, version=ref.version, checksum=ref.checksum
                ),
                graph=gref,
                records_checksum=digest(records),
                scores_checksum=digest(scores) if scores is not None else None,
            )
            if local_aggregation:
                from .accepted_aggregation import AggregationPermission

                if type(self.live_permission) is AggregationPermission:
                    try:
                        self.live_permission.check_commit(state, submission)
                    except ValueError:
                        return self._rejection(
                            "accepted:" + key,
                            digest(submission),
                            state,
                            "domain_acceptance_rejected",
                        )
            # The receipt and new records/graph/scores are ONE original Runtime transaction.
            self.documents.write("accepted:" + key, (records, graph, {"scores": scores}, receipt))
            return receipt

    def read(self, *, key):
        """Immutable historical read; current validity requires revalidation."""
        return self.documents.read("accepted:" + key)

    def revalidate(self, *, key):
        saved = self.documents.read("submission:" + key)
        if saved is None:
            raise ValueError("submission_missing")
        return self._check(read_submission(saved[0]["submission"]), self._state())

    def successor_material(self, *, key):
        """Return an exact saved successor projection for explicit owner installation in 08D.

        Does not advance Planner or install state. The caller must still re-read every
        source/observation dependency through its owner before making another request.
        """
        from packages.llm_provider.models import Material
        from packages.planner_control.models import Graph

        with self.documents.session():
            saved = self.read(key=key)
            if saved is None or len(saved) != 4:
                raise ValueError("accepted_successor_required")
            records = read_records(saved[0])
            graph = validate(ResearchGraphValue, saved[1])
            receipt = read_receipt(saved[3])
            state = self._state()
            if isinstance(receipt, AcceptanceReceiptV2):
                from .live_permission import require_live_permission

                require_live_permission(self.live_permission, state)
                if receipt.permission_checksum != digest(self.live_permission.grant):
                    raise ValueError("research_live_grant_changed")
            if records.scope != state.material.scope:
                from .current_rejection import CurrentResearchRejected

                raise CurrentResearchRejected("research_parent_changed")
            if (
                receipt.status != "accepted"
                or receipt.graph is None
                or graph.scope != records.scope
                or receipt.records_checksum != digest(records)
                or graph.records_checksum != receipt.records_checksum
                or graph.scores_checksum != receipt.scores_checksum
                or (digest(saved[2]["scores"]) if saved[2]["scores"] is not None else None)
                != receipt.scores_checksum
                or receipt.graph
                != self.documents.reference(
                    "accepted:" + key, 1, graph, role="graph", version=graph.schema_version
                )
            ):
                raise ValueError("successor_binding")
            observations = {o.reference.code: o for o in state.material.observations}
            if any(o.reference.code not in observations for o in records.observations):
                raise ValueError("successor_observation_scope")
            observations.update({o.reference.code: o for o in records.observations})
            raw = state.material.model_dump(mode="json")
            raw["observations"] = [o.model_dump(mode="json") for o in observations.values()]
            raw["scope"]["graph"] = Graph(
                **receipt.graph.model_dump(), kind="research_artifact", parent=receipt.parent_graph
            ).model_dump(mode="json")
            raw["scope"].pop("material_checksum")
            raw["scope"]["material_checksum"] = digest(raw)
            result = validate(Material, raw)
            self.documents.validate_scope(result.scope)
            return result
