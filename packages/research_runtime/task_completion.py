"""Read-only proof derived from pre-execution binding and original 06/H owners."""

from datetime import UTC, datetime, timedelta

from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.grounding.models import canonical, digest, seal, validate
from packages.research_runtime.task_models import TaskCompletion, TaskDefinition
from packages.tool_execution.models import WorkflowResult


def require(condition, reason):
    if not condition:
        raise CurrentResearchRejected("task_completion_" + reason)


class ResearchTaskCompletion:
    """Installed by the single-case service; no caller-supplied results or completion flag.

    V1 has one mandatory tool, all planned outputs and all configured H result
    definitions mandatory. No optional-dependency semantics are inferred from outcomes.
    """

    def __init__(self, driver, *, goal, validity_seconds=86400, clock=None):
        self.driver, self.goal = driver, goal
        self.validity_seconds = validity_seconds
        self.clock = clock or (lambda: datetime.now(UTC))

    def definition(self, context, plan, executor, scores, key):
        require(context.synthetic and scores.synthetic, "synthetic_required")
        context.validate_plan(plan)
        require(len(plan.document.steps) == 1, "single_tool_required")
        execution = executor.task_requirements(plan)
        scoring = scores.task_requirements(plan, key)
        # Profile excludes case/run/input identities, while the definition binds them
        # via the original sealed request/context/plan. Exact versions remain in profile.
        profile = dict(
            goal=self.goal,
            use="synthetic_offline_planning_prior",
            purpose=plan.document.purpose,
            catalog=context.request.lock.catalog_ref,
            compatibility=context.request.compatibility_policy,
            candidate=context.request.candidate,
            invocation=context.request.invocation,
            outputs=execution["outputs"],
            scoring=scoring["outputs"],
            models=scoring["models"],
        )
        return seal(
            TaskDefinition,
            goal=self.goal,
            profile=digest(profile),
            run_id=context.run_id,
            case_code=context.request.scope.case_code,
            context_checksum=context.checksum,
            request_checksum=context.request.checksum,
            scope_checksum=context.scope_checksum,
            plan_checksum=plan.checksum,
            config_checksum=self.driver.config_checksum,
            execution_policy=execution["policy"],
            score_policy=scoring["policy"],
            requirements=dict(outputs=execution["outputs"], scores=scoring["outputs"]),
            validity_seconds=self.validity_seconds,
        )

    def read(self, run_id):
        try:
            return self._read(run_id)
        except ValueError as error:
            if type(error) is ValueError and str(error) in {
                "research_registration_mismatch",
                "research_case_scope_mismatch",
                "research_input_fact_mismatch",
                "research_synthetic_mismatch",
                "score_registry_drift",
                "score_saved_facts_not_current",
                "score_artifact_bytes",
                "score_artifact_scope_or_fact",
                "score_object_checksum",
                "score_not_saved",
                "task_completion_score_profile",
            }:
                raise CurrentResearchRejected("task_completion_source_changed") from error
            raise

    def _read(self, run_id):
        d = self.driver
        state = d.runtime.lifecycle(run_id)
        require(state is not None and "task_definition" in state["binding"], "not_bound")
        definition = validate(TaskDefinition, state["binding"]["task_definition"])
        context = d.research.read(run_id)
        choice = d.choice(context.request) if callable(d.choice) else d.choice
        plan = d.research.compile(run_id, choice)
        executor, scores = d.bind_plan(plan, state["binding"]["execution_key"])
        expected = self.definition(
            context, plan, executor, scores, state["binding"]["execution_key"]
        )
        require(definition == expected, "definition_changed")
        require(
            state["phase"] == "FINISHED"
            and d.runtime.run_status(run_id) == "COMPLETED_INSUFFICIENT_EVIDENCE",
            "not_finished",
        )
        start = datetime.fromisoformat(state["started_at"])
        expiry = start + timedelta(seconds=definition.validity_seconds)
        require(start <= self.clock() < expiry, "expired")
        report = state["report"]
        require(
            report is not None
            and report["terminal"]
            and report["run_id"] == run_id
            and report["run_status"] == "COMPLETED_INSUFFICIENT_EVIDENCE"
            and report["technical_status"] == "SUCCEEDED"
            and report["execution_key"] == state["binding"]["execution_key"]
            and report["score_key"] == state["binding"]["score_key"]
            and report["plan_checksum"] == definition.plan_checksum
            and report["config_checksum"] == definition.config_checksum
            and report["policy_checksum"] == definition.score_policy
            and report["checksum"] == digest({k: v for k, v in report.items() if k != "checksum"}),
            "report_binding",
        )
        attempt = executor.recover(run_id, state["binding"]["execution_key"])
        require(
            attempt is not None and attempt["status"] == "CONFIRMED" and attempt["result"],
            "execution_unconfirmed",
        )
        workflow = executor.validate_evidence_source(
            plan, state["binding"]["execution_key"], validate(WorkflowResult, attempt["result"])
        )
        require(workflow.status == "SUCCEEDED" and not workflow.unstarted_steps, "tool_incomplete")
        # This is a separate completeness condition: SUCCEEDED alone cannot authorize.
        step, planned = workflow.steps[0], plan.document.steps[0]
        require(
            len(step.result.artifacts) == len(step.qc_records) == len(planned.outputs) > 0,
            "output_set",
        )
        registered = {a.asset_code: a for a in d.runtime.artifacts(definition.case_code, run_id)}
        output_hashes = []
        for output, artifact, qc in zip(
            planned.outputs, step.result.artifacts, step.qc_records, strict=True
        ):
            require(
                qc.status == "PASS"
                and artifact.qc_status == "PASS"
                and qc.logical_uri == output.logical_uri
                and qc.qc_policy_ref == output.qc_policy_ref
                and registered.get(artifact.asset_code) == artifact,
                "output_qc_or_registration",
            )
            # Validate every required output's current bytes, including outputs without
            # numeric scores. Missing objects/I/O errors are not silently treated as empty.
            raw = scores.objects.read_bounded(artifact.uri, 4 * 1024 * 1024)
            from packages.object_store import checksum

            require(checksum(raw) == artifact.checksum, "output_drift")
            output_hashes.append(digest(dict(artifact=artifact, qc=qc)))
        ledger = scores.read(run_id, state["binding"]["score_key"])
        require(
            ledger.checksum == report["score_checksum"]
            and ledger.policy_checksum == definition.score_policy
            and ledger.scope.run_id == run_id
            and ledger.scope.case_code == definition.case_code
            and ledger.synthetic
            and len(ledger.technical) == 1
            and ledger.technical[0].status == "SUCCEEDED"
            and ledger.technical[0].attempt_id == workflow.attempt_id,
            "score_binding",
        )
        for artifact_type, score_definition, _decoder, _field in scores.observations[0].outputs:
            artifacts = [a for a in step.result.artifacts if a.artifact_type_code == artifact_type]
            require(bool(artifacts), "required_score_output")
            for artifact in artifacts:
                entries = [
                    e
                    for e in ledger.entries
                    if e.definition == score_definition
                    and e.source.artifact_code == artifact.asset_code
                ]
                require(
                    len(entries) == 1
                    and entries[0].status == "available"
                    and entries[0].source.plan_checksum == plan.checksum
                    and entries[0].source.attempt_id == workflow.attempt_id,
                    "required_score_unavailable",
                )
        from .single_case import verified_model_outputs

        entries = [
            e.model_dump(mode="json") for e in ledger.entries if e.definition.layer == "model"
        ]
        outputs = verified_model_outputs(ledger, scores.objects)
        # New exact-storage reports must retain the original H bytes, including
        # signed zero. Legacy lifecycle keys retain their original comparison.
        if state["version"] == "research-single-case/1.1.0":
            unchanged = canonical(report["model_entries"]) == canonical(entries) and canonical(
                report["model_outputs"]
            ) == canonical(outputs)
        else:
            unchanged = report["model_entries"] == entries and report["model_outputs"] == outputs
        require(unchanged, "saved_results_changed")
        require(d.runtime.lifecycle(run_id) == state, "read_changed")
        return seal(
            TaskCompletion,
            definition=definition,
            lifecycle_checksum=digest(state),
            report_checksum=digest(report),
            workflow_checksum=digest(workflow),
            score_checksum=ledger.checksum,
            output_checksums=tuple(output_hashes),
            valid_from=start,
            expires_at=expiry,
        )
