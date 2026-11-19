"""Application-owned admission and execution. No caller-supplied approval or readers."""

import json
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from threading import Event
from uuid import uuid4

from packages.contracts import Artifact, QCStatus, ToolExecutionResult
from packages.contracts import ToolExecutionStatus as Status
from packages.grounding.models import digest, validate
from packages.object_store import checksum
from packages.tool_workflow import ExecutionPlan
from packages.tool_workflow.compiler import local_path

from .models import (
    STEP_RESULT_MAX_BYTES,
    ExecutionDecision,
    ExecutionError,
    ExecutionPolicy,
    StepResult,
    WorkflowResult,
    derive_workflow_status,
    implementation_identity,
    validate_step_binding,
    validate_workflow_plan,
)
from .process import LinuxProcessRunner, read_file, write_copy
from .qc import QCObservation


class ToolExecutionService:
    def __init__(self, *, compiler, catalogs, objects, qc, repository, policy, workspace):
        self._compiler, self._catalogs, self._objects, self._qc, self._repo = (
            compiler,
            catalogs,
            objects,
            qc,
            repository,
        )
        self._policy = validate(ExecutionPolicy, policy)
        self._runner = LinuxProcessRunner(workspace, self._policy)

    def _validate(self, plan):
        if type(plan) is not ExecutionPlan:
            raise ExecutionError("complete_plan_bundle_required")
        self._compiler.validate_plan(plan)
        doc = plan.document
        if doc.config.network != "deny":
            raise ExecutionError("cpu_backend_requires_deny_network_config")
        if (
            doc.checksum not in self._policy.allowed_plan_checksums
            or doc.purpose != self._policy.purpose
        ):
            raise ExecutionError("execution_not_admitted")
        if doc.config.max_output_bytes > 4194304:
            raise ExecutionError("object_port_output_size_unsupported")
        for step in doc.steps:
            if step.deployment.allowlist.synthetic_only and not self._policy.synthetic_execution:
                raise ExecutionError("synthetic_execution_not_registered")
            if not step.deployment.allowlist.synthetic_only and self._policy.synthetic_execution:
                raise ExecutionError("real_model_not_synthetic_fixture")
            for output in step.outputs:
                self._qc.policy_for(output)
        self._runner.check_capabilities()
        return doc

    def execute_bundle(self, bundle_json, idempotency_key, *, cancel=None):
        return self.execute(
            ExecutionPlan.from_bundle_json(bundle_json), idempotency_key, cancel=cancel
        )

    def execution_result(self, plan, idempotency_key, returned):
        """Adapt this backend's synchronous return using its owned durable attempt.

        No admission or process is restarted here. A return value alone is never
        confirmation authority. Trusted backends override only envelope dispatch.
        """
        return self.confirmed_result(plan, idempotency_key, returned)

    def _return_record(self, plan, idempotency_key):
        if type(plan) is not ExecutionPlan:
            raise ExecutionError("complete_plan_bundle_required")
        doc = plan.document
        policy_hash = digest(self._policy)
        if (
            doc.checksum not in self._policy.allowed_plan_checksums
            or doc.purpose != self._policy.purpose
        ):
            raise ExecutionError("return_plan_policy_mismatch")
        record = self.recover(doc.run_id, idempotency_key)
        if record is None:
            return None
        request_hash = digest(
            dict(
                plan=doc.checksum,
                scope=doc.scope,
                run_id=doc.run_id,
                purpose=doc.purpose,
                policy=policy_hash,
            )
        )
        decision = validate(ExecutionDecision, record["decision"])
        if (
            record["run_id"] != doc.run_id
            or record["case_code"] != doc.scope.case_code
            or record["idempotency_key"] != idempotency_key
            or record["plan_checksum"] != doc.checksum
            or record["request_hash"] != request_hash
            or decision.attempt_id != record["attempt_id"]
            or decision.execution_binding_checksum
            != digest(dict(attempt=record["attempt_id"], request=request_hash))
        ):
            raise ExecutionError("return_attempt_binding_mismatch")
        return record

    def confirmed_result(self, plan, idempotency_key, returned):
        """Read/validate a confirmed result, including after a lost execute reply."""
        if type(returned) is not WorkflowResult:
            raise ExecutionError("workflow_return_type_invalid")
        record = self._return_record(plan, idempotency_key)
        if record is None or record["status"] != "CONFIRMED" or record["result"] is None:
            raise ExecutionError("return_attempt_unconfirmed")
        workflow = validate_workflow_plan(returned, plan.document, record["decision"])
        if workflow != validate_workflow_plan(record["result"], plan.document, record["decision"]):
            raise ExecutionError("return_result_source_mismatch")
        return workflow

    def task_requirements(self, plan):
        """Read installed policy and required QC definitions; never reserve or launch."""
        self._compiler.validate_plan(plan)
        return self._requirements(plan)

    def consumption_requirements(self, plan):
        """Current QC/policy for saved artifacts; not a launch authorization."""
        self._compiler.validate_consumed_plan(plan)
        return self._requirements(plan)

    def _requirements(self, plan):
        return dict(
            policy=digest(self._policy),
            outputs=[
                dict(
                    port=o.port,
                    constraint=o.constraint.model_dump(mode="json"),
                    qc=o.qc_policy_ref.model_dump(mode="json"),
                    policy=digest(self._qc.policy_for(o)),
                )
                for step in plan.document.steps
                for o in step.outputs
            ],
        )

    def validate_evidence_source(self, plan, idempotency_key, returned):
        """Read-only 09A handoff: confirmed facts plus current original admission.

        No reservation, recovery commit, runner capability probe or process launch.
        Compiler/owner rejections and unknown storage errors retain their types.
        """
        workflow = self.confirmed_result(plan, idempotency_key, returned)
        self._compiler.validate_plan(plan)
        return workflow

    def execute(self, plan, idempotency_key, *, cancel=None):
        doc = self._validate(plan)
        if type(idempotency_key) is not str or not 1 <= len(idempotency_key) <= 128:
            raise ExecutionError("idempotency_key_invalid")
        policy_hash = digest(self._policy)
        request_hash = digest(
            dict(
                plan=doc.checksum,
                scope=doc.scope,
                run_id=doc.run_id,
                purpose=doc.purpose,
                policy=policy_hash,
            )
        )
        aid = uuid4().hex
        decision = ExecutionDecision(
            schema_version="execution-decision/1.1.0",
            planned_step_ids=tuple(s.step_id for s in doc.steps),
            attempt_id=aid,
            plan_checksum=doc.checksum,
            scope_checksum=digest(doc.scope),
            purpose=doc.purpose,
            policy_checksum=policy_hash,
            execution_binding_checksum=digest(dict(attempt=aid, request=request_hash)),
        )
        # Slot is shared by all service instances at this controlled workspace.
        with self._runner.slot():
            reservation = self._repo.reserve(
                dict(
                    attempt_id=aid,
                    run_id=doc.run_id,
                    case_code=doc.scope.case_code,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    plan_checksum=doc.checksum,
                    decision=decision.model_dump(mode="json"),
                ),
                doc.steps,
            )
            attempt = reservation["attempt"]
            if not reservation["created"]:
                if attempt["decision"]["policy_checksum"] != policy_hash:
                    raise ExecutionError("recovery_policy_mismatch")
                if attempt["result"] is not None:
                    return validate_workflow_plan(attempt["result"], doc, attempt["decision"])
                raise ExecutionError("attempt_unconfirmed_recovery_required")
            results, products = [], {}
            cancellation = cancel if cancel is not None else Event()
            for step in doc.steps:
                result = self._step(plan, step, attempt, products, cancellation)
                results.append(result)
                if result.result.status != Status.SUCCEEDED:
                    break
                for output, artifact in zip(step.outputs, result.result.artifacts, strict=True):
                    products[output.logical_uri] = artifact
                # PENDING/FAIL also stop the workflow when a downstream step needs PASS.
            unstarted = tuple(s.step_id for s in doc.steps[len(results) :])
            workflow = WorkflowResult(
                attempt_id=aid,
                plan_checksum=doc.checksum,
                decision=decision,
                status=derive_workflow_status(results, unstarted),
                steps=tuple(results),
                unstarted_steps=unstarted,
            )
            self._journal(
                self._runner.workspace / aid / "workflow.json", workflow.model_dump(mode="json")
            )
            workflow = validate_workflow_plan(workflow, doc, attempt["decision"])
            return self._repo.confirm(workflow)

    def _step(self, plan, step, attempt, products, cancel):
        doc, started = plan.document, time.monotonic()
        states = [Status.REQUESTED]
        mapping, copied, inputs, outputs, qc_records = {}, {}, [], [], []
        captured = dict(exit_code=None, stdout=b"", stderr=b"")
        base = self._runner.workspace / attempt["attempt_id"] / step.step_id
        error, terminal = None, None

        def advance(target, **kwargs):
            self._repo.advance(attempt, step.step_id, states[-1], target, **kwargs)
            states.append(target)

        try:
            if cancel.is_set():
                raise ExecutionError("cancelled")
            records = {r.artifact.asset_code: r.artifact for r in step.deployment.input_records}

            def selected(source, uri):
                return (
                    records.get(source.asset_code)
                    if source.kind == "artifact"
                    else products.get(uri)
                )

            for bound in step.inputs:
                for source, uri in zip(bound.sources, bound.logical_uris, strict=True):
                    artifact = selected(source, uri)
                    if artifact is None or artifact.qc_status not in bound.constraint.qc_required:
                        raise ExecutionError("dependency_qc_not_satisfied")
                    if (
                        artifact.scope.case_code != doc.scope.case_code
                        or artifact.scope.run_id not in (None, doc.run_id)
                    ):
                        raise ExecutionError("dependency_scope_invalid")
                    inputs.append(artifact)
            base, root, copied, mapping, argv = self._materialize(attempt, step, doc, selected)
            binding = dict(
                plan=doc.checksum,
                decision=attempt["decision"],
                step=step.step_id,
                mapping=mapping,
                copied_checksums=copied,
                argv=argv,
            )
            self._journal(base / "binding.json", binding)
            # The backend owns resource materialization. Current safety/probe/locks
            # are rechecked immediately before the launch state changes.
            self._validate(plan)
            advance(Status.DRY_RUN_VALIDATED)
            advance(Status.QUEUED)
            if cancel.is_set():
                raise ExecutionError("cancelled")
            captured = self._runner.run(
                base,
                root,
                argv,
                doc.config,
                step.timeout_seconds,
                cancel,
                lambda proc: (
                    self._journal(base / "process.json", proc),
                    advance(Status.RUNNING, process_identity=proc),
                ),
            )
            self._journal(
                base / "process-cleanup.json",
                {"group_terminated": True, "exit_code": captured["exit_code"]},
            )
            if captured["error"]:
                raise ExecutionError(captured["error"])
            rules = {r.port: r for r in step.deployment.profile.output_rules}
            for output in step.outputs:
                data = read_file(
                    mapping[output.logical_uri], min(doc.config.max_output_bytes, 4194304)
                )
                if not data:
                    raise ExecutionError("output_empty")
                observation = self._qc.evaluate(output, rules[output.port], data, inputs)
                if not isinstance(observation, QCObservation) or not isinstance(
                    observation.status, QCStatus
                ):
                    raise ExecutionError("qc_observation_invalid")
                qc_policy = self._qc.policy_for(output)
                if (
                    observation.policy_checksum != output.qc_policy_ref.checksum
                    or observation.implementation != qc_policy.implementation
                ):
                    raise ExecutionError("qc_implementation_binding_mismatch")
                if observation.content_checksum != checksum(data):
                    raise ExecutionError("qc_content_binding_mismatch")
                qc_records.append(
                    {
                        **asdict(observation),
                        "status": observation.status,
                        "logical_uri": output.logical_uri,
                        "qc_policy_ref": output.qc_policy_ref,
                        "initial_qc_status": QCStatus.PENDING.value,
                    }
                )
                qc_policy = self._qc.policy_for(output)
                # Objects promoted and read back before any successful URI reaches SQL.
                staged = self._objects.stage(data)
                self._append_object(base, staged)
                uri = self._objects.promote(staged, checksum(data))
                self._append_object(base, uri)
                if checksum(self._objects.read_bounded(uri, 4194304)) != checksum(data):
                    raise ExecutionError("object_verification_failed")
                artifact = validate(
                    Artifact,
                    dict(
                        **{
                            **self._common(doc, step, inputs),
                            "laterality": output.constraint.laterality,
                            "coordinate_space": output.constraint.coordinate_space,
                            "unit": qc_policy.unit,
                        },
                        contract_type="Artifact",
                        asset_code="ASSET:OID_" + uuid4().hex.upper(),
                        artifact_type_code=qc_policy.artifact_type_code,
                        uri=uri,
                        checksum=checksum(data),
                        media_type=observation.media_type,
                        dimensions=observation.dimensions,
                        qc_status=observation.status,
                        derived_from=sorted({a.asset_code for a in inputs}),
                        research_only=True,
                    ),
                )
                outputs.append(artifact)
            terminal = Status.SUCCEEDED
        except (ExecutionError, OSError, ValueError) as exc:
            if isinstance(exc, ExecutionError) and str(exc) == "process_cleanup_unconfirmed":
                raise RuntimeError("attempt_outcome_uncertain_cleanup_required") from exc
            # Do not expose paths, credentials or arbitrary runner text in contracts.
            error = str(exc) if isinstance(exc, ExecutionError) else "execution_validation_failed"
            if error == "cancelled":
                terminal = Status.CANCELLED
            elif states[-1] == Status.RUNNING:
                terminal = Status.TIMED_OUT if error == "timeout" else Status.FAILED
            elif states[-1] == Status.REQUESTED:
                terminal = Status.BLOCKED
            else:
                terminal = Status.CANCELLED
            outputs = []
        catalog = self._catalogs.read_snapshot(doc.catalog_ref)
        tool = next(t for t in catalog.tools if t.implementation_ref == step.implementation_ref)
        qc = QCStatus.UNASSESSED
        if outputs:
            statuses = [a.qc_status for a in outputs]
            qc = (
                QCStatus.FAIL
                if QCStatus.FAIL in statuses
                else QCStatus.PASS
                if all(s == QCStatus.PASS for s in statuses)
                else QCStatus.UNASSESSED
                if all(s == QCStatus.UNASSESSED for s in statuses)
                else QCStatus.PENDING
            )
        result = validate(
            ToolExecutionResult,
            dict(
                **self._common(doc, step, inputs),
                contract_type="ToolExecutionResult",
                request_id="TASK:OID_"
                + digest(dict(attempt=attempt["attempt_id"], step=step.step_id))[7:55].upper(),
                workflow_run_id=doc.run_id,
                status=terminal,
                execution_policy="research",
                implementation_code=implementation_identity(step.implementation_ref),
                implementation_version=step.implementation_ref.version,
                implementation_checksum=step.implementation_ref.checksum,
                catalog_status=tool.catalog_status,
                validation_status=tool.validation_status,
                intended_use=tool.intended_use,
                decisive_evidence=False,
                qc_status=qc,
                artifacts=outputs,
                error=dict(code=error, message=error, retryable=False) if error else None,
            ),
        )
        record = StepResult(
            attempt_id=attempt["attempt_id"],
            step_id=step.step_id,
            implementation_ref=step.implementation_ref,
            plan_checksum=doc.checksum,
            binding_checksum=digest(
                dict(mapping=mapping, copies=copied, decision=attempt["decision"])
            ),
            logical_to_physical=mapping,
            copied_checksums=copied,
            status_history=tuple([*states, terminal]),
            elapsed_seconds=time.monotonic() - started,
            exit_code=captured["exit_code"],
            error_code=error,
            stdout_checksum=checksum(captured["stdout"]),
            stderr_checksum=checksum(captured["stderr"]),
            qc_records=tuple(qc_records),
            result=result,
        )
        base.mkdir(parents=True, exist_ok=True)
        self._journal(base / "result.json", record.model_dump(mode="json"))
        # SQL/transport exceptions propagate: no claimed failure or object deletion
        # after an ambiguous commit. Recovery queries the recorded attempt first.
        advance(terminal, record=record)
        return record

    def _materialize(self, attempt, step, doc, selected):
        mapping = {}
        base, root, copied = self._runner.prepare(attempt["attempt_id"], step, doc.config)
        argv_bindings = {}
        for resource in step.deployment.allowlist.resources:
            if not resource.sha256 or not resource.required:
                raise ExecutionError("resource_binding_incomplete")
            source = local_path(resource.uri)
            data = read_file(source, self._policy.max_resource_bytes)
            if checksum(data) != "sha256:" + resource.sha256 or len(data) != resource.byte_size:
                raise ExecutionError("resource_checksum_mismatch")
            write_copy(root, source, data)
            copied[resource.uri] = checksum(data)
            argv_bindings[source] = source
            mapping[resource.uri] = str(root / source.lstrip("/"))
        executable = local_path(step.deployment.allowlist.executable_uri)
        if copied.get(executable) != step.deployment.allowlist.executable_checksum:
            raise ExecutionError("executable_runtime_closure_mismatch")
        index = 0
        for bound in step.inputs:
            for source, uri in zip(bound.sources, bound.logical_uris, strict=True):
                artifact = selected(source, uri)
                data = self._objects.read_bounded(artifact.uri, 4194304)
                if checksum(data) != artifact.checksum:
                    raise ExecutionError("input_checksum_mismatch")
                child_path = "/in/" + str(index)
                index += 1
                physical = write_copy(root, child_path, data)
                mapping[uri], copied[uri], argv_bindings[uri] = (
                    str(physical),
                    checksum(data),
                    child_path,
                )
        for output in step.outputs:
            filename = output.logical_uri.rsplit("/", 1)[1]
            if filename in (".", "..") or "/" in filename:
                raise ExecutionError("output_path_invalid")
            child_path = "/out/" + filename
            physical = root / "out" / filename
            if physical.exists() or physical.is_symlink():
                raise ExecutionError("output_already_exists")
            mapping[output.logical_uri] = str(physical)
            argv_bindings[output.logical_uri] = child_path
        argv = [argv_bindings.get(token, token) for token in step.argv]
        if any(token.startswith(("object://", "plan-output://")) for token in argv):
            raise ExecutionError("logical_token_unbound")
        return base, root, copied, mapping, argv

    @staticmethod
    def _common(doc, step, inputs):
        return dict(
            schema_version="1.0",
            scope=dict(scope_type="RUN", case_code=doc.scope.case_code, run_id=doc.run_id),
            laterality=doc.scope.laterality,
            coordinate_space="NOT_APPLICABLE",
            unit=None,
            provenance=dict(
                actor="TOOL_AGENT",
                created_at=datetime.now(UTC),
                generator_version="06D.1.0.0",
                source_codes=["SOURCE:TOOL_EXECUTION"],
                input_checksums=sorted({a.checksum for a in inputs}),
                tool_versions=[
                    dict(
                        code=implementation_identity(step.implementation_ref),
                        version=step.implementation_ref.version,
                        checksum=step.implementation_ref.checksum,
                    )
                ],
            ),
        )

    @staticmethod
    def _journal(path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _append_object(self, base, uri):
        self._journal(base / ("object-" + uuid4().hex + ".json"), dict(uri=uri))

    def recover(self, run_id, idempotency_key):
        """Read durable facts. ACTIVE means uncertain, never implicitly re-executable."""
        record = self._repo.recover(run_id, idempotency_key)
        if (
            record is not None
            and record["plan_checksum"] not in self._policy.allowed_plan_checksums
        ):
            raise ExecutionError("recovery_scope_not_admitted")
        if record is not None:
            decision = validate(ExecutionDecision, record["decision"])
            if decision.policy_checksum != digest(self._policy):
                raise ExecutionError("recovery_policy_mismatch")
            if record["result"] is not None:
                workflow = validate(WorkflowResult, record["result"])
                if workflow.decision != decision or workflow.attempt_id != record["attempt_id"]:
                    raise ExecutionError("recovery_result_binding_mismatch")
        return record

    def recover_commit(self, plan, idempotency_key):
        """Re-submit owned durable results only; never launch or replay a subprocess."""
        doc = self._validate(plan)
        with self._runner.slot():
            attempt = self._repo.recover(doc.run_id, idempotency_key)
            if attempt is None or attempt["plan_checksum"] != doc.checksum:
                raise ExecutionError("recovery_attempt_mismatch")
            if attempt["decision"]["policy_checksum"] != digest(self._policy):
                raise ExecutionError("recovery_policy_mismatch")
            if attempt["result"] is not None:
                return validate_workflow_plan(attempt["result"], doc, attempt["decision"])
            decision = validate(ExecutionDecision, attempt["decision"])
            records, pending = [], []
            persisted = {s["step_id"]: s for s in attempt["steps"]}
            if set(persisted) != {s.step_id for s in doc.steps}:
                raise ExecutionError("recovery_step_set_mismatch")
            for step in doc.steps:
                row = persisted[step.step_id]
                base = self._runner.workspace / attempt["attempt_id"] / step.step_id
                path = base / "result.json"
                if not path.exists():
                    if row["status"] != "REQUESTED":
                        raise ExecutionError("execution_outcome_unknown_no_replay")
                    break
                record = validate(StepResult, json.loads(read_file(path, STEP_RESULT_MAX_BYTES)))
                validate_step_binding(record, decision)
                if (
                    record.step_id != step.step_id
                    or record.implementation_ref != step.implementation_ref
                ):
                    raise ExecutionError("recovery_record_binding_mismatch")
                if Status.RUNNING in record.status_history:
                    cleanup = json.loads(read_file(base / "process-cleanup.json", 1024))
                    if cleanup != {"group_terminated": True, "exit_code": record.exit_code}:
                        raise ExecutionError("process_cleanup_unconfirmed")
                for artifact in record.result.artifacts:
                    if (
                        checksum(self._objects.read_bounded(artifact.uri, 4194304))
                        != artifact.checksum
                    ):
                        raise ExecutionError("recovery_object_missing_or_changed")
                if row["result"] is None:
                    if row["status"] != record.status_history[-2].value:
                        raise ExecutionError("recovery_state_conflict")
                    pending.append(record)
                elif validate(StepResult, row["result"]) != record:
                    raise ExecutionError("recovery_result_conflict")
                records.append(record)
            if not records:
                raise ExecutionError("execution_outcome_unknown_no_replay")
            unstarted = tuple(s.step_id for s in doc.steps[len(records) :])
            if any(
                persisted[s]["status"] != "REQUESTED" or persisted[s]["result"] is not None
                for s in unstarted
            ):
                raise ExecutionError("recovery_unstarted_state_mismatch")
            path = self._runner.workspace / attempt["attempt_id"] / "workflow.json"
            if path.exists():
                workflow = validate(WorkflowResult, json.loads(read_file(path, 4194304)))
            else:
                workflow = WorkflowResult(
                    attempt_id=attempt["attempt_id"],
                    plan_checksum=doc.checksum,
                    decision=decision,
                    status=derive_workflow_status(records, unstarted),
                    steps=tuple(records),
                    unstarted_steps=unstarted,
                )
            workflow = validate_workflow_plan(workflow, doc, decision)
            if workflow.steps != tuple(records) or workflow.unstarted_steps != unstarted:
                raise ExecutionError("recovery_workflow_record_mismatch")
            # Validate the entire journal before any recovery write; never rerun a child.
            for record in pending:
                self._repo.advance(
                    attempt,
                    record.step_id,
                    record.status_history[-2],
                    record.status_history[-1],
                    record=record,
                )
            if not path.exists():
                self._journal(path, workflow.model_dump(mode="json"))
            return self._repo.confirm(workflow)

    def cleanup_unreferenced(self, plan, idempotency_key):
        """Only this service's confirmed attempt objects, fenced against active commits."""
        doc = self._validate(plan)
        with self._runner.slot():
            attempt = self._repo.recover(doc.run_id, idempotency_key)
            if (
                attempt is None
                or attempt["status"] != "CONFIRMED"
                or attempt["plan_checksum"] != doc.checksum
            ):
                raise ExecutionError("cleanup_requires_confirmed_owned_attempt")
            base = self._runner.workspace / attempt["attempt_id"]
            uris = {
                json.loads(read_file(path, 1024))["uri"] for path in base.glob("*/object-*.json")
            }
            referenced = self._repo.referenced_objects(uris)
            removed = []
            for uri in sorted(uris - referenced):
                self._objects.discard(uri)
                removed.append(uri)
            return tuple(removed)
