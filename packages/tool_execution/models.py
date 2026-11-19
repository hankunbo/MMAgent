"""06D deployment and execution records; shared contracts v1 remain unchanged."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.contracts import (
    QCStatus,
    ToolExecutionResult,
    ToolExecutionStatus,
    require_transition,
)
from packages.grounding.models import Checksum, Ref, Value, digest, validate

StepId = Annotated[str, Field(pattern=r"^step_[0-9a-f]{64}$")]
WorkflowStatus = Literal["SUCCEEDED", "FAILED", "PARTIAL_FAILED", "CANCELLED"]
# Runtime 0013: bounded metadata only, not model outputs or patient documents.
STEP_RESULT_MAX_BYTES = 4194303


class ExecutionError(ValueError):
    """A bounded technical reason, never a medical negative."""


class RuntimeFile(Value):
    source: str
    target: str
    checksum: Checksum
    executable: bool = False


class ExecutionPolicy(Value):
    version: Literal["06D.1.0.0"] = "06D.1.0.0"
    reference: Ref
    baseline_only: Literal[True] = True
    # Explicit closed deployment, including the interpreter's runtime closure.
    runtime_files: tuple[RuntimeFile, ...] = Field(min_length=1, max_length=8192)
    allowed_plan_checksums: tuple[Checksum, ...] = Field(min_length=1, max_length=512)
    purpose: Literal["debug", "shadow", "decisive_research"]
    synthetic_execution: bool
    environment: dict[str, str] = Field(default_factory=dict, max_length=16)
    cpu_seconds: int = Field(default=2, ge=1, le=3600)
    max_resource_bytes: int = Field(default=67108864, ge=1, le=1073741824)
    sandbox_uid: int = Field(default=61006, ge=60000, le=64999)
    process_limit: int = Field(default=16, ge=2, le=128)
    log_bytes: int = Field(default=65536, ge=1024, le=1048576)
    max_concurrent: Literal[1] = 1
    resource_semantics: Literal["per_process_as_cpu_affinity_and_wall_deadline"] = (
        "per_process_as_cpu_affinity_and_wall_deadline"
    )
    isolation: Literal["linux_chroot_drop_privileges_seccomp_x86_64/1.0.0"] = (
        "linux_chroot_drop_privileges_seccomp_x86_64/1.0.0"
    )

    @model_validator(mode="after")
    def coherent(self):
        if self.reference.kind != "policy":
            raise ValueError("execution_policy_reference_required")
        if len(set(self.allowed_plan_checksums)) != len(self.allowed_plan_checksums):
            raise ValueError("duplicate_allowed_plan")
        targets = [f.target for f in self.runtime_files]
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate_runtime_target")
        return self


class ExecutionDecision(Value):
    schema_version: Literal["execution-decision/1.0.0", "execution-decision/1.1.0"] = (
        "execution-decision/1.0.0"
    )
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    plan_checksum: Checksum
    scope_checksum: Checksum
    purpose: Literal["debug", "shadow", "decisive_research"]
    policy_checksum: Checksum
    execution_binding_checksum: Checksum
    # 1.0 records did not persist plan order. Omit this field on their wire format.
    planned_step_ids: tuple[StepId, ...] | None = Field(
        default=None, min_length=1, max_length=512, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def coherent(self):
        if (self.schema_version == "execution-decision/1.1.0") != (
            self.planned_step_ids is not None
        ):
            raise ValueError("decision_version_order_mismatch")
        if self.planned_step_ids and len(set(self.planned_step_ids)) != len(self.planned_step_ids):
            raise ValueError("decision_duplicate_step")
        return self


class QCRecord(Value):
    initial_qc_status: Literal["PENDING"] = "PENDING"
    status: QCStatus
    dimensions: tuple[int, ...]
    media_type: str
    policy_checksum: Checksum
    qc_policy_ref: Ref
    content_checksum: Checksum
    implementation: str
    logical_uri: str


class StepResult(Value):
    schema_version: Literal["tool-step-execution/1.0.0"] = "tool-step-execution/1.0.0"
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    step_id: str = Field(pattern=r"^step_[0-9a-f]{64}$")
    implementation_ref: Ref
    plan_checksum: Checksum
    binding_checksum: Checksum
    logical_to_physical: dict[str, str]
    copied_checksums: dict[str, Checksum]
    status_history: tuple[ToolExecutionStatus, ...]
    elapsed_seconds: float = Field(ge=0)
    exit_code: int | None = None
    error_code: str | None = None
    stdout_checksum: Checksum
    stderr_checksum: Checksum
    qc_records: tuple[QCRecord, ...] = ()
    result: ToolExecutionResult

    @model_validator(mode="after")
    def coherent(self):
        if (
            not self.status_history
            or self.status_history[0] != ToolExecutionStatus.REQUESTED
            or self.status_history[-1] != self.result.status
        ):
            raise ValueError("execution_history_mismatch")
        if self.result.status not in {
            ToolExecutionStatus.SUCCEEDED,
            ToolExecutionStatus.FAILED,
            ToolExecutionStatus.TIMED_OUT,
            ToolExecutionStatus.CANCELLED,
            ToolExecutionStatus.BLOCKED,
        }:
            raise ValueError("step_result_requires_terminal_status")
        for before, after in zip(self.status_history, self.status_history[1:], strict=False):
            if before == after:
                raise ValueError("duplicate_execution_history_state")
            require_transition(before, after)
        status = self.result.status
        running = ToolExecutionStatus.RUNNING in self.status_history
        inner_error = self.result.error.code if self.result.error is not None else None
        if self.error_code != inner_error:
            raise ValueError("step_error_code_mismatch")
        if not running and self.exit_code is not None:
            raise ValueError("unstarted_step_cannot_observe_exit")
        if status == ToolExecutionStatus.SUCCEEDED:
            if not running or self.exit_code != 0 or self.error_code is not None:
                raise ValueError("step_success_requires_zero_exit_without_error")
        elif not self.error_code:
            raise ValueError("unsuccessful_step_requires_error")
        if status == ToolExecutionStatus.TIMED_OUT and self.error_code != "timeout":
            raise ValueError("timeout_reason_mismatch")
        if self.error_code == "timeout" and status != ToolExecutionStatus.TIMED_OUT:
            raise ValueError("timeout_status_mismatch")
        if self.error_code == "cancelled" and status != ToolExecutionStatus.CANCELLED:
            raise ValueError("cancellation_status_mismatch")
        if status == ToolExecutionStatus.CANCELLED and self.error_code != "cancelled":
            # The v1 state machine can abort QUEUED/VALIDATED setup only via
            # CANCELLED. This is a launch failure, not an observed cancellation.
            if running or self.error_code not in {
                "execution_validation_failed",
                "subreaper_unavailable",
            }:
                raise ValueError("cancellation_reason_mismatch")
        if self.error_code in {"process_nonzero", "isolation_setup_failed", "resource_limit"}:
            if status != ToolExecutionStatus.FAILED or self.exit_code in (None, 0):
                raise ValueError("process_failure_requires_nonzero_exit")
            if self.error_code == "isolation_setup_failed" and self.exit_code != 125:
                raise ValueError("setup_failure_exit_mismatch")
        if self.error_code == "log_size_limit" and status != ToolExecutionStatus.FAILED:
            raise ValueError("log_limit_status_mismatch")
        if self.error_code and self.error_code.startswith(("output_", "qc_", "object_")):
            if running and (status != ToolExecutionStatus.FAILED or self.exit_code != 0):
                raise ValueError("output_validation_requires_observed_zero_exit")
        if (
            running
            and self.error_code == "execution_validation_failed"
            and self.exit_code not in (None, 0)
        ):
            raise ValueError("validation_failure_exit_mismatch")
        if self.result.implementation_code != implementation_identity(self.implementation_ref):
            raise ValueError("implementation_projection_mismatch")
        if (
            self.result.implementation_version != self.implementation_ref.version
            or self.result.implementation_checksum != self.implementation_ref.checksum
        ):
            raise ValueError("implementation_reference_mismatch")
        if self.result.decisive_evidence:
            raise ValueError("executor_cannot_approve_evidence")
        if self.result.status == ToolExecutionStatus.SUCCEEDED:
            if len(self.qc_records) != len(self.result.artifacts):
                raise ValueError("all_outputs_require_qc_record")
            for qc, artifact in zip(self.qc_records, self.result.artifacts, strict=True):
                if (
                    qc.content_checksum != artifact.checksum
                    or qc.status != artifact.qc_status
                    or qc.dimensions != artifact.dimensions
                    or qc.media_type != artifact.media_type
                    or qc.qc_policy_ref.checksum != qc.policy_checksum
                ):
                    raise ValueError("artifact_qc_binding_mismatch")
        return self


class WorkflowResult(Value):
    schema_version: Literal["tool-workflow-execution/1.0.0"] = "tool-workflow-execution/1.0.0"
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    plan_checksum: Checksum
    decision: ExecutionDecision
    status: WorkflowStatus
    steps: tuple[StepResult, ...] = Field(min_length=1, max_length=512)
    unstarted_steps: tuple[StepId, ...]
    decisive_evidence: Literal[False] = False
    clinical_effect: Literal["NONE"] = "NONE"
    medgraph_projection: Literal["not_requested_runtime_is_authority"] = (
        "not_requested_runtime_is_authority"
    )

    @model_validator(mode="after")
    def coherent(self):
        if (
            self.decision.attempt_id != self.attempt_id
            or self.decision.plan_checksum != self.plan_checksum
        ):
            raise ValueError("workflow_decision_mismatch")
        ids = [s.step_id for s in self.steps]
        if (
            len(set(ids)) != len(ids)
            or set(ids) & set(self.unstarted_steps)
            or len(set(self.unstarted_steps)) != len(self.unstarted_steps)
        ):
            raise ValueError("workflow_step_set_mismatch")
        if any(
            s.attempt_id != self.attempt_id or s.plan_checksum != self.plan_checksum
            for s in self.steps
        ):
            raise ValueError("workflow_step_binding_mismatch")
        if not ids and not self.unstarted_steps:
            raise ValueError("workflow_requires_planned_steps")
        if self.decision.planned_step_ids is not None and (
            tuple(ids) + self.unstarted_steps != self.decision.planned_step_ids
        ):
            raise ValueError("workflow_plan_order_mismatch")
        if any(s.result.status != ToolExecutionStatus.SUCCEEDED for s in self.steps[:-1]):
            raise ValueError("workflow_continues_after_terminal_failure")
        all_assets = {a.asset_code for s in self.steps for a in s.result.artifacts}
        seen_assets = set()
        for step in self.steps:
            validate_step_binding(step, self.decision)
            for artifact in step.result.artifacts:
                if (set(artifact.derived_from) & all_assets) - seen_assets:
                    raise ValueError("workflow_derivation_order_mismatch")
            seen_assets.update(a.asset_code for a in step.result.artifacts)
        if self.status != derive_workflow_status(self.steps, self.unstarted_steps):
            raise ValueError("workflow_outcome_mismatch")
        return self


def derive_workflow_status(steps, unstarted_steps) -> WorkflowStatus:
    """One deterministic rule for models, online execution and recovery.

    All executed + execution success + QC PASS; otherwise explicit cancellation;
    otherwise any execution-success prefix (even QC FAIL/PENDING); otherwise FAILED.
    A setup abort via v1 CANCELLED is not an explicit cancellation fact.
    """
    if (
        steps
        and not unstarted_steps
        and all(
            s.result.status == ToolExecutionStatus.SUCCEEDED and s.result.qc_status == QCStatus.PASS
            for s in steps
        )
    ):
        return "SUCCEEDED"
    if any(
        s.result.status == ToolExecutionStatus.CANCELLED and s.error_code == "cancelled"
        for s in steps
    ):
        return "CANCELLED"
    if any(s.result.status == ToolExecutionStatus.SUCCEEDED for s in steps):
        return "PARTIAL_FAILED"
    return "FAILED"


def validate_step_binding(step, decision):
    if step.attempt_id != decision.attempt_id or step.plan_checksum != decision.plan_checksum:
        raise ValueError("step_decision_identity_mismatch")
    if step.binding_checksum != digest(
        dict(mapping=step.logical_to_physical, copies=step.copied_checksums, decision=decision)
    ):
        raise ValueError("step_decision_binding_mismatch")


def validate_workflow_plan(workflow, document, decision):
    """Compare to the controlled compiler's document; a checksum alone is not authority."""
    workflow = validate(WorkflowResult, workflow)
    decision = validate(ExecutionDecision, decision)
    if (
        workflow.decision != decision
        or workflow.plan_checksum != document.checksum
        or decision.scope_checksum != digest(document.scope)
        or decision.purpose != document.purpose
        or tuple(s.step_id for s in workflow.steps) + workflow.unstarted_steps
        != tuple(s.step_id for s in document.steps)
    ):
        raise ValueError("workflow_controlled_plan_mismatch")
    for result, planned in zip(workflow.steps, document.steps, strict=False):
        if result.implementation_ref != planned.implementation_ref:
            raise ValueError("workflow_planned_implementation_mismatch")
        if (
            result.result.workflow_run_id != document.run_id
            or result.result.scope.case_code != document.scope.case_code
        ):
            raise ValueError("workflow_planned_scope_mismatch")
    return workflow


def implementation_identity(reference):
    """Deterministic v1 identity projection; preserve the exact native Ref separately."""
    return "IMPLEMENTATION:OID_" + digest(reference)[7:55].upper()
