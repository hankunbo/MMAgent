"""09A joins confirmed technical facts to their exact plan; grants no medical authority."""

from typing import Literal

from packages.contracts import (
    IntendedUse,
    QCStatus,
    ToolExecutionStatus,
    ToolValidationStatus,
    VersionedReference,
)
from packages.grounding.models import Checksum, Ref, Value, digest, validate
from packages.tool_execution.models import StepResult, implementation_identity
from packages.tool_workflow import ExecutionPlan
from packages.tool_workflow.models import PlanBundle

from .models import PipelineError


class ToolEvidenceCheck(Value):
    schema_version: Literal["tool-evidence-check/1.0.0"] = "tool-evidence-check/1.0.0"
    plan_checksum: Checksum
    step_checksum: Checksum
    implementation_ref: Ref
    catalog_ref: Ref
    tool_ref: Ref
    execution_status: ToolExecutionStatus
    qc_status: QCStatus
    validation_status: ToolValidationStatus
    intended_use: IntendedUse
    purpose: Literal["decisive_research", "debug", "shadow"]
    technical_pass: bool
    evidence_candidate: bool
    reason: str
    # Technical observation only, including when eligible for a subsequent interpreter.
    clinical_effect: Literal["NONE"] = "NONE"
    artifact_checksums: tuple[Checksum, ...]
    qc_checksums: tuple[Checksum, ...]


def inspect_tool_evidence(step, plan):
    """Consume owner-confirmed 06 facts, not a replacement for current admission.

    This port has no runner, writer or model. Source inconsistencies raise; ordinary
    execution/QC/nondecisive outcomes remain observations. The caller owns current
    admission and the durable workflow/decision join via validate_workflow_plan.
    """
    step = validate(StepResult, step)
    if type(plan) is not ExecutionPlan:
        raise PipelineError("complete_plan_bundle_required")
    bundle = PlanBundle.model_validate_json(plan.to_bundle_json())
    doc = bundle.plan
    if digest(bundle.context) != doc.context_checksum:
        raise PipelineError("tool_plan_context_mismatch")
    planned = next((s for s in doc.steps if s.step_id == step.step_id), None)
    result = step.result
    if (
        planned is None
        or step.plan_checksum != doc.checksum
        or step.implementation_ref != planned.implementation_ref
        or result.workflow_run_id != doc.run_id
        or (result.scope.case_code, result.scope.run_id) != (doc.scope.case_code, doc.run_id)
        or result.laterality != doc.scope.laterality
    ):
        raise PipelineError("tool_result_plan_source_mismatch")
    expected = VersionedReference(
        code=implementation_identity(planned.implementation_ref),
        version=planned.implementation_ref.version,
        checksum=planned.implementation_ref.checksum,
    )
    if result.provenance.tool_versions != (expected,):
        raise PipelineError("tool_result_producer_mismatch")
    if result.status.value == "SUCCEEDED":
        if len(result.artifacts) != len(planned.outputs):
            raise PipelineError("tool_output_set_mismatch")
        for output, qc, artifact in zip(
            planned.outputs, step.qc_records, result.artifacts, strict=True
        ):
            if qc.logical_uri != output.logical_uri or qc.qc_policy_ref != output.qc_policy_ref:
                raise PipelineError("tool_output_qc_source_mismatch")
            if (
                artifact.provenance.tool_versions != (expected,)
                or artifact.laterality != output.constraint.laterality
                or artifact.coordinate_space != output.constraint.coordinate_space
            ):
                raise PipelineError("tool_output_applicability_mismatch")
            rule = next(
                (r for r in planned.deployment.profile.output_rules if r.port == output.port), None
            )
            if (
                rule is None
                or artifact.media_type not in rule.media_types
                or len(artifact.dimensions) != len(rule.dimensions_min)
                or any(
                    not low <= n <= high
                    for n, low, high in zip(
                        artifact.dimensions, rule.dimensions_min, rule.dimensions_max, strict=True
                    )
                )
            ):
                raise PipelineError("tool_output_format_or_range_mismatch")
    # A claimed higher validation level cannot override the retained owner admission.
    if result.validation_status.value == "auto_validated_research":
        retained = next(
            (n for n in bundle.context.admission.retained if n.identity == planned.candidate_ref),
            None,
        )
        if retained is None or (
            retained.candidate.implementation_ref != planned.implementation_ref
            or retained.candidate.tool_ref != planned.tool_ref
            or retained.candidate.validation_required != result.validation_status
            or retained.candidate.intended_use != result.intended_use
            or retained.candidate.execution_policy != result.execution_policy
        ):
            raise PipelineError("tool_qualification_source_mismatch")
    technical = result.status.value == "SUCCEEDED" and result.qc_status.value == "PASS"
    if result.qc_status.value == "FAIL":
        reason = "qc_failure"
    elif result.status.value != "SUCCEEDED":
        reason = (
            "tool_execution_timeout"
            if result.status.value == "TIMED_OUT"
            else "tool_execution_failed"
        )
    elif not technical:
        reason = "qc_failure"
    elif result.validation_status.value != "auto_validated_research":
        reason = "tool_validation_level_not_decisive"
    elif doc.purpose != "decisive_research":
        reason = "nondecisive_research_observation"
    else:
        reason = "tool_evidence_candidate_only"
    return ToolEvidenceCheck(
        plan_checksum=doc.checksum,
        step_checksum=digest(step),
        implementation_ref=step.implementation_ref,
        catalog_ref=doc.catalog_ref,
        tool_ref=planned.tool_ref,
        execution_status=result.status,
        qc_status=result.qc_status,
        validation_status=result.validation_status,
        intended_use=result.intended_use,
        purpose=doc.purpose,
        technical_pass=technical,
        evidence_candidate=reason == "tool_evidence_candidate_only",
        reason=reason,
        artifact_checksums=tuple(a.checksum for a in result.artifacts),
        qc_checksums=tuple(digest(q) for q in step.qc_records),
    )
