"""GRO-001 pure hard filters shared with 06B. No execution authorization."""

from .models import Candidate, CatalogSnapshot, Policy, validate
from .ports import ToolSafety


def tool_compatibility(candidate, catalog, policy, safety, *, admission=True) -> tuple[str, ...]:
    candidate = validate(Candidate, candidate)
    catalog = validate(CatalogSnapshot, catalog)
    policy = validate(Policy, policy)
    if admission:
        safety = validate(ToolSafety, safety)
        if safety.catalog_ref != catalog.reference():
            raise ValueError("safety_catalog_mismatch")
    reasons = []
    for constraint in (candidate.output, *(i.constraint for i in candidate.inputs)):
        if constraint.artifact_ref not in catalog.artifacts:
            reasons.append("artifact_outside_catalog")
        terms = (*constraint.scope_terms, constraint.modality.code, constraint.unit.code)
        if any(t and t not in policy.allowed_terms for t in terms):
            reasons.append("unknown_scope_term")
    capabilities = {c for t in catalog.tools for c in t.capabilities}
    if candidate.capability_ref not in capabilities:
        reasons.append("capability_outside_tool")
    matches = [
        t
        for t in catalog.tools
        if t.tool_ref == candidate.tool_ref and t.implementation_ref == candidate.implementation_ref
    ]
    if len(matches) != 1:
        reasons.append("tool_outside_catalog")
        return tuple(dict.fromkeys(reasons))
    tool = matches[0]
    if candidate.capability_ref not in tool.capabilities:
        reasons.append("capability_outside_tool")
    if candidate.workflow_ref and candidate.workflow_ref not in tool.workflows:
        reasons.append("workflow_outside_tool")
    if candidate.estimate_ref and candidate.estimate_ref not in catalog.estimates:
        reasons.append("estimate_outside_catalog")
    if candidate.output_port not in tool.output_ports:
        reasons.append("output_port_scope_mismatch")
    elif tool.outputs[tool.output_ports.index(candidate.output_port)] != candidate.output:
        reasons.append("output_type_mismatch")
    if set(candidate.inputs) != set(tool.inputs):
        reasons.append("complete_input_set_required")
    if admission:
        if not tool.enabled or tool.catalog_status.value != "active":
            reasons.append("tool_not_active")
        if tool.implementation_ref in safety.disabled_implementations:
            reasons.append("tool_safety_revoked")
        if (
            tool.validation_status != candidate.validation_required
            or tool.validation_status not in policy.accepted_validation
        ):
            reasons.append("tool_validation_rejected")
        if (
            candidate.intended_use != policy.intended_use
            or tool.intended_use != policy.intended_use
            or candidate.execution_policy != policy.execution_policy
        ):
            reasons.append("intended_use_rejected")
    return tuple(dict.fromkeys(reasons))


def constraint_compatibility(constraint, request, *, laterality_required=True) -> tuple[str, ...]:
    """Unknown required metadata/conditions fail closed; no implicit conversion."""
    from .models import ArtifactConstraint

    constraint = validate(ArtifactConstraint, constraint)
    reasons = []
    if constraint.laterality.value == "UNKNOWN" or request.scope.laterality.value == "UNKNOWN":
        if laterality_required:
            reasons.append("required_condition_unknown")
    elif constraint.laterality != request.scope.laterality:
        reasons.append("laterality_mismatch")
    if constraint.coordinate_space.value == "UNKNOWN" or any(
        t.state in {"unknown", "unspecified"} for t in (constraint.modality, constraint.unit)
    ):
        reasons.append("required_condition_unknown")
    if not set(constraint.scope_terms) <= set(request.satisfied_terms):
        reasons.append("required_condition_unknown")
    return tuple(dict.fromkeys(reasons))


def artifact_compatibility(
    artifact, constraint, request, *, laterality_required=True, acquisition_time_required=True
) -> tuple[str, ...]:
    """Metadata only. 06B must repeat these rules with verified inputs at execution time."""
    from .query_models import ArtifactMetadata

    artifact = validate(ArtifactMetadata, artifact)
    reasons = list(
        constraint_compatibility(constraint, request, laterality_required=laterality_required)
    )
    for field, reason in (
        ("artifact_ref", "artifact_type_mismatch"),
        ("modality", "modality_mismatch"),
        ("laterality", "laterality_mismatch"),
        ("coordinate_space", "coordinate_mismatch"),
        ("unit", "unit_mismatch"),
    ):
        if (
            field == "laterality"
            and not laterality_required
            and "UNKNOWN" in (artifact.laterality.value, constraint.laterality.value)
        ):
            continue
        if getattr(artifact, field) != getattr(constraint, field):
            reasons.append(reason)
    if (
        not laterality_required
        and "UNKNOWN" not in (artifact.laterality.value, request.scope.laterality.value)
        and artifact.laterality != request.scope.laterality
    ):
        reasons.append("laterality_mismatch")
    if artifact.qc_status not in constraint.qc_required:
        reasons.append("qc_rejected")
    if not set(constraint.scope_terms) <= set(artifact.scope_terms):
        reasons.append("scope_mismatch")
    if (artifact.case_code, artifact.patient_hash, artifact.encounter_code) != (
        request.scope.case_code,
        request.scope.patient_hash,
        request.scope.encounter_code,
    ) or artifact.run_id not in (None, request.lock.run_id):
        reasons.append("scope_mismatch")
    if not acquisition_time_required:
        pass  # An unused acquisition time is not a temporal selection criterion.
    elif artifact.acquired_at is None:
        reasons.append("time_unknown")
    elif not request.window_start <= artifact.acquired_at <= request.window_end:
        reasons.append("time_mismatch")
    return tuple(dict.fromkeys(reasons))
