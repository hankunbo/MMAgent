"""Exact deployment-reviewed metadata requirements, never request-supplied grants.

The review attests non-consumption for ALL permitted parameters. Digests prove
binding, not the truth of that code review or medical validity. Default/formal
admission remains strict. Units and known laterality conflicts are not waived.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Value, check_seal, digest, validate
from packages.grounding.review_files import ReviewedFile
from packages.integrations.medgraph.models import ReviewedDeploymentManifest

from .models import AdmissionError
from .research_metadata import limited_metadata

VERSION = "research-method-metadata/1.0.0"


class MethodMetadata(Value):
    schema_version: Literal["research-method-metadata/1.0.0"] = VERSION
    checksum: Checksum
    owner: Literal["06E_DEPLOYMENT_REVIEWER"] = "06E_DEPLOYMENT_REVIEWER"
    scope: Literal["single_input_current_observation"] = "single_input_current_observation"
    purposes: tuple[Literal["debug", "shadow"], ...] = Field(min_length=1, max_length=2)
    laterality: Literal["required", "not_consumed"]
    acquisition_time: Literal["required", "not_consumed"]
    parameters_scope: Literal["all_values_allowed_by_bound_schema"] = (
        "all_values_allowed_by_bound_schema"
    )
    basis: str = Field(min_length=20, max_length=4096)
    evidence: ReviewedFile
    execution_declaration_checksum: Checksum
    input_constraints_checksum: Checksum
    output_constraints_checksum: Checksum
    parameter_schema_checksum: Checksum
    preprocessing_checksum: Checksum
    postprocessing_checksum: Checksum
    missing_facts: Literal["preserve_UNKNOWN_and_null"] = "preserve_UNKNOWN_and_null"
    excluded_uses: tuple[
        Literal["bilateral_pairing"],
        Literal["longitudinal_comparison"],
        Literal["clinical_conclusion"],
    ] = ("bilateral_pairing", "longitudinal_comparison", "clinical_conclusion")

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        if self.evidence.byte_size == 0 or len(set(self.purposes)) != len(self.purposes):
            raise ValueError("method_metadata_review_required")
        return self


@dataclass(frozen=True)
class MetadataRequirements:
    laterality_required: bool = True
    acquisition_time_required: bool = True


def _bound(pre, post, execution, parameters, inputs, outputs):
    """No recursive hash: metadata_requirements is excluded ONLY from pre hash."""
    try:
        declaration = validate(MethodMetadata, pre["metadata_requirements"])
    except (KeyError, ValueError, TypeError):
        raise AdmissionError("research_method_metadata_binding") from None
    pairs = (
        (declaration.execution_declaration_checksum, execution),
        (declaration.input_constraints_checksum, inputs),
        (declaration.output_constraints_checksum, outputs),
        (declaration.parameter_schema_checksum, parameters),
        (
            declaration.preprocessing_checksum,
            {k: v for k, v in pre.items() if k != "metadata_requirements"},
        ),
        (declaration.postprocessing_checksum, post),
    )
    if len(inputs) != 1 or not outputs or any(expected != digest(v) for expected, v in pairs):
        raise AdmissionError("research_method_metadata_binding")
    if declaration.laterality == "not_consumed" and any(
        c.laterality.value not in {"UNKNOWN", "NOT_APPLICABLE"} for c in (*inputs, *outputs)
    ):
        # Not reading eye identity does NOT qualify unknown inputs for a LEFT/RIGHT-only method.
        raise AdmissionError("research_method_metadata_binding")
    return declaration


def validate_review_metadata(review):
    """Called by the original installed-review owner, not by a caller assertion."""
    declaration = _bound(
        review.preprocessing_policy,
        review.postprocessing_policy,
        review.deployment_declaration,
        review.parameter_schema,
        review.input_constraints,
        review.output_constraints,
    )
    if not set(declaration.purposes) <= set(review.purposes):
        raise AdmissionError("research_method_metadata_binding")
    return declaration


def metadata_requirements(manifest, candidate, purpose, tool=None):
    if manifest is None or purpose is None:
        return MetadataRequirements()
    pre = manifest.preprocessing_policy or {}
    raw = pre.get("metadata_requirements")
    if not isinstance(raw, dict) or raw.get("schema_version") != VERSION:
        limited = limited_metadata(manifest, candidate, purpose)
        return MetadataRequirements(not limited, not limited)
    if (
        not isinstance(manifest, ReviewedDeploymentManifest)
        or manifest.review_ref.checksum != manifest.policy_checksum
        or tool is None
        or (manifest.tool_ref, manifest.implementation_ref)
        != (candidate.tool_ref, candidate.implementation_ref)
        or (tool.tool_ref, tool.implementation_ref)
        != (candidate.tool_ref, candidate.implementation_ref)
        or candidate.inputs != tool.inputs
        or len(candidate.inputs) != 1
        or candidate.inputs[0].cardinality_min != 1
        or candidate.inputs[0].cardinality_max != 1
        or candidate.intended_use.value != "research_only"
        or candidate.execution_policy.value != "research"
        or candidate.output_port not in tool.output_ports
        or candidate.output != tool.outputs[tool.output_ports.index(candidate.output_port)]
    ):
        raise AdmissionError("research_method_metadata_binding")
    declaration = _bound(
        pre,
        manifest.postprocessing_policy or {},
        manifest.execution_manifest,
        manifest.parameter_schema,
        tuple(i.constraint for i in tool.inputs),
        tool.outputs,
    )
    if purpose not in declaration.purposes:
        raise AdmissionError("research_method_metadata_binding")
    return MetadataRequirements(
        declaration.laterality == "required", declaration.acquisition_time == "required"
    )
