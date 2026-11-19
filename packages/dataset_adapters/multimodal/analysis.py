"""Source-order, numeric acquisition audit. Not medical interpretation or a run grant.

This port deliberately does not infer anatomical axes, normative percentiles,
patient diagnoses or disease probabilities from structural completeness.
"""

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.dataset_adapters.models import Value
from packages.dataset_adapters.reading import AdapterError
from packages.disease_packs.models import SealedDeclaration, seal
from packages.object_store import checksum
from packages.tool_execution.process import read_file

from .formats import npy
from .models import SHA, CaseComponents
from .preparation import verify


class ComponentAudit(Value):
    component_id: str
    input_status: Literal["obtained", "partial", "missing", "withheld"]
    acquisition: float = Field(ge=0, le=1, allow_inf_nan=False)
    acquisition_basis: str
    finite_values: int | None = Field(default=None, ge=0)
    total_values: int | None = Field(default=None, ge=0)
    clinical_reliability: None = None
    disease_direction_score: None = None
    issues: tuple[str, ...]
    next_step: Literal[
        "retain_input_gap",
        "review_source_measurement",
        "qualify_array_semantics",
        "qualify_image_tool",
        "qualify_volume_tool",
    ]

    @model_validator(mode="after")
    def bounded(self):
        if self.input_status in ("missing", "withheld") and self.acquisition != 0:
            raise ValueError("multimodal_unavailable_acquisition")
        if (self.finite_values is None) != (self.total_values is None):
            raise ValueError("multimodal_audit_count")
        if self.total_values is not None and not 0 <= self.finite_values <= self.total_values:
            raise ValueError("multimodal_audit_count")
        return self


class InputPlan(SealedDeclaration):
    schema_version: Literal["multimodal-input-plan/1.0.0"] = "multimodal-input-plan/1.0.0"
    input_checksum: str = Field(pattern=SHA)
    task: Literal["glaucoma_detection", "glaucoma_grading", "observed_change"]
    task_meaning: str
    components: tuple[ComponentAudit, ...]
    visit_groups: dict[str, tuple[str, ...]]
    task_gaps: tuple[str, ...]
    shared_dependency_groups: dict[str, tuple[str, ...]]
    integration_gaps: tuple[str, ...]
    available_actions: tuple[Literal["check", "prepare", "plan", "verify"], ...]
    clinical_evaluation_ready: Literal[False] = False
    inference_ready: Literal[False] = False
    medical_requirement_snapshot: None = None
    score: None = None
    api_requests: Literal[0] = 0
    model_runs: Literal[0] = 0


def audit_component(component, payloads, descriptors):
    c = component
    issues = list(c.semantic_limits)
    total = finite = None
    if c.status in ("missing", "withheld"):
        fraction, basis, next_step = 0.0, "input_not_available", "retain_input_gap"
        issues.append(c.reason)
    elif c.kind == "scalar":
        fraction, basis, next_step = 1.0, "finite_source_value", "review_source_measurement"
        if c.unit is None:
            issues.append("unit_unqualified_no_threshold_or_cross_unit_comparison")
    elif c.kind == "image_series":
        fraction = len(c.member_indices) / c.expected_members
        basis, next_step = "provided_slice_count_not_tissue_visibility", "qualify_volume_tool"
        issues.append("jpeg_header_only_not_pixel_decode_or_medical_QC")
    elif c.kind == "image":
        fraction, basis, next_step = 1.0, "container_header_present", "qualify_image_tool"
        if len(c.payload_ids) == 1 and descriptors[c.payload_ids[0]].media_type == "image/png":
            from packages.tool_execution.qc import grayscale_png

            shape, _ = grayscale_png(payloads[c.payload_ids[0]], max_pixels=1048576)
            if shape != c.shape:
                raise AdapterError("multimodal_audit_image_binding")
            basis = "lossless_grayscale_pixels_not_medical_quality"
            issues.append("decoded_grayscale_not_anatomical_visibility_or_medical_QC")
        else:
            issues.append("jpeg_header_only_not_pixel_decode_or_medical_QC")
    else:
        if len(c.payload_ids) != 1:
            raise AdapterError("multimodal_audit_payload_count")
        id = c.payload_ids[0]
        raw = payloads[id]
        if descriptors[id].media_type == "application/x-npy":
            array = npy(raw)
            if array.shape != c.shape or array.dtype != c.dtype:
                raise AdapterError("multimodal_audit_array_binding")
            finite, total = array.finite_counts()
        elif descriptors[id].media_type == "application/json":
            parsed = json.loads(raw)
            if (
                set(parsed) != {"source_order", "values"}
                or parsed["source_order"] != list(range(math.prod(c.shape)))
                or len(parsed["values"]) != math.prod(c.shape)
            ):
                raise AdapterError("multimodal_audit_vector_binding")
            total = finite = 0
            for value in parsed["values"]:
                total += 1
                if value is not None and math.isfinite(value):
                    finite += 1
        else:
            raise AdapterError("multimodal_audit_payload_type")
        fraction = finite / total if total else 0.0
        basis = "finite_elements_not_clinical_quality"
        next_step = "qualify_volume_tool" if c.kind == "volume" else "qualify_array_semantics"
        if finite != total:
            issues.append("nonfinite_or_missing_values_not_imputed")
        if c.axis_semantics != "not_applicable":
            issues.append("source_indices_are_not_anatomical_coordinates")
    return ComponentAudit(
        component_id=c.id,
        input_status=c.status,
        acquisition=fraction,
        acquisition_basis=basis,
        total_values=total,
        finite_values=finite,
        issues=tuple(dict.fromkeys(issues)),
        next_step=next_step,
    )


def build(manifest, payloads):
    manifest = CaseComponents.model_validate_json(manifest.model_dump_json())
    descriptors = {p.id: p for p in manifest.payloads}
    if set(payloads) != set(descriptors) or any(
        len(payloads[id]) != d.byte_size or checksum(payloads[id]) != d.checksum
        for id, d in descriptors.items()
    ):
        raise AdapterError("multimodal_audit_payload_checksum")
    groups, dependencies = {}, {}
    for c in manifest.components:
        groups.setdefault(c.visit, []).append(c.id)
        for p in c.source_paths:
            dependencies.setdefault(p, []).append(c.id)
    gaps = []
    if manifest.task == "observed_change" and len(groups) < 2:
        gaps.append("fewer_than_two_visible_visits_no_observed_change")
    if manifest.patient_group is None:
        gaps.append("patient_independence_not_established")
    if manifest.task == "glaucoma_grading":
        gaps.append("ordinal_grade_definition_and_acceptance_not_installed")
    if manifest.dataset == "harvard_gdp":
        gaps.append("VF_label_derivation_audit_required_before_prediction")
    return seal(
        InputPlan,
        input_checksum=manifest.checksum,
        task=manifest.task,
        task_meaning={
            "glaucoma_detection": "research_detection_not_calibrated_disease_probability",
            "glaucoma_grading": "ordered_grades_not_binary_score_thresholds",
            "observed_change": "compare_visible_history_not_predict_future_progression",
        }[manifest.task],
        components=tuple(audit_component(c, payloads, descriptors) for c in manifest.components),
        visit_groups={k: tuple(v) for k, v in groups.items()},
        shared_dependency_groups={k: tuple(v) for k, v in dependencies.items()},
        task_gaps=tuple(gaps),
        integration_gaps=(
            "new_modality_current_05_06_tool_admission_not_bound",
            "research_run_entry_still_bound_to_installed_FairCLIP_SLO_request",
            "goal_specific_requirement_definition_and_review_binding_pending",
            "legacy_score_coverage_domains_not_new_task_semantics",
            "new_run_acceptance_score_memory_report_readback_not_executed",
        ),
        available_actions=("check", "prepare", "plan", "verify"),
    )


def from_preparation(path):
    p = Path(path).absolute()
    manifest = verify(p)
    payloads = {
        d.id: read_file(p.parent / "payloads" / (checksum(d.id.encode())[7:] + ".bin"), d.byte_size)
        for d in manifest.payloads
    }
    return build(manifest, payloads)
