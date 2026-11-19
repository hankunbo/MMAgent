"""07A manifests are separate from frozen contracts v1 and never authorize Knowledge."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.contracts import CaseBundle, StructuredObservation, VersionedReference
from packages.disease_packs.models import FieldVisibility, SealedDeclaration


class Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ObjectReference(Value):
    uri: str = Field(pattern=r"^object://knowledge/objects/[a-f0-9]{32}$")
    checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    byte_size: int = Field(ge=1, le=4194304)


class FileBinding(Value):
    relative_path: str = Field(max_length=200)
    checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    byte_size: int = Field(ge=1, le=33554432)

    @model_validator(mode="after")
    def safe_path(self):
        from pathlib import PurePosixPath

        p = PurePosixPath(self.relative_path)
        if p.is_absolute() or ".." in p.parts or "\\" in self.relative_path:
            raise ValueError("source_path_rejected")
        return self


class Span(Value):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    category: Literal["objective_candidate", "assessment", "unclassified"]
    hidden: Literal[True] = True
    reason: Literal["review_pending", "diagnosis_assessment", "inseparable_or_ambiguous"]
    lost_evidence: tuple[Literal["IOP", "OCT", "RNFL", "VF", "ANGLE", "CDR"], ...] = ()
    visible_start: Literal[0] = 0
    visible_end: Literal[0] = 0


class ObservationCandidate(Value):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    extraction_version: Literal["controlled-iop/1.0.0"] = "controlled-iop/1.0.0"
    laterality: Literal["LEFT", "RIGHT", "UNKNOWN"]
    timepoint: Literal["current", "unknown"]
    status: Literal["pending_review", "ambiguous"]
    observation: StructuredObservation | None = None


class Projection(Value):
    policy_version: Literal["conservative-note/1.0.0"] = "conservative-note/1.0.0"
    raw_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    raw_length: int = Field(ge=0)
    spans: tuple[Span, ...]
    candidates: tuple[ObservationCandidate, ...]
    visible_text: Literal[""] = ""
    review_status: Literal["pending"] = "pending"

    @model_validator(mode="after")
    def cover_source(self):
        cursor = 0
        for span in self.spans:
            if span.start != cursor or span.end <= span.start:
                raise ValueError("span_mapping_not_contiguous")
            cursor = span.end
        if cursor != self.raw_length:
            raise ValueError("span_mapping_incomplete")
        if any(c.end <= c.start or c.end > cursor for c in self.candidates):
            raise ValueError("candidate_span_invalid")
        return self


class VisibleManifest(SealedDeclaration):
    schema_version: Literal["visible-input/1.0.0"] = "visible-input/1.0.0"
    adapter_version: Literal["dataset-adapters/1.0.0"] = "dataset-adapters/1.0.0"
    dataset_ref: VersionedReference
    pack_ref: VersionedReference
    split: Literal["training"] = "training"
    visibility: FieldVisibility
    case: CaseBundle
    patient_identity: Literal["unknown", "dataset_local_explicit_id"]
    patient_independence: Literal["unproven"] = "unproven"
    split_contamination_risk: Literal[True] = True
    timepoint: Literal["unknown"] = "unknown"
    text_review: Literal["pending", "not_applicable"]
    available_modalities: tuple[Literal["SLO", "color_fundus"], ...]

    @model_validator(mode="after")
    def visible_boundary(self):
        c = self.case
        if self.dataset_ref.checksum is None or self.pack_ref.checksum is None:
            raise ValueError("exact_declaration_reference_required")
        expected = {
            "SOURCE:DATASET:FAIRVLMED": ("SLO", "ARTIFACT:SLO_IMAGE", "unknown", "pending"),
            "SOURCE:DATASET:ODIR": (
                "color_fundus",
                "ARTIFACT:FUNDUS_IMAGE",
                "dataset_local_explicit_id",
                "not_applicable",
            ),
        }.get(self.dataset_ref.code)
        if expected is None or self.available_modalities != (expected[0],):
            raise ValueError("dataset_modality_mismatch")
        if self.patient_identity != expected[2] or self.text_review != expected[3]:
            raise ValueError("dataset_identity_or_review_mismatch")
        if any(a.artifact_type_code != expected[1] for a in c.artifacts):
            raise ValueError("artifact_modality_mismatch")
        if self.patient_identity == "unknown":
            from packages.object_store import checksum

            if c.patient_hash != checksum(b"patient_identity_unknown_not_a_group"):
                raise ValueError("unknown_patient_must_not_be_fabricated")
            if c.laterality.value != "UNKNOWN" or len(c.artifacts) != 1:
                raise ValueError("unproven_eye_grouping")
        elif c.laterality.value != "BILATERAL" or tuple(
            a.laterality.value for a in c.artifacts
        ) != ("LEFT", "RIGHT"):
            raise ValueError("odir_requires_explicit_both_eyes")
        if c.extensions or c.report_uri or c.observations:
            raise ValueError("unreviewed_text_is_not_visible")
        if any(a.extensions or a.measurements or a.derived_from for a in c.artifacts):
            raise ValueError("unregistered_image_metadata")
        if not c.artifacts or any(a.scope != c.scope for a in c.artifacts):
            raise ValueError("input_artifact_scope_invalid")
        return self


class RestrictedManifest(SealedDeclaration):
    schema_version: Literal["restricted-evaluation/1.0.0"] = "restricted-evaluation/1.0.0"
    adapter_version: Literal["dataset-adapters/1.0.0"] = "dataset-adapters/1.0.0"
    access: Literal["offline_evaluator_only"] = "offline_evaluator_only"
    case_code: str
    dataset_ref: VersionedReference
    pack_ref: VersionedReference
    visibility: FieldVisibility
    source_files: tuple[FileBinding, ...]
    source_fields: ObjectReference
    original_note: ObjectReference | None = None
    projection: ObjectReference | None = None
    reference_graph: None = None
    expert_reviews: tuple[()] = ()
    reference_status: Literal["pending"] = "pending"
    label_scope: Literal["source_separated", "patient_coarse_and_derived_image_rows"]


class Registration(SealedDeclaration):
    schema_version: Literal["dataset-registration/1.0.0"] = "dataset-registration/1.0.0"
    visible: VisibleManifest
    restricted_manifest: ObjectReference


class BoundRegistration(SealedDeclaration):
    schema_version: Literal["dataset-registration/2.0.0"] = "dataset-registration/2.0.0"
    preparation_version: Literal["prepared-input/2.0.0"] = "prepared-input/2.0.0"
    fingerprint_version: Literal["registration-fingerprint/2.0.0"] = (
        "registration-fingerprint/2.0.0"
    )
    visible: VisibleManifest
    restricted_manifest: ObjectReference


def read_registration(raw):
    """Explicit read compatibility; never reinterpret a legacy checksum under v2 defaults."""
    import json

    from .multimodal.storage import MultimodalRegistration
    from .runtime_models import RuntimeRegistration

    version = json.loads(raw).get("schema_version")
    model = {
        "dataset-registration/1.0.0": Registration,
        "dataset-registration/2.0.0": BoundRegistration,
        "dataset-registration/3.0.0": RuntimeRegistration,
        "dataset-registration/4.0.0": MultimodalRegistration,
    }.get(version)
    if model is None:
        raise ValueError("dataset_registration_version_unsupported")
    return model.model_validate_json(raw)
