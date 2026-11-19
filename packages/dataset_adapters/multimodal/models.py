"""Additive contracts: original input/clinical contracts keep their exact semantics."""

from typing import Literal

from pydantic import Field, model_validator

from packages.dataset_adapters.models import Value
from packages.disease_packs.models import SealedDeclaration

SHA = r"^sha256:[a-f0-9]{64}$"
Dataset = Literal["harvard_gdp", "gamma", "grape", "fairvlmed"]


class SourceFile(Value):
    path: str = Field(min_length=1, max_length=240)
    checksum: str = Field(pattern=SHA)
    byte_size: int = Field(ge=1, le=134217728)

    @model_validator(mode="after")
    def relative(self):
        from pathlib import PurePosixPath

        p = PurePosixPath(self.path)
        if p.is_absolute() or ".." in p.parts or "\\" in self.path or ":" in self.path:
            raise ValueError("multimodal_source_path")
        return self


class Payload(Value):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.:-]{1,180}$")
    checksum: str = Field(pattern=SHA)
    byte_size: int = Field(ge=1, le=67108864)
    media_type: Literal["application/x-npy", "image/jpeg", "image/png", "application/json"]
    source_path: str
    source_locator: str = Field(min_length=1, max_length=300)
    # Raw NPZ/table containers with reference labels never enter this collection.
    projection: Literal["selected_observation", "source_image", "visible_cells"]


class Component(Value):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.:-]{1,180}$")
    kind: Literal["image", "volume", "image_series", "map", "vector", "scalar"]
    concept: str = Field(min_length=1, max_length=100)
    modality: str = Field(min_length=1, max_length=100)
    status: Literal["obtained", "partial", "missing", "withheld"]
    value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = None
    unit_basis: str | None = None
    shape: tuple[int, ...] = ()
    dtype: str | None = None
    payload_ids: tuple[str, ...] = Field(default=(), max_length=1024)
    source_paths: tuple[str, ...] = ()
    source_locator: str
    reason: str | None = None
    laterality: Literal["LEFT", "RIGHT", "UNKNOWN"] = "UNKNOWN"
    visit: str = "unknown"
    interval_years: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    axis_semantics: Literal["unknown", "source_index_only", "not_applicable"] = "unknown"
    expected_members: int | None = Field(default=None, ge=1, le=1024)
    member_indices: tuple[int, ...] = ()
    semantic_limits: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "obtained") != (self.reason is None):
            raise ValueError("multimodal_status_reason")
        if any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("multimodal_shape")
        if self.status in ("missing", "withheld"):
            if self.value is not None or self.payload_ids:
                raise ValueError("multimodal_unavailable_has_value")
        elif self.kind == "scalar":
            if self.value is None or self.payload_ids:
                raise ValueError("multimodal_scalar_value")
        elif not self.payload_ids or self.value is not None:
            raise ValueError("multimodal_payload_required")
        if (self.unit is None) != (self.unit_basis is None):
            raise ValueError("multimodal_unit_basis")
        if len(self.payload_ids) != len(set(self.payload_ids)):
            raise ValueError("multimodal_duplicate_payload")
        if self.kind == "image_series" and self.status in ("obtained", "partial"):
            if (
                self.expected_members is None
                or len(self.member_indices) != len(self.payload_ids)
                or tuple(sorted(set(self.member_indices))) != self.member_indices
                or any(n < 0 or n >= self.expected_members for n in self.member_indices)
                or (self.status == "obtained") != (len(self.payload_ids) == self.expected_members)
            ):
                raise ValueError("multimodal_series_indices")
        elif self.member_indices or self.expected_members is not None:
            raise ValueError("multimodal_non_series_indices")
        return self


class CaseComponents(SealedDeclaration):
    schema_version: Literal["case-input-components/2.0.0"] = "case-input-components/2.0.0"
    adapter_version: Literal[
        "multimodal-adapters/1.0.0",
        "multimodal-adapters/1.1.0",
        "multimodal-adapters/1.2.0",
        "multimodal-adapters/1.3.0",
    ] = "multimodal-adapters/1.0.0"
    dataset: Dataset
    release_basis: str
    sample: str
    selection: Literal["development_preparation", "training"]
    task: Literal["glaucoma_detection", "glaucoma_grading", "observed_change"]
    visible_policy: Literal["multimodal-visible/1.0.0"] = "multimodal-visible/1.0.0"
    patient_group: str | None = None
    patient_identity_basis: Literal["unknown", "explicit_dataset_id"] = "unknown"
    patient_independence: Literal["unproven"] = "unproven"
    cutoff_visit: int | None = Field(default=None, ge=0, le=1000)
    sources: tuple[SourceFile, ...] = Field(min_length=1, max_length=1024)
    payloads: tuple[Payload, ...] = Field(max_length=1024)
    components: tuple[Component, ...] = Field(min_length=1, max_length=1024)
    quarantined_fields: tuple[str, ...] = ()
    excluded_fields: tuple[str, ...]
    limitations: tuple[str, ...]
    runtime_ready: Literal[False] = False
    hidden_values_used: Literal[False] = False

    @model_validator(mode="after")
    def binding(self):
        if (self.dataset == "fairvlmed") != (self.adapter_version == "multimodal-adapters/1.3.0"):
            raise ValueError("multimodal_fairvlmed_adapter_binding")
        if self.adapter_version != "multimodal-adapters/1.3.0" and any(
            p.media_type == "image/png" for p in self.payloads
        ):
            raise ValueError("multimodal_png_adapter_version")
        if self.adapter_version == "multimodal-adapters/1.1.0" and self.dataset != "harvard_gdp":
            raise ValueError("multimodal_semantic_profile_dataset")
        if self.adapter_version == "multimodal-adapters/1.2.0" and self.dataset != "grape":
            raise ValueError("multimodal_semantic_profile_dataset")
        paths = {s.path for s in self.sources}
        payloads = {p.id: p for p in self.payloads}
        if len(paths) != len(self.sources) or len(payloads) != len(self.payloads):
            raise ValueError("multimodal_duplicate_binding")
        if len({c.id for c in self.components}) != len(self.components):
            raise ValueError("multimodal_duplicate_component")
        if any(p.source_path not in paths for p in self.payloads):
            raise ValueError("multimodal_payload_source")
        if any(
            not set(c.payload_ids) <= payloads.keys() or not set(c.source_paths) <= paths
            for c in self.components
        ):
            raise ValueError("multimodal_component_binding")
        if set(payloads) != {p for c in self.components for p in c.payload_ids}:
            raise ValueError("multimodal_orphan_payload")
        if (self.patient_group is None) != (self.patient_identity_basis == "unknown"):
            raise ValueError("multimodal_patient_basis")
        expected = {
            "harvard_gdp": ("development_preparation", "glaucoma_detection"),
            "gamma": ("training", "glaucoma_grading"),
            "grape": ("development_preparation", "observed_change"),
            "fairvlmed": ("training", "glaucoma_detection"),
        }
        if (self.selection, self.task) != expected[self.dataset]:
            raise ValueError("multimodal_task_or_split")
        if (self.dataset == "grape") != (self.cutoff_visit is not None):
            raise ValueError("multimodal_cutoff_required")
        return self
