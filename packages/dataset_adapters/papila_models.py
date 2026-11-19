"""Clinical preparation only; deliberately not a v1 VisibleManifest or run grant."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Value

from .models import FileBinding


class ClinicalValue(Value):
    value: float | None = Field(allow_inf_nan=False)
    unit: str | None
    unit_basis: Literal["paper", "convention_pending_confirmation", "unresolved", "coding"]
    source_field: str
    source_cell: str
    source_checksum: str
    method: str | None = None
    acquired: bool
    missing_reason: Literal["not_recorded"] | None = None
    timepoint: Literal["unknown"] = "unknown"
    interpretation_ready: Literal[False] = False

    @model_validator(mode="after")
    def consistent(self):
        if self.acquired != (self.value is not None):
            raise ValueError("clinical_acquisition_mismatch")
        if (self.missing_reason is None) != self.acquired:
            raise ValueError("clinical_missing_reason_mismatch")
        return self


class PapilaClinicalInput(Value):
    schema_version: Literal["papila-clinical-preparation/1.0.0"] = (
        "papila-clinical-preparation/1.0.0"
    )
    dataset: Literal["PAPILA"] = "PAPILA"
    source_version: Literal["PAPILAv1.1"] = "PAPILAv1.1"
    purpose: Literal["development_preparation"] = "development_preparation"
    profile: Literal["image_demographics", "image_demographics_clinical"]
    sample_id: str = Field(pattern=r"^RET[0-9]{3}(OD|OS)$")
    patient_group: str = Field(pattern=r"^PAPILA:RET[0-9]{3}$")
    laterality: Literal["RIGHT", "LEFT"]
    modality: Literal["color_fundus"] = "color_fundus"
    image: FileBinding
    dimensions: tuple[int, int]
    age_years: float | None = Field(ge=0, le=120)
    sex: Literal["male", "female"] | None
    sex_mapping: Literal["PAPILA:0=male;1=female"] = "PAPILA:0=male;1=female"
    clinical: dict[str, ClinicalValue]
    clinical_source: FileBinding
    definition_source: Literal["https://doi.org/10.1038/s41597-022-01388-1"] = (
        "https://doi.org/10.1038/s41597-022-01388-1"
    )
    unavailable_modalities: tuple[Literal["OCT_volume", "VF_pointwise", "longitudinal"], ...] = (
        "OCT_volume",
        "VF_pointwise",
        "longitudinal",
    )
    runtime_admission: Literal[False] = False
    final_test_unlocked: Literal[False] = False

    @model_validator(mode="after")
    def bindings(self):
        if self.patient_group != "PAPILA:" + self.sample_id[:6]:
            raise ValueError("papila_patient_binding")
        eye = "RIGHT" if self.sample_id.endswith("OD") else "LEFT"
        if self.laterality != eye:
            raise ValueError("papila_eye_binding")
        if self.image.relative_path != f"FundusImages/{self.sample_id}.jpg":
            raise ValueError("papila_image_binding")
        suffix = "od" if eye == "RIGHT" else "os"
        if self.clinical_source.relative_path != f"ClinicalData/patient_data_{suffix}.xlsx":
            raise ValueError("papila_workbook_binding")
        fields = {
            "dioptre_1",
            "dioptre_2",
            "astigmatism",
            "lens_removed",
            "iop_pneumatic",
            "iop_perkins",
            "pachymetry",
            "axial_length",
            "vf_md",
        }
        expected = set() if self.profile == "image_demographics" else fields
        if set(self.clinical) != expected:
            raise ValueError("papila_profile_fields")
        if any(v.source_checksum != self.clinical_source.checksum for v in self.clinical.values()):
            raise ValueError("papila_cell_source_binding")
        return self
