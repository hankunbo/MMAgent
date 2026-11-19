"""Versioned input capabilities from the original prepared-input owner.

Dataset names select decoders, never abilities here. A readable representation
is not a visible anatomical region, qualified measurement, or execution grant.
This first projection deliberately leaves visual scope unknown until an actual
observation owner supplies it; it does not pretend to perform image inspection.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.dataset_adapters.models import Value
from packages.disease_packs.models import SealedDeclaration, seal
from packages.grounding.models import digest

from .analysis import InputPlan
from .analysis import from_preparation as audit_preparation
from .models import SHA, CaseComponents
from .preparation import verify


class InputAbility(Value):
    name: Literal[
        "read_image",
        "read_image_series",
        "read_volume",
        "read_numeric_map",
        "read_numeric_vector",
        "read_scalar",
        "qualitative_visual_observation",
        "read_source_measurement",
        "absolute_image_measurement",
    ]
    status: Literal["available", "conditional", "unavailable"]
    basis: str = Field(min_length=1, max_length=240)
    prerequisites: tuple[str, ...] = Field(max_length=16)


class ComponentCapability(Value):
    component_id: str
    component_checksum: str = Field(pattern=SHA)
    payload_checksums: tuple[str, ...]
    representation: Literal["image", "image_series", "volume", "map", "vector", "scalar"]
    modality: str
    concept: str
    input_status: Literal["obtained", "partial", "missing", "withheld"]
    unit: str | None
    unit_basis: str | None
    laterality: Literal["LEFT", "RIGHT", "UNKNOWN"]
    visit: str
    axis_semantics: Literal["unknown", "source_index_only", "not_applicable"]
    # Do not infer retinal area from dataset ID, OCT name, or array shape.
    anatomical_region: Literal["unknown"] = "unknown"
    visual_scope: Literal["not_assessed"] = "not_assessed"
    physical_image_calibration: Literal["not_established"] = "not_established"
    acquisition_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    acquisition_basis: str
    limitations: tuple[str, ...]
    abilities: tuple[InputAbility, ...]

    @model_validator(mode="after")
    def consistent(self):
        if len({a.name for a in self.abilities}) != len(self.abilities):
            raise ValueError("capability_duplicate_ability")
        if self.input_status in {"missing", "withheld"} and any(
            a.status != "unavailable" for a in self.abilities
        ):
            raise ValueError("capability_unavailable_input")
        if self.input_status in {"missing", "withheld"} and self.payload_checksums:
            raise ValueError("capability_unavailable_payload")
        if (self.unit is None) != (self.unit_basis is None):
            raise ValueError("capability_unit_basis")
        return self


class InputCapabilities(SealedDeclaration):
    schema_version: Literal["input-capabilities/1.0.0"] = "input-capabilities/1.0.0"
    input_checksum: str = Field(pattern=SHA)
    audit_checksum: str = Field(pattern=SHA)
    components: tuple[ComponentCapability, ...] = Field(min_length=1, max_length=1024)
    provenance: Literal["original_prepared_components_and_input_audit"] = (
        "original_prepared_components_and_input_audit"
    )
    qualification: Literal["representation_only_visual_scope_pending"] = (
        "representation_only_visual_scope_pending"
    )
    execution_authorized: Literal[False] = False
    current_source_qualification: Literal["not_checked"] = "not_checked"
    clinical_acceptance: Literal[False] = False
    api_requests: Literal[0] = 0
    model_runs: Literal[0] = 0
    database_writes: Literal[0] = 0

    @model_validator(mode="after")
    def inventory(self):
        if len({c.component_id for c in self.components}) != len(self.components):
            raise ValueError("capability_duplicate_component")
        return self


def project(manifest, audit):
    """Pure projection after the original owner has checked payloads.

    Revalidating objects catches model_construct/model_copy bypasses. This
    function grants no right to dispatch a tool or to consume stale sources.
    """
    manifest = CaseComponents.model_validate_json(manifest.model_dump_json())
    audit = InputPlan.model_validate_json(audit.model_dump_json())
    if audit.input_checksum != manifest.checksum:
        raise ValueError("capability_audit_input_binding")
    audits = {c.component_id: c for c in audit.components}
    if len(audits) != len(audit.components) or set(audits) != {c.id for c in manifest.components}:
        raise ValueError("capability_audit_inventory")
    payloads = {p.id: p for p in manifest.payloads}
    results = []
    readers = {
        "image": "read_image",
        "image_series": "read_image_series",
        "volume": "read_volume",
        "map": "read_numeric_map",
        "vector": "read_numeric_vector",
        "scalar": "read_scalar",
    }
    for c in manifest.components:
        a = audits[c.id]
        if a.input_status != c.status:
            raise ValueError("capability_audit_status")
        present = c.status in {"obtained", "partial"} and a.acquisition > 0
        abilities = [
            InputAbility(
                name=readers[c.kind],
                status="available" if present else "unavailable",
                basis=a.acquisition_basis,
                prerequisites=(),
            )
        ]
        if c.kind in {"image", "image_series", "volume"}:
            abilities.extend(
                (
                    InputAbility(
                        name="qualitative_visual_observation",
                        status="conditional" if present else "unavailable",
                        basis="readable_container_does_not_establish_visible_anatomy",
                        prerequisites=(
                            "decoded_view",
                            "visible_scope",
                            "current_method_qualification",
                        ),
                    ),
                    InputAbility(
                        name="absolute_image_measurement",
                        status="conditional" if present else "unavailable",
                        basis="no_physical_image_calibration_in_component_contract",
                        prerequisites=(
                            "documented_physical_calibration",
                            "appropriate_anatomical_coverage",
                            "current_measurement_method_qualification",
                        ),
                    ),
                )
            )
        else:
            unit_known = c.unit is not None and c.unit_basis is not None
            abilities.append(
                InputAbility(
                    name="read_source_measurement",
                    status=("available" if unit_known else "conditional")
                    if present
                    else "unavailable",
                    basis="original_source_value_and_unit_not_new_measurement_or_medical_direction",
                    prerequisites=() if unit_known else ("source_unit_and_meaning",),
                )
            )
        results.append(
            ComponentCapability(
                component_id=c.id,
                component_checksum=digest(c),
                payload_checksums=tuple(payloads[k].checksum for k in c.payload_ids),
                representation=c.kind,
                modality=c.modality,
                concept=c.concept,
                input_status=c.status,
                unit=c.unit,
                unit_basis=c.unit_basis,
                laterality=c.laterality,
                visit=c.visit,
                axis_semantics=c.axis_semantics,
                acquisition_fraction=a.acquisition,
                acquisition_basis=a.acquisition_basis,
                limitations=tuple(dict.fromkeys((*c.semantic_limits, *a.issues))),
                abilities=tuple(abilities),
            )
        )
    return seal(
        InputCapabilities,
        input_checksum=manifest.checksum,
        audit_checksum=audit.checksum,
        components=tuple(results),
    )


def from_preparation(path):
    """Read the saved original input; never load credentials or run inference."""
    audit = audit_preparation(path)
    manifest = verify(path)
    return project(manifest, audit)


def read_snapshot(path, snapshot):
    """Rebuild from the original input, not from caller-written eligibility flags."""
    from packages.tool_execution.process import read_file

    saved = InputCapabilities.model_validate_json(read_file(Path(snapshot).absolute(), 4194304))
    current = from_preparation(path)
    if saved != current:
        raise ValueError("capability_snapshot_changed")
    return saved


def write_snapshot(path, output):
    value = from_preparation(path)
    out = Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("capability_output_symlink")
    with out.open("x", encoding="utf-8") as stream:
        stream.write(value.model_dump_json(indent=2) + "\n")
    return value
