"""Additive upstream envelopes; frozen clinical/runtime contracts stay unchanged."""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Value


class Component(Value):
    id: str = Field(min_length=1, max_length=180)
    kind: Literal["image", "scalar", "missing_modality"]
    concept: str = Field(min_length=1, max_length=100)
    status: Literal["obtained", "missing"]
    value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = None
    unit_basis: str | None = None
    method: str | None = None
    reason: str | None = None
    laterality: Literal["LEFT", "RIGHT", "UNKNOWN"] = "UNKNOWN"
    timepoint: str = "unknown"
    source_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    source_locator: str = Field(min_length=1, max_length=240)
    source_field: str | None = None
    shape: tuple[int, ...] = ()
    transforms: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistency(self):
        if (self.reason is None) != (self.status == "obtained"):
            raise ValueError("component_missing_reason")
        if self.kind == "scalar" and (self.value is not None) != (self.status == "obtained"):
            raise ValueError("component_scalar_availability")
        if self.kind != "scalar" and self.value is not None:
            raise ValueError("component_non_scalar_value")
        if self.kind == "image" and (not self.shape or any(n <= 0 for n in self.shape)):
            raise ValueError("component_image_shape")
        if self.kind == "missing_modality" and self.status != "missing":
            raise ValueError("component_unavailable_modality")
        return self


class InputComponents(Value):
    schema_version: Literal["case-input-components/1.0.0"] = "case-input-components/1.0.0"
    dataset: str
    adapter_version: str
    scope: str
    input_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    split: Literal["training", "development_preparation"]
    patient_group: str | None
    patient_independence: Literal["unproven"] = "unproven"
    components: tuple[Component, ...] = Field(max_length=128)
    runtime_ready: bool
    hidden_fields_used: Literal[False] = False

    @model_validator(mode="after")
    def unique(self):
        if len({c.id for c in self.components}) != len(self.components):
            raise ValueError("component_duplicate_id")
        return self


class ProcedureStep(Value):
    id: str
    port: Literal["public_context", "segman", "review", "report"]
    depends_on: tuple[str, ...]
    input_roles: tuple[str, ...]
    output_roles: tuple[str, ...]


class SkillPlan(Value):
    schema_version: Literal["research-skill-plan/1.0.0"] = "research-skill-plan/1.0.0"
    scope: str
    input_checksum: str
    requirement_id: str
    definition_checksum: str
    components_checksum: str
    origin: Literal["current_owner_baseline", "research_memory_procedure"]
    memory_key: str | None
    memory_checksum: str | None
    steps: tuple[ProcedureStep, ...]
    missing_components: tuple[str, ...]
    fallback: Literal["source_backed_partial_observation", "record_input_gap"]
    publication_upgraded: Literal[False] = False
    execution_authorized: Literal[False] = False
    old_patient_values_copied: Literal[False] = False

    @model_validator(mode="after")
    def dag(self):
        seen = set()
        for step in self.steps:
            if step.id in seen or not set(step.depends_on) <= seen:
                raise ValueError("skill_dependency_order")
            seen.add(step.id)
        if (self.memory_key is None) != (self.memory_checksum is None):
            raise ValueError("skill_memory_binding")
        if (self.origin == "current_owner_baseline") != (self.memory_key is None):
            raise ValueError("skill_origin_binding")
        return self


def check(model, value):
    from pydantic import ValidationError

    try:
        return model.model_validate_json(json.dumps(value, allow_nan=False))
    except ValidationError as error:
        paths = ["/".join(map(str, e["loc"])) for e in error.errors(include_input=False)]
        raise ValueError("reuse_contract:" + model.__name__ + ":" + ",".join(paths)) from None


def main():
    from packages.evidence_reuse.definitions import DefinitionDraft

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for model in (InputComponents, SkillPlan, DefinitionDraft):
        path = Path(__file__).parent / "schemas" / (model.__name__ + ".json")
        expected = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        if args.write:
            path.parent.mkdir(exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            raise SystemExit("reuse_schema_drift:" + model.__name__)
    print("3 upstream envelope schemas verified")


if __name__ == "__main__":
    main()
