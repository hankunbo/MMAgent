"""Explicit task selection over components; dataset task hints never select models.

This projection grants no medical interpretation or external execution. Profiles
declare the question, not its answer. Medical definitions are a separate owner.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.disease_packs.models import SealedDeclaration, seal
from packages.grounding.models import Code, Token, Value, Version, digest
from packages.research_runtime.task_context import TaskIntent

from .models import CaseComponents

DEFAULT_TASKS = Path("configs/research/multimodal_tasks_v1.json")


class Profile(Value):
    id: Token
    code: Code
    target_code: Code
    output_kind: Literal[
        "binary_evidence", "ordinal_evidence", "observed_change", "evidence_review"
    ]
    allowed_modalities: tuple[Token, ...] = Field(min_length=1, max_length=32)
    ordered_labels: tuple[Token, ...] = ()
    meaning: str = Field(min_length=1, max_length=1000)
    definition_status: Literal["research_question_not_medical_rule"]


class Registry(Value):
    schema_version: Literal["multimodal-task-registry/1.0.0"]
    version: Version
    profiles: tuple[Profile, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unique(self):
        if len({p.id for p in self.profiles}) != len(self.profiles) or len(
            {p.code for p in self.profiles}
        ) != len(self.profiles):
            raise ValueError("task_registry_duplicate")
        return self


class SelectedComponent(Value):
    id: str
    modality: str
    kind: str
    input_status: Literal["obtained", "partial", "missing", "withheld"]
    selected: bool
    disposition: Literal["available_input", "partial_input", "input_missing", "excluded"]
    reason: str
    unit: str | None
    axis_semantics: str
    requires_semantic_binding: bool
    visit: str
    laterality: str
    payload_ids: tuple[str, ...]
    next_step: Literal[
        "bind_measurement_definition", "bind_view_and_tool", "retain_gap", "not_in_task"
    ]


class TaskProjection(SealedDeclaration):
    schema_version: Literal["multimodal-task-projection/1.0.0"] = "multimodal-task-projection/1.0.0"
    input_checksum: str
    registry_checksum: str
    source_task_hint: str
    task: TaskIntent
    components: tuple[SelectedComponent, ...]
    selected_components: tuple[str, ...]
    cutoff_visit: int | None
    task_gaps: tuple[str, ...]
    clinical_acceptance: Literal[False] = False
    execution_authorized: Literal[False] = False
    medical_requirement_snapshot: None = None
    required_classifier: None = None
    score: None = None

    @model_validator(mode="after")
    def inventory(self):
        if tuple(c.id for c in self.components if c.selected) != self.selected_components:
            raise ValueError("task_projection_components")
        return self


def load_intent(task_id, path=DEFAULT_TASKS):
    from packages.grounding.models import seal as seal_value
    from packages.tool_execution.process import read_file

    registry = Registry.model_validate_json(read_file(Path(path).absolute(), 1048576))
    profile = next((p for p in registry.profiles if p.id == task_id), None)
    if profile is None:
        raise ValueError("task_profile_not_registered")
    intent = seal_value(
        TaskIntent,
        code=profile.code,
        version=registry.version,
        target_code=profile.target_code,
        output_kind=profile.output_kind,
        allowed_modalities=profile.allowed_modalities,
        ordered_labels=profile.ordered_labels,
        definition_refs=(
            dict(
                role="task_definition",
                code=profile.code,
                version=registry.version,
                checksum=digest(profile),
            ),
        ),
    )
    return intent, digest(registry)


def project(manifest, task_id, path=DEFAULT_TASKS):
    manifest = CaseComponents.model_validate_json(manifest.model_dump_json())
    task, registry_checksum = load_intent(task_id, path)
    components = []
    for c in manifest.components:
        selected = c.modality in task.allowed_modalities and c.status != "withheld"
        if not selected:
            disposition, reason, step = "excluded", "outside_task_or_withheld", "not_in_task"
        elif c.status == "missing":
            disposition, reason, step = "input_missing", c.reason, "retain_gap"
        else:
            disposition = "partial_input" if c.status == "partial" else "available_input"
            reason = c.reason or "source_observation_available_not_yet_interpreted"
            step = (
                "bind_measurement_definition"
                if c.kind in ("scalar", "vector", "map")
                else ("bind_view_and_tool")
            )
        semantic = c.unit is None or c.axis_semantics == "unknown" or bool(c.semantic_limits)
        components.append(
            SelectedComponent(
                id=c.id,
                modality=c.modality,
                kind=c.kind,
                input_status=c.status,
                selected=selected,
                disposition=disposition,
                reason=reason,
                unit=c.unit,
                axis_semantics=c.axis_semantics,
                requires_semantic_binding=selected and semantic,
                visit=c.visit,
                laterality=c.laterality,
                payload_ids=c.payload_ids,
                next_step=step,
            )
        )
    selected = tuple(c.id for c in components if c.selected)
    gaps = ["medical_requirement_owner_binding_pending", "task_execution_ports_pending"]
    if not selected:
        gaps.append("no_input_component_for_task")
    if task.output_kind == "observed_change":
        visits = {
            c.visit for c in components if c.selected and c.input_status in {"obtained", "partial"}
        }
        if len(visits) < 2:
            gaps.append("fewer_than_two_visible_visits")
        if manifest.cutoff_visit is None:
            gaps.append("cutoff_not_bound_no_longitudinal_comparison")
    if manifest.dataset == "harvard_gdp" and task.output_kind == "binary_evidence":
        gaps.append("VF_label_derivation_must_be_audited_before_evaluation")
    return seal(
        TaskProjection,
        input_checksum=manifest.checksum,
        registry_checksum=registry_checksum,
        source_task_hint=manifest.task,
        task=task,
        components=tuple(components),
        selected_components=selected,
        cutoff_visit=manifest.cutoff_visit,
        task_gaps=tuple(gaps),
    )


def write_projection(input_path, task_id, output, *, registry=DEFAULT_TASKS):
    from .preparation import verify

    value = project(verify(input_path), task_id, registry)
    out = Path(output).absolute()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("task_projection_output_symlink")
    with out.open("x") as stream:
        stream.write(value.model_dump_json(indent=2) + "\n")
    return json.loads(value.model_dump_json())
