"""Task-scoped research identity in the original Runtime, independent of tools.

This is NOT a tool admission, medical-source approval or a Provider permit.
Prepared inputs remain immutable. The chosen task is a separate bound value.
"""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import (
    Checksum,
    Code,
    Ref,
    Token,
    Value,
    Version,
    check_seal,
    digest,
    seal,
    validate,
)
from packages.grounding.query_models import CaseScope
from packages.planner_control.models import Reference
from packages.tool_admission.models import ResearchLock


class TaskIntent(Value):
    schema_version: Literal["research-task-intent/1.0.0"] = "research-task-intent/1.0.0"
    checksum: Checksum
    code: Code
    version: Version
    target_code: Code
    output_kind: Literal[
        "binary_evidence", "ordinal_evidence", "observed_change", "evidence_review"
    ]
    ordered_labels: tuple[Token, ...] = Field(default=(), max_length=16)
    allowed_modalities: tuple[Token, ...] = Field(min_length=1, max_length=32)
    definition_refs: tuple[Reference, ...] = Field(min_length=1, max_length=16)
    research_only: Literal[True] = True
    clinical_acceptance: Literal[False] = False
    missing_evidence: Literal["report_scope_and_gaps"] = "report_scope_and_gaps"

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        if not self.code.startswith("TASK:") or not self.target_code.startswith("GOAL:"):
            raise ValueError("task_intent_namespace")
        if len(set(self.allowed_modalities)) != len(self.allowed_modalities) or len(
            {r.code for r in self.definition_refs}
        ) != len(self.definition_refs):
            raise ValueError("task_intent_unique")
        if self.output_kind == "ordinal_evidence":
            if len(self.ordered_labels) < 2 or len(set(self.ordered_labels)) != len(
                self.ordered_labels
            ):
                raise ValueError("task_ordered_labels_required")
        elif self.ordered_labels:
            raise ValueError("task_ordered_labels_not_applicable")
        return self


class TaskInput(Value):
    asset_code: Code
    checksum: Checksum
    artifact_type_code: Code
    media_type: str = Field(min_length=1, max_length=128)


class TaskRunRequest(Value):
    schema_version: Literal["task-research-request/1.0.0"] = "task-research-request/1.0.0"
    checksum: Checksum
    scope: CaseScope
    lock: ResearchLock
    task: TaskIntent
    policy_ref: Ref
    input_checksum: Checksum
    visible_checksum: Checksum
    artifacts: tuple[TaskInput, ...] = Field(min_length=1, max_length=256)
    selected_components: tuple[str, ...] = Field(max_length=1024)
    cutoff_visit: int | None = Field(default=None, ge=0, le=1000)
    purpose: Literal["shadow"] = "shadow"
    decisive_evidence: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        if self.policy_ref.kind != "policy":
            raise ValueError("task_policy_reference")
        if len(set(self.selected_components)) != len(self.selected_components) or len(
            {a.asset_code for a in self.artifacts}
        ) != len(self.artifacts):
            raise ValueError("task_request_duplicate")
        return self


class TaskRunContext(Value):
    schema_version: Literal["research-run/2.0.0"] = "research-run/2.0.0"
    checksum: Checksum
    request: TaskRunRequest
    synthetic: bool
    research_only: Literal[True] = True
    decisive_evidence: Literal[False] = False
    scope_checksum: Checksum

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        if self.scope_checksum != digest(self.request.scope):
            raise ValueError("research_context_scope_checksum")
        return self

    @property
    def run_id(self):
        return self.request.lock.run_id

    def validate_plan(self, plan):
        # A task identity never authorizes an old tool request by shape alone.
        # A future explicit task-to-admitted-plan port must own that association.
        raise ValueError("task_tool_admission_binding_not_installed")


def read_context(value):
    from .models import ResearchRunContext

    raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    if type(raw) is not dict:
        raise ValueError("research_context_version")
    model = {
        "research-run/1.0.0": ResearchRunContext,
        "research-run/2.0.0": TaskRunContext,
    }.get(raw.get("schema_version"))
    if model is None:
        raise ValueError("research_context_version")
    return validate(model, value)


def project_request(visible, *, run_id, task, policy_ref, catalog_ref, selected_components=None):
    """Deterministic projection, rechecked again by the original input owner on save."""
    from packages.dataset_adapters.multimodal.storage import MultimodalVisible

    visible = validate(MultimodalVisible, visible)
    task = validate(TaskIntent, task)
    available = {c.id: c for c in visible.components.components}
    selected = (
        tuple(
            c.id
            for c in visible.components.components
            if c.modality in task.allowed_modalities and c.status != "withheld"
        )
        if selected_components is None
        else tuple(selected_components)
    )
    if any(
        id not in available
        or available[id].modality not in task.allowed_modalities
        or available[id].status == "withheld"
        for id in selected
    ):
        raise ValueError("task_component_scope")
    c = visible.case
    return seal(
        TaskRunRequest,
        scope=dict(
            case_code=c.case_code,
            encounter_code=c.encounter_code,
            patient_hash=c.patient_hash,
            laterality=c.laterality,
        ),
        lock=dict(run_id=run_id, catalog_ref=catalog_ref),
        task=task,
        policy_ref=policy_ref,
        input_checksum=visible.components.checksum,
        visible_checksum=visible.checksum,
        artifacts=tuple(
            dict(
                asset_code=a.asset_code,
                checksum=a.checksum,
                artifact_type_code=a.artifact_type_code,
                media_type=a.media_type,
            )
            for a in c.artifacts
        ),
        selected_components=selected,
        cutoff_visit=visible.components.cutoff_visit,
    )


def check_current_inputs(context, reader, *, include_components=False):
    """No external I/O inside the SQL transaction; the original owner checks objects."""
    from packages.dataset_adapters.service import DatasetPreparationService

    context = validate(TaskRunContext, context)
    if not isinstance(reader, DatasetPreparationService):
        raise ValueError("task_original_input_owner_required")
    req = context.request
    if include_components:
        visible, stored = reader.read_visible_components(req.scope.case_code)
    else:
        visible = reader.read_visible(req.scope.case_code)
    expected = project_request(
        visible,
        run_id=context.run_id,
        task=req.task,
        policy_ref=req.policy_ref,
        catalog_ref=req.lock.catalog_ref,
        selected_components=req.selected_components,
    )
    if expected != req:
        raise ValueError("task_current_input_binding")
    # This return value is local to one synchronous owner call, not a persisted
    # qualification or a cache that can authorize a later request.
    return (visible, stored) if include_components else visible


class TaskRuns:
    """Composition port: no model/catalog installation is required to create a run."""

    def __init__(self, *, runtime, inputs):
        self.runtime, self.inputs = runtime, inputs

    def create(self, *, case_code, key, task, policy_ref, catalog_ref, synthetic):
        from db.runtime.repository.case_pipeline import identity

        if type(key) is not str or not 1 <= len(key) <= 128:
            raise ValueError("research_key_invalid")
        run_id = identity("RUN", "task-research:" + key)
        visible = self.inputs.read_visible(case_code)
        req = project_request(
            visible, run_id=run_id, task=task, policy_ref=policy_ref, catalog_ref=catalog_ref
        )
        context = seal(
            TaskRunContext, request=req, synthetic=synthetic, scope_checksum=digest(req.scope)
        )
        return self.runtime.create_task_run(context, key, inputs=self.inputs)
