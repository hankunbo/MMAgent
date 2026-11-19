"""Opt-in task-success qualification. Old 1.0 schemas and medical success stay exact."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value, digest, validate
from packages.memory_review.models import (
    Binding,
    Conditions,
    Opinion,
    Policy,
    Publication,
    Query,
    ReviewInput,
)
from packages.research_runtime.task_models import TaskCompletion


class TaskBinding(Binding):
    schema_version: Literal["memory-use-binding/1.1.0"] = "memory-use-binding/1.1.0"
    group: Literal["success"] = "success"
    success_semantics: Literal["research-task-completion/1.0.0"] = "research-task-completion/1.0.0"
    task_profile: Checksum
    proofs: tuple[TaskCompletion, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def task_identity(self):
        runs = [p.definition.run_id for p in self.proofs]
        if len(set(runs)) != len(runs) or runs != sorted(runs):
            raise ValueError("memory_task_proof_set")
        if any(
            p.definition.profile != self.task_profile or p.definition.goal != self.conditions.goal
            for p in self.proofs
        ):
            raise ValueError("memory_task_profile")
        return self


class TaskPolicy(Policy):
    schema_version: Literal["memory-publication-policy/1.1.0"] = "memory-publication-policy/1.1.0"
    task_profile: Checksum
    success_semantics: Literal["research-task-completion/1.0.0"] = "research-task-completion/1.0.0"


class TaskOpinion(Opinion):
    schema_version: Literal["memory-opinion/1.1.0"] = "memory-opinion/1.1.0"
    binding: TaskBinding


class TaskPublication(Publication):
    schema_version: Literal["memory-publication/1.1.0"] = "memory-publication/1.1.0"
    binding: TaskBinding


class TaskQuery(Query):
    schema_version: Literal["memory-query/1.1.0"] = "memory-query/1.1.0"
    task_profile: Checksum
    success_semantics: Literal["research-task-completion/1.0.0"] = "research-task-completion/1.0.0"


class TaskReviewInput(Value):
    """Prompt projection: references/conditions only, never full H definitions or outputs."""

    schema_version: Literal["memory-agent-input/1.1.0"] = "memory-agent-input/1.1.0"
    role: Literal["expert"] = "expert"
    scope: dict
    target: Token
    target_checksum: Checksum
    content_version: Literal[
        "research-memory/1.0.0", "research-memory/1.1.0", "research-memory/1.2.0"
    ]
    binding_checksum: Checksum
    conditions: Conditions
    task_profile: Checksum
    task_proofs: tuple[Checksum, ...]
    sources: tuple[Checksum, ...]
    instruction: Literal["candidate_advice_only_no_publication"] = (
        "candidate_advice_only_no_publication"
    )

    @classmethod
    def project(cls, *, scope, binding, sources):
        binding = validate(TaskBinding, binding)
        return cls(
            scope=scope,
            target=binding.target,
            target_checksum=binding.target_checksum,
            content_version=binding.content_version,
            binding_checksum=digest(binding),
            conditions=binding.conditions,
            task_profile=binding.task_profile,
            task_proofs=tuple(p.checksum for p in binding.proofs),
            sources=sources,
        )


TASK_MODELS = {
    Binding: TaskBinding,
    Policy: TaskPolicy,
    Opinion: TaskOpinion,
    Publication: TaskPublication,
    Query: TaskQuery,
    ReviewInput: TaskReviewInput,
}


def qualification(model, value):
    version = value.get("schema_version") if isinstance(value, dict) else value.schema_version
    task = TASK_MODELS[model]
    selected = task if version == task.model_fields["schema_version"].default else model
    return validate(selected, value)
