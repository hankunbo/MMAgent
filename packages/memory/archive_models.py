"""Historical case/branch references. Retrieval suggests a procedure, never a diagnosis."""

import json
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Token, Value
from packages.memory.models import ReadResult, Record, Source


class ArtifactSource(Source):
    owner: Literal["case_artifact"] = "case_artifact"
    kind: Literal["case_artifact"] = "case_artifact"
    version: Literal["case-artifact-reference/1.0.0"] = "case-artifact-reference/1.0.0"


class RunArchiveSource(Source):
    owner: Literal["runtime_archive"] = "runtime_archive"
    kind: Literal["runtime_archive"] = "runtime_archive"
    version: Literal["case-execution-reference/1.0.0"] = "case-execution-reference/1.0.0"


class WorkflowStep(Value):
    operation: Literal["read_input", "segment", "measure", "review", "report", "recover"]
    tool: Token
    version: Token
    implementation: str = Field(min_length=1, max_length=256)
    source_checksum: Token
    input_roles: tuple[Token, ...] = Field(max_length=16)
    output_roles: tuple[Token, ...] = Field(min_length=1, max_length=16)
    artifact_codes: tuple[Token, ...] = Field(max_length=32)


class ArchiveDetail(Value):
    type: Literal["case_snapshot", "branch_workflow", "execution_outcome"]
    target: Token
    modality: Token
    origin_code: Token
    modality_source: Token
    steps: tuple[WorkflowStep, ...] = Field(max_length=16)
    outcome: Literal["recorded", "success", "failure", "partial"]
    failure_reason: Token | None = None
    execution_snapshot: dict | None = None
    clinical_correctness: Literal["not_established"] = "not_established"
    reuse: Literal["procedure_hint_recompute_all_patient_values"] = (
        "procedure_hint_recompute_all_patient_values"
    )


class ArchiveRecord(Record):
    schema_version: Literal["research-memory/1.3.0"] = "research-memory/1.3.0"
    kind: Literal["graph_snapshot", "tool_experience", "recovery_experience"]
    sources: tuple[ArtifactSource | RunArchiveSource, ...] = Field(min_length=1, max_length=64)
    detail: ArchiveDetail

    @model_validator(mode="after")
    def bindings(self):
        if self.parents or any(
            (s.case_code, s.run_id) != (self.case_code, self.run_id) for s in self.sources
        ):
            raise ValueError("memory_scope")
        if len({s.object_code for s in self.sources}) != len(self.sources):
            raise ValueError("memory_duplicate_source")
        if any(not set(self.applicability) <= set(s.applicability) for s in self.sources):
            raise ValueError("memory_applicability")
        expected = {
            "graph_snapshot": "case_snapshot",
            "tool_experience": "branch_workflow",
            "recovery_experience": "execution_outcome",
        }
        if self.detail.type != expected[self.kind] or self.detail.origin_code not in {
            s.object_code for s in self.sources
        }:
            raise ValueError("memory_archive_binding")
        if any(
            not set(step.artifact_codes) <= {s.object_code for s in self.sources}
            for step in self.detail.steps
        ):
            raise ValueError("memory_archive_step_source")
        if self.kind != "recovery_experience" and self.nature != "source_fact":
            raise ValueError("memory_archive_nature")
        return self


class ArchiveReadResult(ReadResult):
    record: ArchiveRecord


class ExtendedArchiveRecord(ArchiveRecord):
    """Explicit larger archive; old 1.3 records and schemas retain their limits."""

    schema_version: Literal["research-memory/1.4.0"] = "research-memory/1.4.0"
    sources: tuple[ArtifactSource | RunArchiveSource, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def bounded_bytes(self):
        # Mirrors the original JSONB payload guard with a versioned larger limit.
        # Source artifacts still pass the original bounded owner/object readers.
        if len(json.dumps(self.model_dump(mode="json"), ensure_ascii=False).encode()) > 4194304:
            raise ValueError("memory_archive_payload_capacity")
        return self


class ExtendedArchiveReadResult(ReadResult):
    record: ExtendedArchiveRecord


class ArchiveSourceCapacityError(ValueError):
    def __init__(self, sources):
        from packages.grounding.models import digest

        self.required_sources = len(sources)
        self.maximum_sources = 256
        self.source_set_checksum = digest(sources)
        super().__init__("memory_multimodal_source_capacity")


def archive_record_class(sources):
    """Stable old projection at <=64; never truncate a complete owner source set."""
    if len(sources) > 256:
        raise ArchiveSourceCapacityError(sources)
    return ExtendedArchiveRecord if len(sources) > 64 else ArchiveRecord
