"""Read-only history of real or synthetic technical outcomes, never live acceptance."""

from packages.grounding.models import digest, validate
from packages.memory.archive_models import ArchiveRecord, RunArchiveSource


class ExecutionArchiveSources:
    def __init__(self, runtime):
        self.runtime = runtime

    def snapshot(self, run_id):
        value = self.runtime.read_memory_execution(run_id)
        if (
            value["lifecycle_phase"] not in {"FINISHED", "REJECTED"}
            and not value["preparation_rejected"]
        ):
            raise ValueError("memory_source_missing")
        if value.get("report") is None:
            raise ValueError("memory_source_missing")
        return value

    def reference(self, value):
        return RunArchiveSource(
            case_code=value["case_code"],
            run_id=value["run_id"],
            key=value["run_id"],
            object_code=value["run_id"],
            checksum=digest(value),
            object_checksum=digest(value),
            dependencies=(digest(["runtime_archive", value["run_id"]]),),
            applicability=("USE:RESEARCH_WORKFLOW",),
            nature="source_fact",
        )

    def prepare(self, *, key, run_id):
        value = self.snapshot(run_id)
        status = value["report"].get("technical_status")
        outcome = (
            "success"
            if status == "SUCCEEDED"
            else "failure"
            if status in {"FAILED", "REJECTED", "CANCELLED"} or value["preparation_rejected"]
            else "partial"
        )
        return ArchiveRecord(
            key=key,
            kind="recovery_experience",
            case_code=value["case_code"],
            run_id=run_id,
            applicability=("USE:RESEARCH_WORKFLOW",),
            sources=(self.reference(value),),
            nature="technical_failure" if outcome == "failure" else "source_fact",
            detail=dict(
                type="execution_outcome",
                target="tool_execution",
                modality="NOT_APPLICABLE",
                modality_source="runtime_technical_history",
                origin_code=run_id,
                steps=(),
                outcome=outcome,
                failure_reason=status if outcome == "failure" else None,
                execution_snapshot=value,
            ),
        )

    def verify(self, source):
        if self.reference(self.snapshot(source.run_id)) != source:
            raise ValueError("memory_source_binding")

    def verify_record(self, record):
        r = validate(ArchiveRecord, record)
        if self.prepare(key=r.key, run_id=r.run_id) != r:
            raise ValueError("memory_projection_binding")


class ArchiveOwners:
    def __init__(self, artifacts, executions):
        self.artifacts, self.executions = artifacts, executions

    def verify(self, source):
        owner = {"case_artifact": self.artifacts, "runtime_archive": self.executions}.get(
            source.owner
        )
        if owner is None:
            raise ValueError("memory_source_port_unavailable")
        owner.verify(source)

    def verify_record(self, record):
        (self.executions if record.kind == "recovery_experience" else self.artifacts).verify_record(
            record
        )
