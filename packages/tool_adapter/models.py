"""11C synthetic stage contracts, separate from formal ToolOnboardingJob v1."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import (
    ArtifactConstraint,
    Checksum,
    PortRef,
    Token,
    Value,
    Version,
    digest,
    validate,
)
from packages.tool_discovery.models_v11 import (
    CandidateObservation,
    DiscoveryQuery,
    DiscoveryResult,
    Provenance,
)
from packages.tool_gap_queue.models import Application


class AdapterPolicy(Value):
    schema_version: Literal["synthetic-adapter-policy/1.0.0"] = "synthetic-adapter-policy/1.0.0"
    baseline_only: Literal[True] = True
    code: Literal["POLICY:SYNTHETIC_ADAPTER"] = "POLICY:SYNTHETIC_ADAPTER"
    version: Version = "1.0.0"
    max_seconds: int = Field(default=10, ge=1, le=30)
    lease_seconds: int = Field(default=30, ge=2, le=60)
    max_memory_mb: int = Field(default=512, ge=16, le=4096)
    max_attempts: Literal[1] = 1
    real_benchmark: None = None
    intended_use: Literal["research_only"] = "research_only"


class ModelConfiguration(Value):
    device: Literal["cpu"] = "cpu"
    batch: int = Field(default=1, ge=1, le=8)
    normalization: Token
    resize_crop: Token
    class_semantics: tuple[Token, ...] = Field(min_length=1, max_length=64)
    coordinate_restoration: Literal["identity"] = "identity"


class AlgorithmConfiguration(Value):
    formula_version: Version
    units: Token
    numeric_range: tuple[float, float]
    deterministic: Literal[True] = True

    @model_validator(mode="after")
    def ordered(self):
        if self.numeric_range[0] > self.numeric_range[1]:
            raise ValueError("adapter_numeric_range")
        return self


class AdapterManifest(Value):
    schema_version: Literal["synthetic-adapter-manifest/1.0.0"] = "synthetic-adapter-manifest/1.0.0"
    synthetic: Literal[True] = True
    adapter_id: Token
    version: Version
    kind: Literal["model", "algorithm"]
    candidate: Provenance
    output_port: PortRef
    model: ModelConfiguration | None = None
    algorithm: AlgorithmConfiguration | None = None
    # Metadata only: no command, path, import, or third-party executable is accepted.
    execution_entry: Literal["original_06_not_installed"] = "original_06_not_installed"
    argv: tuple[Literal["fake-metadata-only"], ...] = ("fake-metadata-only",)
    shell: Literal[False] = False
    network: Literal["disabled"] = "disabled"
    timeout_seconds: int = Field(default=2, ge=1, le=10)
    cpu_count: Literal[1] = 1
    gpu_count: Literal[0] = 0
    memory_mb: int = Field(default=128, ge=16, le=4096)
    staging: Literal["no_files_created"] = "no_files_created"
    output_collection: Literal["synthetic_metadata_only"] = "synthetic_metadata_only"
    cleanup: Literal["no_process_or_files"] = "no_process_or_files"
    type_conversion: Literal["identity"] = "identity"
    modality_conversion: Literal["identity"] = "identity"
    coordinate_conversion: Literal["identity"] = "identity"
    laterality_conversion: Literal["identity"] = "identity"
    error_protocol: Literal["synthetic-adapter-errors/1.0.0"] = "synthetic-adapter-errors/1.0.0"

    @model_validator(mode="after")
    def configuration(self):
        if self.output_port.direction != "output":
            raise ValueError("adapter_output_direction")
        if self.candidate.semantics is None or not self.candidate.semantics.complete:
            raise ValueError("adapter_complete_contract_required")
        validate(ArtifactConstraint, self.candidate.semantics.output)
        if (self.kind == "model") != (self.model is not None) or (self.kind == "algorithm") != (
            self.algorithm is not None
        ):
            raise ValueError("adapter_kind_configuration")
        return self


class SyntheticGrant(Value):
    """Installed stage authority, never an argument to consume or inferred from selected."""

    schema_version: Literal["synthetic-adapter-grant/1.0.0"] = "synthetic-adapter-grant/1.0.0"
    synthetic: Literal[True] = True
    run_id: Token
    task_id: Token
    stage_key: Token
    worker: Token
    application_checksum: Checksum
    discovery_checksum: Checksum
    candidate_identity: Checksum
    manifest_checksum: Checksum
    policy_checksum: Checksum
    expires_at: datetime
    purpose: Literal["synthetic_adapter_contract_only"] = "synthetic_adapter_contract_only"
    budget: Literal[1] = 1

    @model_validator(mode="after")
    def timezone(self):
        if self.expires_at.tzinfo is None:
            raise ValueError("adapter_grant_timezone_required")
        return self


class AdapterPlan(Value):
    schema_version: Literal["synthetic-adapter-plan/1.0.0"] = "synthetic-adapter-plan/1.0.0"
    task_id: Token
    application: Application
    discovery: DiscoveryResult
    query: DiscoveryQuery
    candidate: CandidateObservation
    manifest: AdapterManifest
    policy: AdapterPolicy
    grant: SyntheticGrant
    fake_protocol_checksum: Checksum
    lease_token: Token
    runtime_context: dict

    @model_validator(mode="after")
    def bindings(self):
        g = self.grant
        if (
            self.task_id != g.task_id
            or self.discovery.task_id != g.task_id
            or self.application.run_id != g.run_id
            or digest(self.application) != g.application_checksum
            or digest(self.discovery) != g.discovery_checksum
            or self.candidate.identity != g.candidate_identity
            or digest(self.manifest) != g.manifest_checksum
            or digest(self.policy) != g.policy_checksum
            or self.query != self.discovery.query
        ):
            raise ValueError("adapter_plan_binding")
        return self


GateOutcome = Literal[
    "passed", "failed", "incompatible", "missing", "not_executed", "technical_failure", "unknown"
]


class GateEvidence(Value):
    schema_version: Literal["synthetic-adapter-gate/1.0.0"] = "synthetic-adapter-gate/1.0.0"
    synthetic: Literal[True] = True
    stage: Literal["contract", "fake_smoke", "sandbox", "benchmark", "policy"]
    outcome: GateOutcome
    policy_checksum: Checksum
    input_checksum: Checksum
    evidence_checksum: Checksum
    evidence: dict
    reasons: tuple[Token, ...]

    @model_validator(mode="after")
    def evidence_binding(self):
        if digest(self.evidence) != self.evidence_checksum:
            raise ValueError("adapter_gate_evidence_binding")
        return self


class AdapterResult(Value):
    schema_version: Literal["synthetic-adapter-result/1.0.0"] = "synthetic-adapter-result/1.0.0"
    task_id: Token
    plan_checksum: Checksum
    synthetic: Literal[True] = True
    outcome: Literal[
        "failed", "incompatible", "missing", "not_executed", "technical_failure", "unknown"
    ]
    candidate_disposition: Literal["discovered", "incomplete", "quarantined"]
    catalog_status: Literal["candidate", "quarantined", "failed"]
    catalog_written: Literal[False] = False
    extension_job_status: Literal["REQUIREMENT_DOCUMENTED", "FAILED"]
    formal_job_created: Literal[False] = False
    validation_status: Literal["unvalidated"] = "unvalidated"
    intended_use: Literal["research_only"] = "research_only"
    gates: tuple[GateEvidence, ...] = Field(min_length=1, max_length=5)
    missing: tuple[Token, ...]
    completed: Literal[False] = False
    available_next_run: Literal[False] = False
    live_domain_acceptance: Literal["live_domain_acceptance_not_enabled"] = (
        "live_domain_acceptance_not_enabled"
    )

    @model_validator(mode="after")
    def state_domains(self):
        failed = self.outcome in {"failed", "incompatible"}
        if (self.extension_job_status == "FAILED") != failed:
            raise ValueError("adapter_job_outcome")
        if (self.catalog_status in {"quarantined", "failed"}) != failed:
            raise ValueError("adapter_catalog_outcome")
        return self
