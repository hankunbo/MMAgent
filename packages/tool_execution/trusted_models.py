"""06E-TRUSTED-LOCAL-001. Deployment-owned research values, never v1 enums."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Ref, Value, unique
from packages.grounding.review_files import ReviewedFile as ReviewedFile

from .models import WorkflowResult

GPUUUID = Annotated[str, Field(pattern=r"^GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")]


class TrustedCapabilities(Value):
    version: Literal["trusted-local-capabilities/1.0.0"] = "trusted-local-capabilities/1.0.0"
    backend: Literal["trusted_local_gpu_research/1.0.0"] = "trusted_local_gpu_research/1.0.0"
    filesystem: Literal["admission_paths_only_no_process_confinement"] = (
        "admission_paths_only_no_process_confinement"
    )
    code_files: Literal["inventory_and_digest_checks_no_immutable_mount"] = (
        "inventory_and_digest_checks_no_immutable_mount"
    )
    network: Literal["not_guaranteed"] = "not_guaranteed"
    credentials: Literal["minimal_environment_not_file_secrecy"] = (
        "minimal_environment_not_file_secrecy"
    )
    devices: Literal["uuid_visibility_and_project_lock_not_device_isolation"] = (
        "uuid_visibility_and_project_lock_not_device_isolation"
    )
    memory: Literal["no_tree_ram_or_vram_hard_limit"] = "no_tree_ram_or_vram_hard_limit"
    cleanup: Literal["observed_process_group_only_escape_not_prevented"] = (
        "observed_process_group_only_escape_not_prevented"
    )
    files: Literal["per_file_limit_not_aggregate_disk_quota"] = (
        "per_file_limit_not_aggregate_disk_quota"
    )


class TrustedToolReview(Value):
    """Installed by deployment assembly, not accepted on the execute request port.

    Digests detect drift. They do not make code reviewed. review_record must explain
    provenance, inspected implementation/weights and the exact research scope.
    """

    version: Literal["trusted-tool-review/1.0.0", "trusted-tool-review/1.1.0"] = (
        "trusted-tool-review/1.0.0"
    )
    implementation_ref: Ref
    manifest_checksum: Checksum
    allowlist_checksum: Checksum
    review_record: ReviewedFile
    environment_root: str
    executable: ReviewedFile
    entrypoint: ReviewedFile
    files: tuple[ReviewedFile, ...] = Field(min_length=1, max_length=16384)
    # Every code/data directory whose inventory is pinned, including environment.
    # Symlinks are rejected: independently prepare a resolved deployment closure.
    inventory_roots: tuple[str, ...] = Field(min_length=1, max_length=16)
    working_directory: Literal["attempt/root"] = "attempt/root"
    research_only: Literal[True] = True
    decisive_evidence: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        if self.implementation_ref.kind != "implementation":
            raise ValueError("review_implementation_required")
        if any(f.byte_size == 0 for f in (self.review_record, self.executable, self.entrypoint)):
            raise ValueError("review_record_and_program_must_not_be_empty")
        if self.version == "trusted-tool-review/1.0.0" and any(
            f.byte_size == 0 for f in self.files
        ):
            raise ValueError("empty_closure_file_requires_review_1_1")
        unique(self.files, lambda f: f.path)
        unique(self.inventory_roots)
        by_path = {f.path: f for f in self.files}
        if any(by_path.get(f.path) != f for f in (self.executable, self.entrypoint)):
            raise ValueError("review_entrypoint_and_runtime_closure_required")
        return self


class TrustedLocalPolicy(Value):
    # 1.0: empty device + free VRAM admission. 1.1: project-serial/shared GPU;
    # occupancy and VRAM are observations, not pre-launch rejection thresholds.
    # The unchanged default preserves old policy serialization and semantics.
    version: Literal["trusted-local-policy/1.0.0", "trusted-local-policy/1.1.0"] = (
        "trusted-local-policy/1.0.0"
    )
    decision: Literal["06E-TRUSTED-LOCAL-001"] = "06E-TRUSTED-LOCAL-001"
    capabilities: TrustedCapabilities = Field(default_factory=TrustedCapabilities)
    reference: Ref
    purpose: Literal["debug", "shadow"]
    research_only: Literal[True] = True
    decisive_evidence: Literal[False] = False
    allowed_plan_checksums: tuple[Checksum, ...] = Field(min_length=1, max_length=512)
    reviews: tuple[TrustedToolReview, ...] = Field(min_length=1, max_length=64)
    gpu_pool: tuple[GPUUUID, ...] = Field(min_length=1, max_length=4)
    selected_gpu: GPUUUID
    max_concurrent: Literal[1] = 1
    per_gpu_concurrent: Literal[1] = 1
    cpu_count: int = Field(ge=1, le=128)
    ram_budget_bytes: int = Field(ge=1048576, le=1099511627776)
    vram_budget_mib: int = Field(ge=1, le=1048576)
    timeout_seconds: int = Field(ge=1, le=3600)
    log_bytes: int = Field(default=65536, ge=1024, le=1048576)
    max_resource_bytes: int = Field(default=8589934592, ge=1, le=8589934592)
    closure_timeout_seconds: int = Field(default=60, ge=1, le=600)
    cleanup_seconds: int = Field(default=5, ge=1, le=10)
    nvidia_smi: ReviewedFile

    @model_validator(mode="after")
    def coherent(self):
        if self.reference.kind != "policy" or self.selected_gpu not in self.gpu_pool:
            raise ValueError("trusted_policy_binding_invalid")
        unique(self.gpu_pool)
        unique(self.allowed_plan_checksums)
        unique(self.reviews, lambda r: r.implementation_ref)
        return self


class TrustedTechnicalOutcome(Value):
    schema_version: Literal["trusted-local-outcome/1.0.0"] = "trusted-local-outcome/1.0.0"
    policy_checksum: Checksum
    capabilities: TrustedCapabilities
    state: Literal["recorded", "technical_rejection", "uncertain"]
    reason: str | None = Field(default=None, max_length=128)
    workflow: WorkflowResult | None = None
    retry_execution: Literal[False] = False
    clinical_effect: Literal["NONE"] = "NONE"
    decisive_evidence: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        if (self.state == "recorded") != (self.workflow is not None):
            raise ValueError("technical_outcome_workflow_mismatch")
        if (self.state == "recorded") != (self.reason is None):
            raise ValueError("technical_outcome_reason_mismatch")
        if self.workflow is not None and (
            self.workflow.decision.policy_checksum != self.policy_checksum
            or self.workflow.decision.purpose not in {"debug", "shadow"}
        ):
            raise ValueError("technical_outcome_policy_mismatch")
        return self
