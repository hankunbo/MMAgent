"""11B offline discovery values; separate from v1 jobs and 11A receipts."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Code, Token, Value

SourceKind = Literal["local_contract", "official_package", "official_model", "paper_code", "lead"]
SOURCE_ORDER = ("local_contract", "official_package", "official_model", "paper_code", "lead")
DIMENSIONS = (
    "source_trust",
    "contract_completeness",
    "task_match",
    "reproducibility",
    "dependency_simplicity",
    "public_evaluation",
    "resource_cost",
)


class DiscoveryPolicy(Value):
    schema_version: Literal["tool-discovery-policy/1.0.0"] = "tool-discovery-policy/1.0.0"
    baseline_only: Literal[True] = True
    max_records: int = Field(default=32, ge=1, le=128)
    max_sources: int = Field(default=8, ge=1, le=16)
    max_seconds: int = Field(default=10, ge=1, le=30)
    commit_margin_seconds: int = Field(default=2, ge=1, le=10)
    max_metadata_bytes: int = Field(default=262144, ge=1024, le=524288)
    top_k: int = Field(default=5, ge=1, le=32)
    max_memory_mb: int = Field(default=4096, ge=1, le=65536)
    allow_gpu: Literal[False] = False
    # Illustrative engineering policy, never a fitted/medical conclusion.
    weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.5)

    @model_validator(mode="after")
    def valid_weights(self):
        if len(self.weights) != len(DIMENSIONS) or any(not 0 <= w <= 10 for w in self.weights):
            raise ValueError("discovery_weights")
        if sum(self.weights) == 0:
            raise ValueError("discovery_weights")
        return self


class Semantics(Value):
    capability: dict
    inputs: tuple[dict, ...]
    output: dict
    task_type: Token
    intended_use: Literal["research_only"] = "research_only"


class DiscoveryQuery(Value):
    schema_version: Literal["tool-discovery-query/1.0.0"] = "tool-discovery-query/1.0.0"
    application_checksum: Checksum
    requirement: dict
    purpose: Token
    contracts: tuple[Semantics, ...] = Field(min_length=1, max_length=128)
    max_memory_mb: int
    gpu_allowed: Literal[False] = False
    missing: tuple[Token, ...] = ("evaluation_dataset", "threshold_policy")


class Provenance(Value):
    """Missing evidence is null; digests refer to fixture bytes in this phase."""

    schema_version: Literal["tool-candidate-provenance/1.0.0"] = "tool-candidate-provenance/1.0.0"
    synthetic: Literal[True] = True
    record_id: Token
    source_kind: SourceKind
    source_url: str = Field(max_length=2048)
    identity_ref: Token | None = None
    repository: str | None = Field(default=None, max_length=2048)
    repository_owner: Token | None = None
    model_id: Token | None = None
    revision: Token | None = None
    revision_kind: Literal["commit", "immutable_version", "branch"] | None = None
    claimed_official: bool = False
    paper_doi: str | None = Field(default=None, max_length=1024)
    code_license: str | None = Field(default=None, max_length=256)
    weights_license: str | None = Field(default=None, max_length=256)
    code_checksum: Checksum | None = None
    weights_checksum: Checksum | None = None
    weights_required: bool = True
    model_card: str | None = Field(default=None, max_length=4096)
    framework: Token | None = None
    dependencies: tuple[str, ...] = Field(default=(), max_length=64)
    inference_entry: str | None = Field(default=None, max_length=256)
    semantics: Semantics | None = None
    input_shape: tuple[str, ...] = ()
    output_shape: tuple[str, ...] = ()
    preprocessing: str | None = Field(default=None, max_length=4096)
    postprocessing: str | None = Field(default=None, max_length=4096)
    memory_mb: int | None = Field(default=None, ge=1)
    requires_gpu: bool | None = None
    uploads_patient_data: bool = False
    arbitrary_remote_code: bool = False
    training_data: str | None = Field(default=None, max_length=4096)
    evaluation_data: str | None = Field(default=None, max_length=4096)
    performance_claim: str | None = Field(default=None, max_length=4096)
    applicability: str | None = Field(default=None, max_length=4096)
    limitations: str | None = Field(default=None, max_length=4096)


class RegisteredIdentity(Value):
    """Installed fixture registry evidence, independent of a record's official claim."""

    identity_ref: Token
    repository: str
    repository_owner: Token
    model_id: Token
    revision: Token
    code_checksum: Checksum
    weights_checksum: Checksum | None
    source_urls: tuple[str, ...] = Field(min_length=1, max_length=32)
    official: bool
    synthetic: Literal[True] = True


class CandidateObservation(Value):
    identity: Checksum
    records: tuple[Provenance, ...] = Field(min_length=1, max_length=128)
    identity_evidence: tuple[RegisteredIdentity, ...] = Field(max_length=128)
    disposition: Literal["discovered", "incomplete", "quarantined"]
    missing: tuple[Token, ...]
    rejection_codes: tuple[Token, ...]
    dimensions: dict[str, float | None]
    score: float
    source_priority: int
    machine_gate: None = None
    onboarding_job: None = None


class DiscoveryResult(Value):
    schema_version: Literal["tool-discovery-result/1.0.0"] = "tool-discovery-result/1.0.0"
    task_id: Code
    plan_checksum: Checksum
    query: DiscoveryQuery
    policy: DiscoveryPolicy
    status: Literal[
        "discovered", "no_candidates", "incomplete", "technical_failure", "budget_exhausted"
    ]
    candidates: tuple[CandidateObservation, ...] = Field(max_length=128)
    selected: tuple[Checksum, ...] = Field(max_length=32)
    sources: tuple[dict, ...] = Field(max_length=16)
    records_examined: int = Field(ge=0, le=128)
    elapsed_seconds: float = Field(ge=0)
    memory: dict
    completed: Literal[False] = False
    available_next_run: Literal[False] = False
    historical_read_not_authorization: Literal[True] = True
    live_domain_acceptance: Literal["live_domain_acceptance_not_enabled"] = (
        "live_domain_acceptance_not_enabled"
    )


class SourceBinding(Value):
    name: Token
    kind: SourceKind
    fingerprint: Checksum


class DiscoveryPlan(Value):
    schema_version: Literal["tool-discovery-plan/1.0.0"] = "tool-discovery-plan/1.0.0"
    task_id: Code
    application_checksum: Checksum
    query: DiscoveryQuery
    policy: DiscoveryPolicy
    sources: tuple[SourceBinding, ...] = Field(max_length=16)
    claim_key: Token
    ack_key: Token
    worker: Token
    lease_token: Token
    ack_command_checksum: Checksum
    ack_event_id: Code
    ack_idempotency_key: Code
