"""06C deployment-only plans. Shared contracts and tool master data remain unchanged."""

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import Field, JsonValue, model_validator

from packages.grounding.models import (
    ArtifactConstraint,
    Checksum,
    PortRef,
    Ref,
    Token,
    Value,
    Version,
    canonical,
    check_seal,
    unique,
)
from packages.grounding.query_models import CaseScope, QueryRequest, QueryResult
from packages.integrations.medgraph.models import (
    DeploymentManifest,
    ManifestResource,
    ReviewedDeploymentManifest,
)
from packages.tool_admission.models import (
    AdmissionResult,
    AdmissionSelection,
    DeploymentProfile,
    InputRecord,
    ResearchRequest,
    Text,
)


class CompileError(ValueError):
    """Stable technical/selection reason; never a medical finding."""


class FormalContext(Value):
    kind: Literal["formal"] = "formal"
    request: QueryRequest
    query_result: QueryResult
    selection: AdmissionSelection
    admission: AdmissionResult


class ResearchContext(Value):
    kind: Literal["research"] = "research"
    request: ResearchRequest
    admission: AdmissionResult


Context = Annotated[FormalContext | ResearchContext, Field(discriminator="kind")]


class AssetSource(Value):
    kind: Literal["artifact"] = "artifact"
    asset_code: Text


class ProducerSource(Value):
    kind: Literal["producer"] = "producer"
    candidate_ref: Checksum
    port: PortRef


Source = Annotated[AssetSource | ProducerSource, Field(discriminator="kind")]


class InputChoice(Value):
    port: PortRef
    sources: tuple[Source, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def coherent(self):
        unique(self.sources)
        return self


class NodeChoice(Value):
    candidate_ref: Checksum
    inputs: tuple[InputChoice, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def coherent(self):
        unique(self.inputs, lambda i: i.port)
        return self


class WorkflowChoice(Value):
    version: Literal["explicit/1.0.0"] = "explicit/1.0.0"
    roots: tuple[Checksum, ...] = Field(max_length=128)
    nodes: tuple[NodeChoice, ...] = Field(max_length=512)
    tie_break: Literal["caller_explicit_no_medical_ranking"] = "caller_explicit_no_medical_ranking"

    @model_validator(mode="after")
    def coherent(self):
        unique(self.roots)
        unique(self.nodes, lambda n: n.candidate_ref)
        return self


class CompilerConfig(Value):
    version: Version
    checksum: Checksum
    baseline_only: Literal[True] = True
    max_steps: int = Field(ge=1, le=512)
    max_plan_bytes: int = Field(ge=1024, le=16777216)
    timeout_seconds: int = Field(ge=1, le=86400)
    memory_bytes: int = Field(ge=1048576, le=1099511627776)
    cpu_count: int = Field(ge=1, le=128)
    max_output_bytes: int = Field(ge=1, le=1099511627776)
    network: Literal["deny"] = "deny"
    environment_variables: tuple[Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")], ...] = ()
    failure_policy: Literal["stop_descendants_no_retry"] = "stop_descendants_no_retry"
    defaults: Literal["explicit_parameters_only"] = "explicit_parameters_only"

    @model_validator(mode="after")
    def coherent(self):
        unique(self.environment_variables)
        if tuple(sorted(self.environment_variables)) != self.environment_variables:
            raise ValueError("canonical_environment_required")
        check_seal(self)
        return self


class TrustedResearchCompilerConfig(CompilerConfig):
    """Explicit weak-isolation demand; never accepted by the old CPU service."""

    version: Literal["trusted-local-compiler/1.0.0"] = "trusted-local-compiler/1.0.0"
    network: Literal["not_guaranteed"] = "not_guaranteed"


def compiler_config(value):
    from packages.grounding.models import validate

    version = value.get("version") if isinstance(value, dict) else value.version
    model = (
        TrustedResearchCompilerConfig
        if version == "trusted-local-compiler/1.0.0"
        else CompilerConfig
    )
    return validate(model, value)


class OutputSpec(Value):
    port: PortRef
    filename: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,95}$")]
    qc_port: Literal["technical_output_qc/1.0.0"] = "technical_output_qc/1.0.0"
    qc_policy_ref: Ref


class CommandAllowlist(Value):
    """Controlled owner declares exact command, not a caller's approved boolean."""

    version: Version
    checksum: Checksum
    source_ref: Ref
    manifest_checksum: Checksum
    profile_checksum: Checksum
    compiler_config_checksum: Checksum
    synthetic_only: bool
    executable_uri: Text
    executable_checksum: Checksum
    entrypoint_uri: Text
    entrypoint_checksum: Checksum
    resources: tuple[ManifestResource, ...] = Field(min_length=1, max_length=128)
    argv: tuple[Text, ...] = Field(min_length=2, max_length=128)
    shell: Literal[False] = False
    parameter_schema_version: Version
    parameter_schema_checksum: Checksum
    outputs: tuple[OutputSpec, ...] = Field(min_length=1, max_length=64)
    optional_absence: Literal["omit_token_no_value"] = "omit_token_no_value"

    @model_validator(mode="after")
    def coherent(self):
        unique(self.resources, lambda r: r.resource_key)
        unique(self.outputs, lambda p: p.port)
        if len({o.filename for o in self.outputs}) != len(self.outputs):
            raise CompileError("output_path_collision")
        if self.source_ref.kind != "source_version":
            raise ValueError("allowlist_source_required")
        if any(o.qc_policy_ref.kind != "policy" for o in self.outputs):
            raise ValueError("qc_policy_required")
        check_seal(self)
        return self


class AllowlistReader(Protocol):
    def read_allowlist(
        self, manifest: DeploymentManifest, profile: DeploymentProfile, config: CompilerConfig
    ) -> CommandAllowlist: ...


class BoundInput(Value):
    port: PortRef
    constraint: ArtifactConstraint
    sources: tuple[Source, ...]
    logical_uris: tuple[Text, ...]
    artifact_checksums: tuple[Checksum, ...]
    optional_absence: Literal["not_absent", "omit_token_no_value"]
    qc_before_consume: tuple[Text, ...]


class PlannedOutput(Value):
    port: PortRef
    constraint: ArtifactConstraint
    logical_uri: Text
    producer_step: Token
    exists: Literal[False] = False
    bytes_checksum: None = None
    initial_qc_status: Literal["PENDING"] = "PENDING"
    qc_port: Literal["technical_output_qc/1.0.0"]
    qc_policy_ref: Ref
    downstream_condition: Literal["runtime_bytes_scope_format_and_required_qc"] = (
        "runtime_bytes_scope_format_and_required_qc"
    )


class StepDeployment(Value):
    manifest: DeploymentManifest | ReviewedDeploymentManifest
    profile: DeploymentProfile
    allowlist: CommandAllowlist
    probe_checksum: Checksum
    resources_checksum: Checksum
    invocation_checksum: Checksum
    input_records: tuple[InputRecord, ...]


class PlannedStep(Value):
    step_id: Token
    candidate_ref: Checksum
    implementation_ref: Ref
    tool_ref: Ref
    dependencies: tuple[Token, ...]
    inputs: tuple[BoundInput, ...]
    outputs: tuple[PlannedOutput, ...]
    argv: tuple[Text, ...]
    shell: Literal[False] = False
    effective_parameters: dict[Token, JsonValue]
    deployment: StepDeployment
    timeout_seconds: int
    pending_dynamic_checks: tuple[Text, ...]


class PlanDocument(Value):
    schema_version: Literal["tool-workflow-plan/1.0.0"] = "tool-workflow-plan/1.0.0"
    checksum: Checksum
    compiler_version: Literal["06C.1.0.0"] = "06C.1.0.0"
    scope: CaseScope
    run_id: Text
    purpose: Literal["decisive_research", "debug", "shadow"]
    catalog_ref: Ref
    admission_checksum: Checksum
    context_checksum: Checksum
    selection: WorkflowChoice
    config: CompilerConfig | TrustedResearchCompilerConfig
    steps: tuple[PlannedStep, ...] = Field(min_length=1, max_length=512)
    execution_authorized: Literal[False] = False
    decisive_evidence: Literal[False] = False
    clinical_effect: Literal["NONE"] = "NONE"
    attempt_binding: Literal["06D.allocate_after_authorization"] = (
        "06D.allocate_after_authorization"
    )
    authorization_required: Literal["controlled_plan_scope_policy_and_current_revalidation"] = (
        "controlled_plan_scope_policy_and_current_revalidation"
    )
    risks: tuple[Text, ...]
    gaps: tuple[Text, ...]

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        unique(self.steps, lambda s: s.step_id)
        return self


class PlanBundle(Value):
    plan: PlanDocument
    context: Context


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Deeply immutable canonical bytes-as-text; accessors return fresh, detached values.

    Reconstructing this wrapper or its seal is not authorization. Only the service's
    validate_plan performs controlled revalidation against the retained context.
    """

    _document_json: str
    _context_json: str

    @property
    def checksum(self):
        return self.document.checksum

    @property
    def document(self):
        return PlanDocument.model_validate_json(self._document_json)

    def dry_run(self):
        return self.document.model_dump(mode="json")

    def dry_run_json(self):
        return canonical(self.document).decode()

    def to_bundle_json(self):
        """Persist the original revalidation context beside the reviewable plan."""
        import json

        return canonical(dict(plan=self.document, context=json.loads(self._context_json))).decode()

    @classmethod
    def from_bundle_json(cls, text):
        """Parse only; caller must still use ToolWorkflowCompiler.validate_plan."""
        from packages.grounding.models import digest

        bundle = PlanBundle.model_validate_json(text)
        if digest(bundle.context) != bundle.plan.context_checksum:
            raise CompileError("context_binding_mismatch")
        return cls(canonical(bundle.plan).decode(), canonical(bundle.context).decode())
