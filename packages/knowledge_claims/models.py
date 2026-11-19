"""04B5 internal, strict Draft extraction ports; independent of contracts v1."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_serializer,
    model_validator,
)

from packages.knowledge_index.models import Position

Text = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
Term = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Checksum = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
Version = Annotated[str, StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
    )


class Span(Model):
    coordinate_system: Literal["canonical_unicode"] = "canonical_unicode"
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("invalid_span")
        return self


class NumericEvidence(Model):
    value: Annotated[str, StringConstraints(pattern=r"^-?\d+(?:\.\d+)?$")]
    unit: Term | None
    comparator: Literal["<", "<=", ">", ">=", "=", "≤", "≥"] | None
    quote: Term
    span: Span


class DraftOutput(Model):
    """Extractive only: no paraphrase or latent reasoning accepted in version 1."""

    claim_type: Literal[
        "exam_requirement", "evidence_direction", "exclusion", "applicability", "validation", "stop"
    ]
    statement: Text
    target_proposition: Text
    polarity: Literal["POSITIVE", "NEGATIVE", "UNCERTAIN"]
    uncertainty: tuple[Term, ...] = Field(max_length=16)
    recommendation_strength: tuple[Term, ...] = Field(max_length=16)
    subject: Term
    population: tuple[Term, ...] = Field(max_length=16)
    disease_stage: tuple[Term, ...] = Field(max_length=16)
    condition: tuple[Term, ...] = Field(max_length=16)
    intervention: tuple[Term, ...] = Field(max_length=16)
    exam: tuple[Term, ...] = Field(max_length=16)
    modality: tuple[Term, ...] = Field(max_length=16)
    outcome: tuple[Term, ...] = Field(max_length=16)
    time: tuple[Term, ...] = Field(max_length=16)
    laterality: tuple[Term, ...] = Field(max_length=16)
    device: tuple[Term, ...] = Field(max_length=16)
    applicability: tuple[Term, ...] = Field(max_length=16)
    numbers: tuple[NumericEvidence, ...] = Field(max_length=32)
    quote: Text
    span: Span
    confidence: float = Field(ge=0, le=1)


class StructuredOutput(Model):
    schema_version: Literal["claim-output/1.0.0"] = "claim-output/1.0.0"
    candidates: tuple[DraftOutput, ...] = Field(max_length=8)


class Reference(Model):
    kind: Literal["source", "source_version", "source_document", "document_chunk"]
    code: Term | Annotated[str, StringConstraints(min_length=129, max_length=256)]
    version: Version
    checksum: Checksum


class BatchPolicy(Model):
    version: Literal["claim-batch/1.0.0"] = "claim-batch/1.0.0"
    max_requests: int = Field(ge=1, le=256)
    max_input_chars: int = Field(ge=1, le=512000)
    max_output_tokens: int = Field(ge=256, le=16000)
    total_seconds: int = Field(ge=1, le=3600)


class ValidationPolicyBinding(Model):
    version: Literal["claim-validation/2.0.0"]
    numeric_version: Literal["numeric-occurrence/1.0.0"]
    polarity_version: Literal["polarity-grammar/1.0.0"]
    checksum: Checksum


class N124ValidationPolicyBinding(ValidationPolicyBinding):
    version: Literal["claim-validation/3.0.0"]
    numeric_version: Literal["n124-document-metadata/1.0.0"]
    polarity_version: Literal["n124-exact-clause/1.0.0"]


class CandidateSourceBinding(Model):
    index_key: Checksum
    source: Reference
    source_version: Reference
    document: Reference
    chunk: Reference
    original_checksum: Checksum
    parse_checksum: Checksum
    chunk_checksum: Checksum
    parse_uri: Annotated[
        str,
        StringConstraints(pattern=r"^object://knowledge/objects/[A-Za-z0-9_-]+$", max_length=256),
    ]
    fidelity_policy: Literal["text-fidelity/2.0.0"]
    position: Position

    @model_validator(mode="after")
    def kinds(self):
        for field, kind in (
            ("source", "source"),
            ("source_version", "source_version"),
            ("document", "source_document"),
            ("chunk", "document_chunk"),
        ):
            if getattr(self, field).kind != kind:
                raise ValueError("source_binding")
        return self


class CandidateRequest(Model):
    schema_version: Literal["candidate-request/1.0.0"] = "candidate-request/1.0.0"
    request_key: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9:_-]{1,128}$")]
    method: Literal["RULE"] = "RULE"
    rule_version: Literal["exact-candidate/1.0.0"] = "exact-candidate/1.0.0"
    rule_config_checksum: Checksum
    source: CandidateSourceBinding
    candidate: DraftOutput
    display_zh: Text | None = None


class RuleExecution(Model):
    version: Literal["extractor-execution/1.0.0"] = "extractor-execution/1.0.0"
    implementation: Literal["registered_exact_candidate"] = "registered_exact_candidate"
    rule_version: Literal["exact-candidate/1.0.0"] = "exact-candidate/1.0.0"
    transport: Literal["local"] = "local"
    external_services_called: Literal[False] = False
    model_called: Literal[False] = False
    # Content origin is inherited from SourceVersion, independent of transport.


class N124RuleExecution(RuleExecution):
    version: Literal["extractor-execution/2.0.0"] = "extractor-execution/2.0.0"
    implementation: Literal["registered_nice_ng81_n124"] = "registered_nice_ng81_n124"
    rule_version: Literal["nice-ng81-n124/1.0.0"] = "nice-ng81-n124/1.0.0"


class CandidateSubmission(Model):
    version: Literal["candidate-submission/1.0.0"] = "candidate-submission/1.0.0"
    producer: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9:_-]{1,128}$")]
    request_key: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9:_-]{1,128}$")]
    request_checksum: Checksum
    source_binding_checksum: Checksum
    display_zh: Text | None = None


class ConfigurationSnapshot(Model):
    version: Literal[
        "claim-config/1.0.0", "claim-config/2.0.0", "claim-config/3.0.0", "claim-config/4.0.0"
    ] = "claim-config/1.0.0"
    validation_policy: ValidationPolicyBinding | N124ValidationPolicyBinding | None = None
    execution: RuleExecution | N124RuleExecution | None = None
    submission: CandidateSubmission | None = None
    method: Literal["RULE", "LLM"]
    extractor_version: Version
    extractor_config_checksum: Checksum
    code_checksum: Checksum
    model: Term | None
    model_version: Term | None
    base_url: str | None = Field(max_length=2048)
    api_key_env: str | None = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    endpoint: Literal["chat_completions"] | None
    timeout_seconds: int = Field(ge=1, le=300)
    max_retries: int = Field(ge=0, le=3)
    prompt_checksum: Checksum
    schema_checksum: Checksum
    budget: BatchPolicy

    @model_serializer(mode="wrap")
    def legacy_policy_bytes(self, handler):
        result = handler(self)
        for name in ("validation_policy", "execution", "submission"):
            if getattr(self, name) is None:
                result.pop(name, None)
        return result

    @model_validator(mode="after")
    def explicit_policy_generation(self):
        if (self.version != "claim-config/1.0.0") != (self.validation_policy is not None):
            raise ValueError("validation_policy_binding")
        if (self.version == "claim-config/4.0.0") != isinstance(
            self.validation_policy, N124ValidationPolicyBinding
        ):
            raise ValueError("validation_policy_version_binding")
        rule = self.version in {"claim-config/3.0.0", "claim-config/4.0.0"}
        if rule and (
            (self.version == "claim-config/4.0.0") != isinstance(self.execution, N124RuleExecution)
        ):
            raise ValueError("execution_version_binding")
        if rule != (self.execution is not None) or rule != (self.submission is not None):
            raise ValueError("execution_binding")
        if rule and (
            self.method != "RULE"
            or self.model is not None
            or self.model_version is not None
            or any(v is not None for v in (self.base_url, self.api_key_env, self.endpoint))
            or self.max_retries != 0
        ):
            raise ValueError("rule_execution_binding")
        if not rule and (self.model is None or self.model_version is None):
            raise ValueError("legacy_model_binding")
        return self


class Provenance(Model):
    method: Literal["RULE", "LLM"]
    extractor_version: Version
    prompt_version: Version
    prompt_checksum: Checksum
    schema_checksum: Checksum
    model: Term | None
    model_version: Term | None
    config_checksum: Checksum
    configuration: ConfigurationSnapshot
    input_checksum: Checksum
    output_checksum: Checksum

    @model_validator(mode="after")
    def config_binding(self):
        from packages.knowledge_index.models import digest

        cfg = self.configuration
        if self.config_checksum != digest(cfg) or (
            self.method,
            self.extractor_version,
            self.model,
            self.model_version,
            self.prompt_checksum,
            self.schema_checksum,
        ) != (
            cfg.method,
            cfg.extractor_version,
            cfg.model,
            cfg.model_version,
            cfg.prompt_checksum,
            cfg.schema_checksum,
        ):
            raise ValueError("configuration_binding")
        return self


class ClinicalClaimCandidate(DraftOutput):
    schema_version: Literal["clinical-claim-candidate/1.0.0"] = "clinical-claim-candidate/1.0.0"
    code: Annotated[str, StringConstraints(pattern=r"^CLAIM:CANDIDATE:[A-F0-9]{64}$")]
    version: Version
    status: Literal["draft"] = "draft"
    source: Reference
    source_version: Reference
    document: Reference
    chunk: Reference
    page: int | None = Field(ge=1)
    paragraph: int = Field(ge=1)
    section_path: tuple[Term, ...] = Field(max_length=64)
    quote_checksum: Checksum
    parse_checksum: Checksum
    grid_checksum: Checksum | None
    provenance: Provenance
    # Unassessed source quality is explicit and is independent of retrieval scores.
    evidence_grade: Literal["unassessed"] = "unassessed"
    authority: Literal["unassessed"] = "unassessed"
    currency: Literal["unassessed"] = "unassessed"
    directness: Literal["extractive"] = "extractive"

    @model_validator(mode="after")
    def reference_kinds(self):
        for field, kind in (
            ("source", "source"),
            ("source_version", "source_version"),
            ("document", "source_document"),
            ("chunk", "document_chunk"),
        ):
            if getattr(self, field).kind != kind:
                raise ValueError("source_binding")
        from packages.object_store import checksum

        if checksum(self.quote.encode()) != self.quote_checksum:
            raise ValueError("quote_checksum")
        return self


class ExtractionAudit(Model):
    version: Literal["claim-audit/1.0.0"] = "claim-audit/1.0.0"
    batch_key: Checksum
    status: Literal["succeeded", "rejected", "failed"]
    reason: Term | None
    attempts: int = Field(ge=0, le=4)
    started_at: Term
    completed_at: Term
    input_checksum: Checksum
    output_checksum: Checksum | None
    config_checksum: Checksum
    prompt_checksum: Checksum
    schema_checksum: Checksum
    configuration: ConfigurationSnapshot
    accepted_codes: tuple[str, ...] = Field(max_length=8)
    # No raw provider errors, raw responses, credentials or hidden reasoning.
