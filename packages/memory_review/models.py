"""10D independent qualification envelopes; no content or clinical contract upgrade."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from db.knowledge.models import KnowledgeReference
from packages.grounding.models import Checksum, Token, Value


class Policy(Value):
    schema_version: Literal["memory-publication-policy/1.0.0"] = "memory-publication-policy/1.0.0"
    key: Token
    approved: bool = False
    purpose: Literal["synthetic_offline", "research_offline"] = "research_offline"
    test_context: Token | None = None
    min_cases: int | None = Field(default=None, ge=2)
    case_counting: Literal["distinct_source_case_code_not_patient_count"] = (
        "distinct_source_case_code_not_patient_count"
    )
    min_reviewers: int | None = Field(default=None, ge=1)
    reviewers: tuple[Token, ...] = ()
    identities: dict[Token, Token] = {}
    domains: tuple[Token, ...]
    valid_from: datetime
    expires_at: datetime
    validation: Literal["synthetic_engineering", "not_evaluated"] = "not_evaluated"
    clinical_level: Literal["not_evaluated"] = "not_evaluated"
    revocation: Literal["immediate_recheck_no_cached_authority"] = (
        "immediate_recheck_no_cached_authority"
    )

    @model_validator(mode="after")
    def boundaries(self):
        if (
            not self.valid_from.tzinfo
            or not self.expires_at.tzinfo
            or self.expires_at <= self.valid_from
        ):
            raise ValueError("memory_policy_validity")
        if self.approved and (
            self.purpose != "synthetic_offline"
            or not self.test_context
            or self.min_cases is None
            or self.min_reviewers is None
            or self.min_reviewers > len(set(self.reviewers))
            or any(r not in self.identities for r in self.reviewers)
            or self.validation != "synthetic_engineering"
        ):
            raise ValueError("memory_real_policy_unapproved")
        return self


class Conditions(Value):
    goal: Token
    template: Token
    graph_conditions: tuple[Token, ...]
    modalities: tuple[Token, ...]
    inputs: tuple[Token, ...]
    patient_conditions: dict[Token, Token]
    tool_versions: dict[Token, Token]
    rule_versions: dict[Token, Token]
    failure_types: tuple[Token, ...]
    validation: Literal["synthetic_engineering"] = "synthetic_engineering"
    domain: Token


class Query(Conditions):
    schema_version: Literal["memory-query/1.0.0"] = "memory-query/1.0.0"
    run_id: Token
    case_code: Token
    test_context: Token
    purpose: Literal["synthetic_offline"] = "synthetic_offline"
    text: tuple[Token, ...] = ()
    groups: tuple[Literal["success", "counterfactual", "recovery"], ...] = (
        "success",
        "counterfactual",
        "recovery",
    )


class Binding(Value):
    schema_version: Literal["memory-use-binding/1.0.0"] = "memory-use-binding/1.0.0"
    target: Token
    target_checksum: Checksum
    content_version: Literal[
        "research-memory/1.0.0", "research-memory/1.1.0", "research-memory/1.2.0"
    ]
    conditions: Conditions
    group: Literal["success", "counterfactual", "recovery"]
    preferred_branches: tuple[Token, ...]
    terms: tuple[Token, ...] = ()
    # Applies only inside a declared synthetic fixture. Not an inferred clinical claim.
    basis: Literal["synthetic_fixture_mapping"] = "synthetic_fixture_mapping"


class Opinion(Value):
    schema_version: Literal["memory-opinion/1.0.0"] = "memory-opinion/1.0.0"
    key: Token
    binding: Binding
    kind: Literal["agent", "engineering"]
    decision: Literal["candidate_supported", "candidate_rejected", "needs_more_cases"]
    reasons: tuple[Token, ...] = Field(min_length=1, max_length=32)
    source_checksums: tuple[Checksum, ...] = Field(min_length=1, max_length=64)
    valid_from: datetime
    expires_at: datetime
    request_checksums: tuple[Checksum, ...] = ()
    clinical_approval: Literal[False] = False

    @model_validator(mode="after")
    def validity(self):
        if (
            not self.valid_from.tzinfo
            or not self.expires_at.tzinfo
            or self.expires_at <= self.valid_from
        ):
            raise ValueError("memory_opinion_validity")
        if (self.kind == "agent") != bool(self.request_checksums):
            raise ValueError("memory_opinion_request_binding")
        return self


class Publication(Value):
    schema_version: Literal["memory-publication/1.0.0"] = "memory-publication/1.0.0"
    key: Token
    binding: Binding
    policy_key: Token
    policy_checksum: Checksum
    opinions: tuple[Token, ...] = Field(min_length=1, max_length=32)
    valid_from: datetime
    expires_at: datetime
    purpose: Literal["synthetic_offline"] = "synthetic_offline"
    test_context: Token

    @model_validator(mode="after")
    def validity(self):
        if (
            not self.valid_from.tzinfo
            or not self.expires_at.tzinfo
            or self.expires_at <= self.valid_from
        ):
            raise ValueError("memory_publication_validity")
        if len(set(self.opinions)) != len(self.opinions):
            raise ValueError("memory_duplicate_opinion")
        return self


class ReviewInput(Value):
    schema_version: Literal["memory-agent-input/1.0.0"] = "memory-agent-input/1.0.0"
    role: Literal["expert"] = "expert"
    scope: dict
    binding: Binding
    sources: tuple[Checksum, ...]
    instruction: Literal["candidate_advice_only_no_publication"] = (
        "candidate_advice_only_no_publication"
    )


class Advice(Value):
    schema_version: Literal["memory-agent-advice/1.0.0"] = "memory-agent-advice/1.0.0"
    target_checksum: Checksum
    binding_checksum: Checksum
    decision: Literal["candidate_supported", "candidate_rejected", "needs_more_cases"]
    reasons: tuple[Token, ...] = Field(min_length=1, max_length=32)


class JointKnowledgeBinding(Value):
    schema_version: Literal["memory-joint-knowledge-binding/1.0.0"] = (
        "memory-joint-knowledge-binding/1.0.0"
    )
    run_id: Token
    run_context_checksum: Checksum
    release: KnowledgeReference
    context_checksum: Checksum
    config_checksum: Checksum
