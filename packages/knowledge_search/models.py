"""04B2 internal, secret-free planning and discovery metadata; no persistence IO."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, StringConstraints, field_validator, model_validator

from packages.knowledge_connectors.ports import (
    Checksum,
    Code,
    ConfiguredIdentity,
    Identifier,
    RequestMetadata,
    StrictModel,
    Version,
    WireReceipt,
    model_checksum,
)

from .discovery import DiscoveryPolicy, validate_candidate
from .discovery import identifier as discovery_identifier
from .discovery import record_id as discovery_record_id
from .privacy import (
    CATALOG,
    IDENTIFIER_CHECKSUM,
    POLICY_CHECKSUM,
    POLICY_VERSION,
    PrivacyError,
    canonical_doi,
    coded,
    controlled,
    safe_text,
)
from .provider import ProviderConfigurationProof, assert_identity

SafeText = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=2048), AfterValidator(safe_text)
]
Term = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=160), AfterValidator(safe_text)
]
EnvName = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z][A-Z0-9_]{0,127}$")]
SourceCategory = Literal[
    "guideline",
    "systematic_review",
    "original_research",
    "educational_overview",
    "web",
    "consensus",
]
EvidenceNeed = Literal[
    "definition",
    "diagnostic_criteria",
    "association",
    "test_accuracy",
    "prognosis",
    "management",
    "contraindication",
]
StopReason = Literal[
    "complete",
    "empty",
    "hit_budget",
    "request_budget",
    "page_limit",
    "duplicate_page",
    "timeout",
    "cancelled",
    "provider_error",
]


def normalize_term(value: str) -> str:
    if type(value) is not str:
        raise ValueError("strict text required")
    value = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    safe_text(value)
    if not value or len(value) > 160 or re.search(r'["\[\]{}<>\\;=]', value):
        raise ValueError("literal bounded search term required")
    return value


class CodedTerm(StrictModel):
    terminology_source: Literal["MMAGENT_GENERAL_KNOWLEDGE_CATALOG"] = (
        "MMAGENT_GENERAL_KNOWLEDGE_CATALOG"
    )
    terminology_version: Literal["1.0.0", "2.0.0", "3.0.0"] = "1.0.0"
    code: Code
    label: Term
    synonyms: tuple[Term, ...] = Field(default=(), max_length=12)

    @field_validator("code", mode="before")
    @classmethod
    def code_case(cls, value):
        return value.strip().upper() if type(value) is str else value

    @field_validator("label", mode="before")
    @classmethod
    def label_case(cls, value):
        return normalize_term(value)

    @field_validator("synonyms", mode="before")
    @classmethod
    def synonym_order(cls, value):
        if type(value) not in (tuple, list):
            raise ValueError("synonym collection required")
        return tuple(sorted({normalize_term(v) for v in value}))

    @model_validator(mode="after")
    def canonical_group(self) -> Self:
        # Choosing a different synonym as display label has no semantic effect.
        terms = sorted({self.label, *self.synonyms})
        if self.terminology_version == "1.0.0":
            coded(self.code, tuple(terms))
        else:
            from .broad import BROAD_CATALOG
            from .targeted import TARGET_CATALOG

            catalog = TARGET_CATALOG if self.terminology_version == "3.0.0" else BROAD_CATALOG
            allowed = catalog["concepts"].get(self.code, ())
            if not allowed or any(t not in allowed for t in terms):
                raise PrivacyError()
        object.__setattr__(self, "label", terms[0])
        object.__setattr__(self, "synonyms", tuple(terms[1:]))
        return self


class IdentifierPolicy(StrictModel):
    version: Literal["bibliographic-identifier/1.0.0"] = "bibliographic-identifier/1.0.0"
    checksum: Literal[IDENTIFIER_CHECKSUM] = IDENTIFIER_CHECKSUM
    purpose: Literal["general_bibliographic_lookup"] = "general_bibliographic_lookup"
    existence: Literal["not_checked"] = "not_checked"


class QueryIdentifier(StrictModel):
    # Bare numeric locators cannot prove a bibliographic source in this input API.
    kind: Literal["DOI"]
    value: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
    source: Literal["self_identifying_doi"] = "self_identifying_doi"
    policy: IdentifierPolicy = Field(default_factory=IdentifierPolicy)

    @field_validator("value", mode="before")
    @classmethod
    def normalize(cls, value):
        return canonical_doi(value)


class ProviderBudget(StrictModel):
    provider_code: Code
    priority: int = Field(default=0, ge=0, le=100)
    max_hits: int = Field(default=20, ge=1, le=100)
    max_requests: int = Field(default=16, ge=0, le=128)
    max_pages: int = Field(default=3, ge=1, le=20)
    page_size: int = Field(default=20, ge=1, le=100)
    timeout_ms: int = Field(default=30000, ge=1, le=300000)
    max_retries: int = Field(default=2, ge=0, le=7)


class ExpansionSuggestion(StrictModel):
    concept_code: Code
    kind: Literal["synonym", "mesh", "spelling"]
    term: Term

    @field_validator("term", mode="before")
    @classmethod
    def normalize(cls, value):
        return normalize_term(value)

    @model_validator(mode="after")
    def catalog_term(self):
        coded(self.concept_code, (self.term,))
        return self


def suggestion_set(values):
    return tuple(sorted(set(values), key=lambda s: (s.concept_code, s.kind, s.term)))


class ExpansionOutput(StrictModel):
    suggestions: tuple[ExpansionSuggestion, ...] = Field(max_length=24)

    @field_validator("suggestions")
    @classmethod
    def canonical(cls, values):
        return suggestion_set(values)


SYNONYM_PROMPT = (
    "Suggest only allowlisted synonyms, MeSH candidates or spelling variants for existing "
    "concept codes. Return a JSON object with suggestions only. Never change goal, claim, "
    "evidence need, population, age, sex, dates or other applicability conditions."
)


class ExpansionConfig(StrictModel):
    enabled: bool = False
    prompt_version: Version = "1.0.0"
    config_version: Version = "1.0.0"
    model: Term = "disabled"
    allowlist: tuple[ExpansionSuggestion, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def canonical_allowlist(self):
        controlled("models", self.model)
        values = tuple(sorted(set(self.allowlist), key=lambda x: (x.concept_code, x.kind, x.term)))
        object.__setattr__(self, "allowlist", values)
        return self


class ExpansionTrace(StrictModel):
    suggestion_policy: Literal["set/2.0.0"]
    prompt_version: Version
    model: Term
    config_version: Version
    config_checksum: Checksum
    prompt_checksum: Checksum
    input_plan_checksum: Checksum
    prompt: Literal[SYNONYM_PROMPT] = SYNONYM_PROMPT
    config: ExpansionConfig
    output_checksum: Checksum
    suggestions: tuple[ExpansionSuggestion, ...] = Field(max_length=24)

    @model_validator(mode="after")
    def bound_output(self) -> Self:
        object.__setattr__(self, "suggestions", suggestion_set(self.suggestions))
        if (
            self.config_checksum != model_checksum(self.config)
            or self.model != self.config.model
            or self.config_version != self.config.config_version
            or self.prompt_version != self.config.prompt_version
            or not self.config.enabled
            or any(s not in self.config.allowlist for s in self.suggestions)
        ):
            raise ValueError("expansion configuration binding mismatch")
        from packages.knowledge_connectors.ports import digest

        expected_prompt = digest(
            (self.prompt + self.prompt_version + self.input_plan_checksum).encode()
        )
        if self.prompt_checksum != expected_prompt:
            raise ValueError("expansion prompt binding mismatch")
        if self.output_checksum != model_checksum(ExpansionOutput(suggestions=self.suggestions)):
            raise ValueError("expansion output checksum mismatch")
        return self


class CohortConditions(StrictModel):
    provenance: Literal["research_cohort_catalog"] = "research_cohort_catalog"
    source_version: Literal["1.0.0"] = "1.0.0"
    code: Literal["UNRESTRICTED", "ADULTS", "ADULT_FEMALES"] = "UNRESTRICTED"


class PatientFreePolicy(StrictModel):
    version: Literal[POLICY_VERSION, "patient-free/4.0.0", "patient-free/5.0.0"] = POLICY_VERSION
    checksum: Checksum = POLICY_CHECKSUM
    term_source: Literal["MMAGENT_GENERAL_KNOWLEDGE_CATALOG"] = "MMAGENT_GENERAL_KNOWLEDGE_CATALOG"
    term_version: Literal["1.0.0", "2.0.0", "3.0.0"] = "1.0.0"
    temporal_scope: Literal["publication_window"] = "publication_window"

    @model_validator(mode="after")
    def bound_version(self):
        from .broad import BROAD_CHECKSUM
        from .targeted import TARGET_CHECKSUM

        expected = (
            (POLICY_CHECKSUM, "1.0.0")
            if self.version == POLICY_VERSION
            else (TARGET_CHECKSUM, "3.0.0")
            if self.version == "patient-free/5.0.0"
            else (BROAD_CHECKSUM, "2.0.0")
        )
        if (self.checksum, self.term_version) != expected:
            raise PrivacyError()
        return self


class KnowledgeQueryPlan(StrictModel):
    query_plan_version: Literal["4.0.0", "5.0.0", "6.0.0"] = "4.0.0"
    identifier_policy: IdentifierPolicy = Field(default_factory=IdentifierPolicy)
    privacy: PatientFreePolicy = Field(default_factory=PatientFreePolicy)
    cohort: CohortConditions = Field(default_factory=CohortConditions)
    template_version: Literal["2.0.0", "3.0.0", "4.0.0"] = "2.0.0"
    goal: Code
    claim: Code
    evidence_need: EvidenceNeed
    diseases: tuple[CodedTerm, ...] = Field(min_length=1, max_length=8)
    concepts: tuple[CodedTerm, ...] = Field(default=(), max_length=8)
    population: tuple[Term, ...] = Field(min_length=1, max_length=8)
    age_min: int | None = Field(default=None, ge=0, le=130)
    age_max: int | None = Field(default=None, ge=0, le=130)
    sexes: tuple[Literal["female", "male", "intersex", "unknown"], ...] = ()
    modalities: tuple[Term, ...] = Field(default=(), max_length=8)
    examinations: tuple[CodedTerm, ...] = Field(default=(), max_length=8)
    anatomy: tuple[Term, ...] = Field(default=(), max_length=8)
    date_from: date | None = None
    date_to: date | None = None
    source_version: Version | None = None
    languages: tuple[Term, ...] = Field(default=("en",), min_length=1, max_length=8)
    source_types: tuple[SourceCategory, ...] = Field(min_length=1, max_length=6)
    identifiers: tuple[QueryIdentifier, ...] = Field(default=(), max_length=8)
    providers: tuple[ProviderBudget, ...] = Field(min_length=1, max_length=8)
    total_max_hits: int = Field(default=80, ge=1, le=200)
    total_max_requests: int = Field(default=64, ge=0, le=256)
    total_timeout_ms: int = Field(default=60000, ge=1, le=600000)
    max_concurrency: int = Field(default=3, ge=1, le=8)
    stop_on_empty: bool = True
    stop_on_duplicate_page: bool = True
    expansion: ExpansionTrace | None = None

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def dates(cls, value):
        if type(value) is str:
            if not re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", value):
                raise ValueError("full calendar date required")
            return date.fromisoformat(value.replace("/", "-"))
        return value

    @field_validator("population", "modalities", "anatomy", "languages", mode="before")
    @classmethod
    def normalized_sets(cls, value):
        if type(value) not in (tuple, list):
            raise ValueError("term collection required")
        return tuple(sorted({normalize_term(v) for v in value}))

    @model_validator(mode="after")
    def canonical(self) -> Self:
        broad = self.query_plan_version in {"5.0.0", "6.0.0"}
        expected = (
            ("4.0.0", "patient-free/5.0.0", "3.0.0")
            if self.query_plan_version == "6.0.0"
            else ("3.0.0", "patient-free/4.0.0", "2.0.0")
            if broad
            else ("2.0.0", POLICY_VERSION, "1.0.0")
        )
        if (self.template_version, self.privacy.version, self.privacy.term_version) != expected:
            raise PrivacyError()
        if any(
            t.terminology_version != expected[2]
            for t in (*self.diseases, *self.concepts, *self.examinations)
        ):
            raise PrivacyError()
        if broad and (
            tuple(t.code for t in self.diseases) != ("CONCEPT:DISEASE:GLAUCOMA",)
            or self.examinations
            or self.expansion is not None
            or self.identifiers
            or self.modalities
            or self.anatomy
            or self.cohort.code != "UNRESTRICTED"
            or self.date_from is not None
            or self.date_to is not None
            or self.source_version is not None
        ):
            raise ValueError("broad_discovery_requires_supported_public_topic_scope")
        if self.query_plan_version == "6.0.0":
            from .targeted import TARGET_CODES

            if sum(t.code in TARGET_CODES.values() for t in self.concepts) != 1:
                raise ValueError("public_proposition_requires_exactly_one_target")
        if self.goal not in CATALOG["goals"] or self.claim not in CATALOG["claims"]:
            raise PrivacyError()
        for field in ("population", "modalities", "anatomy", "languages"):
            for term in getattr(self, field):
                controlled(field, term)
        profile = CATALOG["cohorts"][self.cohort.code]
        if (self.age_min, self.age_max, tuple(sorted(set(self.sexes)))) != (
            profile["age_min"],
            profile["age_max"],
            tuple(profile["sexes"]),
        ):
            raise PrivacyError()
        if self.age_min is not None and self.age_max is not None and self.age_min > self.age_max:
            raise ValueError("age interval is reversed")
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError("date interval is reversed")
        if any(not re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2})?", v) for v in self.languages):
            raise ValueError("language codes required")
        for name in ("diseases", "concepts", "examinations"):
            values = getattr(self, name)
            if len({v.code for v in values}) != len(values):
                raise ValueError("duplicate concept code")
            object.__setattr__(self, name, tuple(sorted(values, key=lambda v: v.code)))
        for name in ("sexes", "source_types"):
            object.__setattr__(self, name, tuple(sorted(set(getattr(self, name)))))
        object.__setattr__(
            self,
            "identifiers",
            tuple(sorted(set(self.identifiers), key=lambda v: (v.kind, v.value))),
        )
        if len({v.provider_code for v in self.providers}) != len(self.providers):
            raise ValueError("duplicate provider budget")
        object.__setattr__(
            self,
            "providers",
            tuple(sorted(self.providers, key=lambda v: (v.priority, v.provider_code))),
        )
        if self.expansion:
            base = self.model_copy(update={"expansion": None})
            if self.expansion.input_plan_checksum != model_checksum(base):
                raise ValueError("expansion cannot change input conditions")
            codes = {v.code for v in (*self.diseases, *self.concepts, *self.examinations)}
            if any(s.concept_code not in codes for s in self.expansion.suggestions):
                raise ValueError("expansion cannot change the target concepts")
        return self

    def checksum(self) -> str:
        return model_checksum(self)


class ProviderIdentity(ConfiguredIdentity):
    configuration: ProviderConfigurationProof
    identity_policy: Literal["contact-env/2.0.0"]
    contact_environment: EnvName | None
    auth_environment_name: EnvName | None
    identity_configured: bool
    policy_version: Version
    policy_checksum: Checksum
    deployment_workers: int = Field(ge=1, le=1)

    @model_validator(mode="after")
    def configuration_bound(self):
        assert_identity(self, self.configuration)
        return self


class ProviderQuery(StrictModel):
    identity: ProviderIdentity
    plan_checksum: Checksum
    template_version: Version
    texts: tuple[SafeText, ...] = Field(min_length=1, max_length=16)
    constraints_mode: Literal[
        "search_terms", "catalog_title_only", "broad_terms_preferences_unfiltered"
    ]


class ExecutionBinding(StrictModel):
    provider_proofs: tuple[ProviderConfigurationProof, ...] = Field(min_length=1, max_length=8)
    plan: KnowledgeQueryPlan
    queries: tuple[ProviderQuery, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def bound_queries(self):
        from .planning import query_texts

        if tuple(q.identity.configuration for q in self.queries) != self.provider_proofs:
            raise ValueError("execution provider proofs mismatch")
        if tuple(q.identity.provider_code for q in self.queries) != tuple(
            b.provider_code for b in self.plan.providers
        ):
            raise ValueError("complete ordered provider scope required")
        for query in self.queries:
            texts, mode = query_texts(self.plan, query.identity.adapter)
            if (
                query.plan_checksum,
                query.template_version,
                query.texts,
                query.constraints_mode,
            ) != (self.plan.checksum(), self.plan.template_version, texts, mode):
                raise ValueError("query binding mismatch")
        return self


class PageReceipt(StrictModel):
    subquery_index: int = Field(ge=0, le=15)
    page_index: int = Field(ge=0, le=19)
    query_checksum: Checksum
    page_checksum: Checksum
    cursor_checksum: Checksum | None = None
    requests: tuple[RequestMetadata, ...] = Field(default=(), max_length=2)
    response_checksums: tuple[Checksum, ...] = Field(default=(), max_length=2)
    retrieved_at: datetime


class HitMetadata(ConfiguredIdentity):
    configuration: ProviderConfigurationProof
    discovery_policy: DiscoveryPolicy
    record_id_checksum: Checksum
    provider_record_id: (
        Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2048)] | None
    )
    discovery_identifiers: tuple[Identifier, ...] = Field(default=(), max_length=16)
    source_type: Literal["GUIDELINE", "ARTICLE", "CONSENSUS", "WEB"]
    subquery_index: int = Field(ge=0, le=15)
    page_index: int = Field(ge=0, le=19)
    provider_rank: int = Field(ge=1)
    page_checksum: Checksum
    query_checksum: Checksum
    request_checksums: tuple[Checksum, ...] = Field(max_length=2)
    response_checksums: tuple[Checksum, ...] = Field(max_length=2)
    ranking_semantics: Literal["retrieval_relevance_only"] = "retrieval_relevance_only"

    @model_validator(mode="after")
    def safe_discovery_identifiers(self):
        if self.scope() != self.configuration.scope():
            raise PrivacyError()
        if (
            discovery_record_id(self.provider_record_id, self.configuration)
            != self.provider_record_id
        ):
            raise PrivacyError()
        for identifier in self.discovery_identifiers:
            if identifier.identifier_type == "URL":
                raise ValueError("URLs belong only to query-free SearchHit identifiers")
            if (
                discovery_identifier(
                    identifier.identifier_type, identifier.value, self.configuration
                )
                != identifier.value
            ):
                raise PrivacyError()
        return self


class CandidateHit(StrictModel):
    metadata: HitMetadata
    title: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2048)]
    identifiers: tuple[Identifier, ...] = Field(min_length=1, max_length=16)
    discovered_at: datetime

    @model_validator(mode="after")
    def safe_identifiers(self) -> Self:
        return validate_candidate(self)


class OutcomeMetadata(StrictModel):
    query: ProviderQuery
    started_at: datetime
    completed_at: datetime
    elapsed_ms: int = Field(ge=0)
    stop_reason: StopReason
    error_category: (
        Literal[
            "disabled",
            "configuration",
            "unsupported",
            "security",
            "timeout",
            "network",
            "authentication",
            "rate_limit",
            "http",
            "parse",
            "response_size",
            "mime",
            "retry_exhausted",
            "page_limit",
            "internal",
        ]
        | None
    ) = None
    wire_requests: int = Field(ge=0, le=256)
    wire_receipts: tuple[WireReceipt, ...] = Field(default=(), max_length=128)
    receipts: tuple[PageReceipt, ...] = Field(default=(), max_length=20)
    error_requests: tuple[RequestMetadata, ...] = Field(default=(), max_length=20)
    hit_count: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def consistent_outcome(self) -> Self:
        if (
            self.started_at.tzinfo is None
            or self.completed_at.tzinfo is None
            or self.completed_at < self.started_at
        ):
            raise ValueError("aware ordered execution times required")
        scope = self.query.identity.scope()
        if any(r.scope() != scope for r in self.wire_receipts):
            raise ValueError("wire receipt provider scope mismatch")
        for receipt in self.receipts:
            if any(r.scope() != scope for r in receipt.requests):
                raise ValueError("receipt provider scope mismatch")
            if len(receipt.requests) != len(receipt.response_checksums):
                raise ValueError("request/response receipt mismatch")
        if any(r.scope() != scope for r in self.error_requests):
            raise ValueError("error metadata provider scope mismatch")
        if self.stop_reason in ("complete", "empty") and self.error_category is not None:
            raise ValueError("successful discovery cannot carry an error")
        if self.stop_reason == "provider_error" and self.error_category is None:
            raise ValueError("failed provider needs a typed error")
        if self.stop_reason == "complete" and not self.hit_count:
            raise ValueError("zero-hit completion must be empty")
        if self.stop_reason == "empty" and self.hit_count:
            raise ValueError("empty discovery cannot have hits")
        return self


def hit_set_checksum(hits: tuple[CandidateHit, ...]) -> str:
    import json

    from packages.knowledge_connectors.ports import digest

    return digest(
        json.dumps(sorted(model_checksum(hit) for hit in hits), separators=(",", ":")).encode()
    )


class ProviderOutcome(OutcomeMetadata):
    representation: Literal["live/2.0.0"] = "live/2.0.0"
    hits: tuple[CandidateHit, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def live_hits(self):
        if len(self.hits) != self.hit_count or len(
            {h.metadata.record_id_checksum for h in self.hits}
        ) != len(self.hits):
            raise ValueError("live hit count/set mismatch")
        for hit in self.hits:
            m = hit.metadata
            if (
                m.scope() != self.query.identity.scope()
                or m.configuration != self.query.identity.configuration
            ) or not any(
                (
                    m.page_checksum,
                    m.query_checksum,
                    m.page_index,
                    m.subquery_index,
                    m.request_checksums,
                    m.response_checksums,
                )
                == (
                    r.page_checksum,
                    r.query_checksum,
                    r.page_index,
                    r.subquery_index,
                    tuple(q.request_checksum for q in r.requests),
                    r.response_checksums,
                )
                for r in self.receipts
            ):
                raise ValueError("live hit receipt binding mismatch")
        return self

    def persisted(self):
        live = ProviderOutcome.model_validate(self)
        return PersistedProviderOutcome(
            **live.model_dump(exclude={"hits", "representation"}),
            representation="persisted/2.0.0",
            hit_set_checksum=hit_set_checksum(live.hits),
        )


class PersistedProviderOutcome(OutcomeMetadata):
    representation: Literal["persisted/2.0.0"]
    hit_set_checksum: Checksum


FederatedStatus = Literal[
    "succeeded", "empty", "partial", "failed", "cancelled", "timeout", "budget_exhausted"
]


def derive_federated_status(outcomes: tuple[OutcomeMetadata, ...]) -> FederatedStatus:
    if not outcomes:
        raise ValueError("nonempty outcomes required")
    for outcome in outcomes:
        if type(outcome) not in (ProviderOutcome, PersistedProviderOutcome):
            raise ValueError("explicit live or persisted outcome required")
        type(outcome).model_validate(outcome)
    reasons = {o.stop_reason for o in outcomes}
    hits = sum(o.hit_count for o in outcomes)
    if not reasons - {"complete", "empty"}:
        return "succeeded" if hits else "empty"
    if hits or reasons & {"complete", "empty"}:
        return "partial"
    if "cancelled" in reasons:
        return "cancelled"
    if "timeout" in reasons:
        return "timeout"
    if reasons & {"request_budget", "hit_budget", "page_limit"}:
        return "budget_exhausted"
    return "failed"


class FederatedResult(StrictModel):
    binding: ExecutionBinding
    plan_checksum: Checksum
    execution_checksum: Checksum
    status: FederatedStatus
    providers: tuple[ProviderOutcome, ...] = Field(min_length=1, max_length=8)
    ranking_semantics: Literal["retrieval_relevance_only"] = "retrieval_relevance_only"

    @model_validator(mode="after")
    def closed_result(self):
        if (
            self.plan_checksum != self.binding.plan.checksum()
            or self.execution_checksum != model_checksum(self.binding)
            or tuple(o.query for o in self.providers) != self.binding.queries
            or self.status != derive_federated_status(self.providers)
        ):
            raise ValueError("federated result binding/status mismatch")
        return self
