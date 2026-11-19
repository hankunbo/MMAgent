"""Bounded source-first drafting. Only a definition owner may accept a draft.

Exact citations prove traceability, not medical correctness. No self-approval or
automatically fabricated thresholds. The caller composes original search/LLM ports.
"""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Value, digest

from .models import check


class Excerpt(Value):
    source_code: str
    source_checksum: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=4000)


class DefinitionDraft(Value):
    schema_version: Literal["source-definition-draft/1.0.0"] = "source-definition-draft/1.0.0"
    target: str
    relation: Literal["qualitative", "quantitative", "context", "dependency"]
    examination: str = Field(min_length=1, max_length=4000)
    interpretation: str = Field(min_length=1, max_length=4000)
    applicability: str = Field(min_length=1, max_length=4000)
    input_concepts: tuple[str, ...] = Field(min_length=1, max_length=16)
    method: str | None
    unit: str | None
    excerpts: tuple[Excerpt, ...] = Field(min_length=1, max_length=8)
    automatically_accepted: Literal[False] = False


class ResearchDefinitionDraft(DefinitionDraft):
    """Compact, source-bound definition suitable for the existing task projection."""

    schema_version: Literal["source-definition-draft/2.0.0"] = "source-definition-draft/2.0.0"
    examination: str = Field(min_length=1, max_length=240)
    interpretation: str = Field(min_length=1, max_length=240)
    applicability: str = Field(min_length=1, max_length=240)
    modalities: tuple[str, ...] = Field(min_length=1, max_length=8)
    acquisition: Literal["vision", "source_measurement"]
    relation_use: Literal["directional", "examination_only"]
    visible_scope: str = Field(min_length=1, max_length=240)
    extraction_reason: str = Field(min_length=1, max_length=1000)
    alternatives: tuple[str, ...] = Field(min_length=1, max_length=8)
    limitations: tuple[str, ...] = Field(min_length=1, max_length=8)
    dependencies: tuple[str, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def method_binding(self):
        if self.relation == "quantitative" and (not self.method or not self.unit):
            raise ValueError("definition_quantitative_method_unit")
        if self.acquisition == "vision" and (self.method, self.unit) != (
            "live_image_observation",
            "qualitative",
        ):
            raise ValueError("definition_vision_method_unit")
        return self


class DefinitionProposal(Value):
    schema_version: Literal["research-definition-proposal/1.0.0"] = (
        "research-definition-proposal/1.0.0"
    )
    definitions: tuple[ResearchDefinitionDraft, ...] = Field(max_length=3)
    no_addition_reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def outcome(self):
        if (not self.definitions) != bool(self.no_addition_reason):
            raise ValueError("definition_proposal_outcome")
        return self


class QuotedExcerpt(Value):
    """Model supplies exact text; only the source owner computes coordinates."""

    source_code: str
    source_checksum: str
    text: str = Field(min_length=1, max_length=4000)
    prefix: str | None = Field(default=None, max_length=240)
    suffix: str | None = Field(default=None, max_length=240)


class QuotedResearchDefinitionDraft(ResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/2.1.0"] = "source-definition-draft/2.1.0"
    excerpts: tuple[QuotedExcerpt, ...] = Field(min_length=1, max_length=8)


class QuotedDefinitionProposal(DefinitionProposal):
    schema_version: Literal["research-definition-proposal/1.1.0"] = (
        "research-definition-proposal/1.1.0"
    )
    definitions: tuple[QuotedResearchDefinitionDraft, ...] = Field(max_length=3)


class DependencyContract(Value):
    """Source-proposed meaning, never a caller assertion of current satisfaction."""

    index: int = Field(ge=0, le=15)
    text: str = Field(min_length=1, max_length=1000)
    kind: Literal["patient_input", "method", "quality"]
    blocks: tuple[Literal["acquisition", "interpretation"], ...] = Field(min_length=1, max_length=2)
    modality: str | None = Field(default=None, min_length=1, max_length=100)
    concept: str | None = Field(default=None, min_length=1, max_length=100)
    rationale: str = Field(min_length=1, max_length=1000)
    excerpt_indices: tuple[int, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def binding(self):
        if len(set(self.blocks)) != len(self.blocks) or len(set(self.excerpt_indices)) != len(
            self.excerpt_indices
        ):
            raise ValueError("definition_dependency_duplicate")
        if self.kind == "patient_input" and (self.modality is None or self.concept is None):
            raise ValueError("definition_dependency_input_selector")
        if self.kind != "patient_input" and (self.modality is not None or self.concept is not None):
            raise ValueError("definition_dependency_non_input_selector")
        return self


class ScopedResearchDefinitionDraft(ResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/3.0.0"] = "source-definition-draft/3.0.0"
    dependency_contracts: tuple[DependencyContract, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def dependency_binding(self):
        if sorted(c.index for c in self.dependency_contracts) != list(
            range(len(self.dependencies))
        ):
            raise ValueError("definition_dependency_coverage")
        for c in self.dependency_contracts:
            if c.text != self.dependencies[c.index] or any(
                not 0 <= i < len(self.excerpts) for i in c.excerpt_indices
            ):
                raise ValueError("definition_dependency_binding")
        return self


class QuotedScopedResearchDefinitionDraft(ScopedResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/3.1.0"] = "source-definition-draft/3.1.0"
    excerpts: tuple[QuotedExcerpt, ...] = Field(min_length=1, max_length=8)


class ScopedDefinitionProposal(DefinitionProposal):
    schema_version: Literal["research-definition-proposal/2.0.0"] = (
        "research-definition-proposal/2.0.0"
    )
    definitions: tuple[QuotedScopedResearchDefinitionDraft, ...] = Field(max_length=3)


class EvidenceDependencyContract(DependencyContract):
    """Availability predicates are not substitutes for method/quality qualification."""

    predicate: Literal["input_available", "source_numeric_available", "unresolved"]
    method: Literal["source_numeric_read"] | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def binding(self):
        if len(set(self.blocks)) != len(self.blocks) or len(set(self.excerpt_indices)) != len(
            self.excerpt_indices
        ):
            raise ValueError("definition_dependency_duplicate")
        if self.predicate == "unresolved":
            if self.method is not None or self.unit is not None:
                raise ValueError("definition_dependency_unresolved_selector")
            return DependencyContract.binding(self)
        if self.modality is None or self.concept is None or self.kind != "patient_input":
            raise ValueError("definition_dependency_availability_not_qualification")
        if self.predicate == "source_numeric_available":
            if self.method != "source_numeric_read" or self.unit is None:
                raise ValueError("definition_dependency_numeric_selector")
        elif self.method is not None or self.unit is not None:
            raise ValueError("definition_dependency_presence_selector")
        return self


class EvidenceResearchDefinitionDraft(ScopedResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/4.0.0"] = "source-definition-draft/4.0.0"
    dependency_contracts: tuple[EvidenceDependencyContract, ...] = Field(max_length=16)


class QuotedEvidenceResearchDefinitionDraft(EvidenceResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/4.1.0"] = "source-definition-draft/4.1.0"
    excerpts: tuple[QuotedExcerpt, ...] = Field(min_length=1, max_length=8)


class EvidenceDefinitionProposal(DefinitionProposal):
    schema_version: Literal["research-definition-proposal/3.0.0"] = (
        "research-definition-proposal/3.0.0"
    )
    definitions: tuple[QuotedEvidenceResearchDefinitionDraft, ...] = Field(max_length=3)


class ConfirmedInterpretationDraft(EvidenceResearchDefinitionDraft):
    """Interpret an owner-confirmed fact, not a request to synthesize a measurement.

    The alias is local to the frozen public requirement context. This syntax
    alone grants no acceptance or scheduling authority.
    """

    schema_version: Literal["source-definition-draft/5.0.0"] = "source-definition-draft/5.0.0"
    acquisition: Literal["confirmed_observation"] = "confirmed_observation"
    requirement_alias: str = Field(pattern=r"^requirement:(?:[0-9]|[1-5][0-9]|6[0-3])$")
    method: str = Field(min_length=1, max_length=240)
    unit: str = Field(min_length=1, max_length=100)


class QuotedConfirmedInterpretationDraft(ConfirmedInterpretationDraft):
    schema_version: Literal["source-definition-draft/5.1.0"] = "source-definition-draft/5.1.0"
    excerpts: tuple[QuotedExcerpt, ...] = Field(min_length=1, max_length=8)


class ConfirmedInterpretationProposal(DefinitionProposal):
    schema_version: Literal["research-definition-proposal/4.0.0"] = (
        "research-definition-proposal/4.0.0"
    )
    definitions: tuple[
        QuotedEvidenceResearchDefinitionDraft | QuotedConfirmedInterpretationDraft, ...
    ] = Field(max_length=3)


CONFIRMED_METHOD_CONDITION = "The original owner confirms this observation's method and unit."
TECHNICAL_CONDITIONS = {
    "artifact_integrity": (
        "The original output artifact bytes and binding pass the original owner's integrity checks."
    ),
    "mask_measurement_available": (
        "Disc and cup masks are nonempty; ratios are computable on the analysis grid."
    ),
}


class OwnerDependencyContract(DependencyContract):
    """Versioned selectors; no model-supplied assertion of satisfaction."""

    predicate: Literal[
        "input_available",
        "source_numeric_available",
        "confirmed_method",
        "confirmed_technical_qc",
        "unresolved",
    ]
    input_alias: str | None = Field(default=None, pattern=r"^input:(?:0|[1-9][0-9]{0,2})$")
    requirement_alias: str | None = Field(
        default=None, pattern=r"^requirement:(?:[0-9]|[1-5][0-9]|6[0-3])$"
    )
    method: str | None = Field(default=None, min_length=1, max_length=240)
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    quality_check: Literal["artifact_integrity", "mask_measurement_available"] | None = None

    @model_validator(mode="after")
    def binding(self):
        DependencyContract.binding(self)
        if self.predicate in {"confirmed_method", "confirmed_technical_qc"}:
            if (
                self.input_alias is not None
                or self.requirement_alias is None
                or self.method is None
                or self.unit is None
                or self.kind != ("method" if self.predicate == "confirmed_method" else "quality")
                or (self.quality_check is not None) != (self.predicate == "confirmed_technical_qc")
            ):
                raise ValueError("definition_dependency_confirmed_selector")
            expected_text = (
                CONFIRMED_METHOD_CONDITION
                if self.predicate == "confirmed_method"
                else TECHNICAL_CONDITIONS[self.quality_check]
            )
            if self.text != expected_text:
                raise ValueError("definition_dependency_technical_scope")
        elif self.predicate == "unresolved":
            if any(
                x is not None
                for x in (
                    self.input_alias,
                    self.requirement_alias,
                    self.method,
                    self.unit,
                    self.quality_check,
                )
            ):
                raise ValueError("definition_dependency_unresolved_selector")
        else:
            if (
                self.kind != "patient_input"
                or self.input_alias is None
                or self.requirement_alias is not None
                or self.quality_check is not None
            ):
                raise ValueError("definition_dependency_input_alias")
            if self.predicate == "source_numeric_available":
                if self.method != "source_numeric_read" or self.unit is None:
                    raise ValueError("definition_dependency_numeric_selector")
            elif self.method is not None or self.unit is not None:
                raise ValueError("definition_dependency_presence_selector")
        return self


class OwnerBoundResearchDefinitionDraft(EvidenceResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/6.0.0"] = "source-definition-draft/6.0.0"
    input_aliases: tuple[str, ...] = Field(max_length=8)
    dependency_contracts: tuple[OwnerDependencyContract, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def owner_scope(self):
        import re

        if len(set(self.input_aliases)) != len(self.input_aliases) or any(
            re.fullmatch(r"input:(?:0|[1-9][0-9]{0,2})", a) is None for a in self.input_aliases
        ):
            raise ValueError("definition_input_aliases")
        if any(c.predicate.startswith("confirmed_") for c in self.dependency_contracts):
            raise ValueError("definition_dependency_confirmed_draft_required")
        return self


class QuotedOwnerBoundResearchDefinitionDraft(OwnerBoundResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/6.1.0"] = "source-definition-draft/6.1.0"
    excerpts: tuple[QuotedExcerpt, ...] = Field(min_length=1, max_length=8)


class OwnerBoundConfirmedInterpretationDraft(ConfirmedInterpretationDraft):
    schema_version: Literal["source-definition-draft/7.0.0"] = "source-definition-draft/7.0.0"
    dependency_contracts: tuple[OwnerDependencyContract, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def confirmed_contract_scope(self):
        for contract in self.dependency_contracts:
            if contract.predicate.startswith("confirmed_") and (
                contract.requirement_alias != self.requirement_alias
                or (contract.method, contract.unit) != (self.method, self.unit)
            ):
                raise ValueError("definition_dependency_confirmed_scope")
        return self


class QuotedOwnerBoundConfirmedInterpretationDraft(OwnerBoundConfirmedInterpretationDraft):
    schema_version: Literal["source-definition-draft/7.1.0"] = "source-definition-draft/7.1.0"
    excerpts: tuple[QuotedExcerpt, ...] = Field(min_length=1, max_length=8)


class OwnerBoundDefinitionProposal(DefinitionProposal):
    schema_version: Literal["research-definition-proposal/5.0.0"] = (
        "research-definition-proposal/5.0.0"
    )
    definitions: tuple[
        QuotedOwnerBoundResearchDefinitionDraft | QuotedOwnerBoundConfirmedInterpretationDraft, ...
    ] = Field(max_length=3)


def locate_draft(raw, *, target, source_owner):
    """Exact matching only, against the current fixed source version, before acceptance.

    The resolved legacy-shaped projection is stored alongside the new candidate,
    not substituted into any historical reply or receipt.
    """
    value = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
    owner_confirmed = value.get("schema_version") == "source-definition-draft/7.1.0"
    owner_evidence = value.get("schema_version") == "source-definition-draft/6.1.0"
    confirmed = value.get("schema_version") == "source-definition-draft/5.1.0"
    evidence = value.get("schema_version") == "source-definition-draft/4.1.0"
    scoped = value.get("schema_version") == "source-definition-draft/3.1.0"
    if value.get("schema_version") not in {
        "source-definition-draft/2.1.0",
        "source-definition-draft/3.1.0",
        "source-definition-draft/4.1.0",
        "source-definition-draft/5.1.0",
        "source-definition-draft/6.1.0",
        "source-definition-draft/7.1.0",
    }:
        return validate_draft(value, target=target, source_owner=source_owner)
    draft = check(
        QuotedOwnerBoundConfirmedInterpretationDraft
        if owner_confirmed
        else QuotedOwnerBoundResearchDefinitionDraft
        if owner_evidence
        else QuotedConfirmedInterpretationDraft
        if confirmed
        else QuotedEvidenceResearchDefinitionDraft
        if evidence
        else QuotedScopedResearchDefinitionDraft
        if scoped
        else QuotedResearchDefinitionDraft,
        value,
    )
    if draft.target != target:
        raise ValueError("definition_draft_target")
    excerpts = []
    for quote in draft.excerpts:
        source = source_owner.read(quote.source_code)
        if (
            source["checksum"] != quote.source_checksum
            or source["purpose"] != "research_material"
            or source.get("state", "available") != "available"
        ):
            raise ValueError("definition_draft_exact_excerpt")
        text, matches, start = source["text"], [], 0
        while (start := text.find(quote.text, start)) >= 0:
            end = start + len(quote.text)
            if (quote.prefix is None or text[:start].endswith(quote.prefix)) and (
                quote.suffix is None or text[end:].startswith(quote.suffix)
            ):
                matches.append((start, end))
            start += 1
        if len(matches) != 1:
            raise ValueError(
                "definition_quote_ambiguous" if matches else "definition_quote_not_found"
            )
        start, end = matches[0]
        excerpts.append(
            dict(
                source_code=quote.source_code,
                source_checksum=quote.source_checksum,
                text=quote.text,
                start=start,
                end=end,
            )
        )
    resolved = dict(
        draft.model_dump(mode="json"),
        schema_version="source-definition-draft/7.0.0"
        if owner_confirmed
        else "source-definition-draft/6.0.0"
        if owner_evidence
        else "source-definition-draft/5.0.0"
        if confirmed
        else "source-definition-draft/4.0.0"
        if evidence
        else "source-definition-draft/3.0.0"
        if scoped
        else "source-definition-draft/2.0.0",
        excerpts=excerpts,
    )
    return validate_draft(resolved, target=target, source_owner=source_owner)


def validate_draft(raw, *, target, source_owner):
    value = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
    model = (
        OwnerBoundConfirmedInterpretationDraft
        if value.get("schema_version") == "source-definition-draft/7.0.0"
        else OwnerBoundResearchDefinitionDraft
        if value.get("schema_version") == "source-definition-draft/6.0.0"
        else ConfirmedInterpretationDraft
        if value.get("schema_version") == "source-definition-draft/5.0.0"
        else EvidenceResearchDefinitionDraft
        if value.get("schema_version") == "source-definition-draft/4.0.0"
        else ScopedResearchDefinitionDraft
        if value.get("schema_version") == "source-definition-draft/3.0.0"
        else ResearchDefinitionDraft
        if value.get("schema_version") == "source-definition-draft/2.0.0"
        else DefinitionDraft
    )
    draft = check(model, value)
    if draft.target != target:
        raise ValueError("definition_draft_target")
    seen, refs = set(), []
    for excerpt in draft.excerpts:
        source = source_owner.read(excerpt.source_code)
        token = (excerpt.source_code, excerpt.start, excerpt.end)
        if (
            token in seen
            or source["checksum"] != excerpt.source_checksum
            or not 0 <= excerpt.start < excerpt.end <= len(source["text"])
            or source["text"][excerpt.start : excerpt.end] != excerpt.text
            or source["purpose"] != "research_material"
            or source.get("state", "available") != "available"
        ):
            raise ValueError("definition_draft_exact_excerpt")
        seen.add(token)
        refs.append(source)
    # Current owner on both sides of proposal checking; errors remain actual errors.
    if any(source_owner.read(s["code"]) != s for s in refs):
        raise ValueError("definition_draft_source_changed")
    return draft


def prepare_definition(
    *,
    target,
    existing_codes,
    source_owner,
    proposer,
    search=None,
    max_queries=3,
    max_candidates=5,
    before_request,
):
    """One proposal, existing sources first, at most three bounded search calls.

    before_request must enforce the original budget/deadline/cancellation at *each*
    network boundary. Providers with internal retries must also honor that callback.
    Returns a draft and receipt, never installs a rule or modifies a running policy.
    """
    if (
        type(max_queries) is not int
        or not 0 <= max_queries <= 3
        or type(max_candidates) is not int
        or not 1 <= max_candidates <= 5
    ):
        raise ValueError("definition_search_budget")
    codes = list(dict.fromkeys(existing_codes))
    if len(codes) > max_candidates:
        raise ValueError("definition_source_scope_requires_batching")
    queries = 0
    if not codes and search is not None:
        while queries < max_queries and len(codes) < max_candidates:
            before_request()
            found = search(target=target, index=queries, limit=max_candidates - len(codes))
            queries += 1
            fresh = [c for c in found if c not in codes]
            if len(fresh) > max_candidates - len(codes):
                raise ValueError("definition_search_result_budget")
            codes.extend(dict.fromkeys(fresh))
            if not fresh:
                break
    if not codes:
        return dict(status="source_missing", target=target, queries=queries, proposals=0)
    sources = [source_owner.read(c) for c in codes]
    before_request()
    raw = proposer(target=target, sources=sources)
    # Returned drafts cannot cite sources outside this explicitly frozen prompt.
    draft = check(DefinitionDraft, raw)
    if not {e.source_code for e in draft.excerpts} <= set(codes):
        raise ValueError("definition_draft_source_scope")
    draft = validate_draft(raw, target=target, source_owner=source_owner)
    if [source_owner.read(c) for c in codes] != sources:
        raise ValueError("definition_draft_source_changed")
    return dict(
        status="draft_pending_owner_acceptance",
        target=target,
        draft=draft.model_dump(mode="json"),
        source_set_checksum=digest(sources),
        queries=queries,
        proposals=1,
        auto_installed=False,
    )
