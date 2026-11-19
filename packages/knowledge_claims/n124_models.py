"""04-owned N124 envelope; the v1 output and request schemas remain distinct."""

from typing import Literal

from .models import (
    CandidateRequest,
    CandidateSourceBinding,
    ClinicalClaimCandidate,
    DraftOutput,
    Model,
    NumericEvidence,
    Span,
    Text,
)


class N124Scope(Model):
    source: CandidateSourceBinding
    quote: Text
    span: Span
    population_text: Literal["people aged 18 and over"]
    age: NumericEvidence
    assessment: Literal["pending_review"] = "pending_review"


class N124Metadata(Model):
    version: Literal["nice-ng81-n124-metadata/1.0.0"]
    anchor: Literal["ng81-1_2_4"]
    recommendation: Literal["1.2.4"]
    recommendation_span: Span
    semantic_text: Text
    semantic_span: Span
    update_text: Literal["[2009, amended 2017]"]
    update_span: Span
    examples_text: Text
    examples_span: Span
    source_scope: N124Scope


class N124DraftOutput(DraftOutput):
    output_version: Literal["nice-ng81-n124-output/1.0.0"]
    metadata: N124Metadata


class N124StructuredOutput(Model):
    schema_version: Literal["claim-output/2.0.0"] = "claim-output/2.0.0"
    candidates: tuple[N124DraftOutput]  # exactly one complete paragraph


class N124CandidateRequest(CandidateRequest):
    schema_version: Literal["candidate-request/2.0.0"]
    rule_version: Literal["nice-ng81-n124/1.0.0"]
    candidate: N124DraftOutput


class N124ClinicalClaimCandidate(ClinicalClaimCandidate, N124DraftOutput):
    schema_version: Literal["clinical-claim-candidate/2.0.0"] = "clinical-claim-candidate/2.0.0"
