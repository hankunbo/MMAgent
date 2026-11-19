"""Versioned internal 04B6 contracts. They convey proposals, never clinical approval."""

from typing import Literal

from pydantic import Field, model_validator

from packages.knowledge_claims.models import Model, Term, Text

DIMENSIONS = (
    "proposition",
    "polarity",
    "population",
    "disease_stage",
    "anatomy",
    "modality",
    "method",
    "threshold",
    "unit",
    "time",
    "recommendation_strength",
    "publication_date",
    "source_lifecycle",
    "source_authority",
    "source_span",
    "applicability",
    "claim_type",
)


class ReviewAssessment(Model):
    version: Literal["offline-review/1.0.0"] = "offline-review/1.0.0"
    disposition: Literal["accept", "conditional_coexist", "reject", "needs_evidence", "supersede"]
    rationale: Text
    source_classes: tuple[
        Literal[
            "guideline",
            "consensus",
            "systematic_review",
            "original_research",
            "professional_education",
            "ordinary_web",
        ],
        ...,
    ] = Field(min_length=1, max_length=6)
    authority_note: Text
    currency_note: Text
    semantic_note: Text
    checked_dimensions: tuple[Term, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def complete_checklist(self):
        if set(self.checked_dimensions) != set(DIMENSIONS):
            raise ValueError("complete offline review dimensions required")
        if len(set(self.checked_dimensions)) != len(self.checked_dimensions):
            raise ValueError("duplicate review dimension")
        return self


class Comparison(Model):
    version: Literal["claim-comparison/1.0.0"] = "claim-comparison/1.0.0"
    relation: Literal[
        "SUPPORT",
        "CONTRADICTION",
        "POPULATION",
        "STAGE",
        "ANATOMY",
        "MODALITY",
        "METHOD",
        "THRESHOLD",
        "UNIT",
        "TIME",
        "STRENGTH",
        "UPDATE",
        "CORRECTION",
        "RETRACTION",
        "NON_EQUIVALENT",
        "UNDETERMINED",
    ]
    differences: tuple[Term, ...] = Field(max_length=32)
    reason: Text
    same_source_version: bool
    candidate_only: Literal[True] = True


class SourceContext(Model):
    publication_date: Term | None = None
    notice: Literal["none", "update", "correction", "retraction"] = "none"
    anatomy: tuple[Term, ...] = ()
