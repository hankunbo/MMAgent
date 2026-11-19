"""Independent relation expectations; no production comparator constructs expected labels."""

import json
from pathlib import Path

import pytest

from db.knowledge import models as m
from packages.knowledge_review.conflicts import compare
from packages.knowledge_review.models import DIMENSIONS, ReviewAssessment, SourceContext

ROOT = Path(__file__).resolve().parents[2]


def candidate(text="OCT result is positive.", **changes):
    data = json.loads((ROOT / "docs/acceptance/04b5-policy/claim-demo.json").read_text())
    raw = next(
        s["record"]["candidate"] for s in data["records"] if s["record"]["kind"] == "clinical_claim"
    )
    from packages.object_store import checksum

    raw.update(
        statement=text,
        target_proposition=text,
        quote=text,
        quote_checksum=checksum(text.encode()),
        polarity="POSITIVE",
        population=[],
        disease_stage=[],
        condition=[],
        applicability=[],
        laterality=[],
        modality=[],
        exam=[],
        intervention=[],
        outcome=[],
        device=[],
        time=[],
        numbers=[],
        uncertainty=[],
        recommendation_strength=[],
    )
    raw.update(changes)
    from packages.knowledge_claims.models import ClinicalClaimCandidate

    return ClinicalClaimCandidate.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize(
    "field,value,relation",
    [
        ("population", ["adults"], "POPULATION"),
        ("disease_stage", ["early"], "STAGE"),
        ("laterality", ["left eye"], "ANATOMY"),
        ("modality", ["OCT"], "MODALITY"),
        ("exam", ["method A"], "METHOD"),
        ("device", ["device A"], "METHOD"),
        ("time", ["1 year"], "TIME"),
        ("recommendation_strength", ["strong"], "STRENGTH"),
        ("outcome", ["different endpoint"], "NON_EQUIVALENT"),
    ],
)
def test_differences_are_not_direct_contradictions(field, value, relation):
    result = compare(candidate(), candidate(**{field: value}))
    assert result.relation == relation and field in result.differences and result.candidate_only


@pytest.mark.parametrize(
    "left,right,polarity,relation",
    [
        ("OCT result is positive.", "OCT result is positive.", "POSITIVE", "SUPPORT"),
        ("OCT result is positive.", "OCT result is negative.", "NEGATIVE", "CONTRADICTION"),
        ("OCT supports evaluation.", "OCT may support evaluation.", "UNCERTAIN", "UNDETERMINED"),
        ("OCT supports evaluation.", "OCT might mean something else.", "POSITIVE", "UNDETERMINED"),
    ],
)
def test_explicit_propositions(left, right, polarity, relation):
    assert compare(candidate(left), candidate(right, polarity=polarity)).relation == relation


@pytest.mark.parametrize(
    "notice,relation",
    [("update", "UPDATE"), ("correction", "CORRECTION"), ("retraction", "RETRACTION")],
)
def test_source_notices_never_medical_negatives(notice, relation):
    assert (
        compare(candidate(), candidate(), right_context=SourceContext(notice=notice)).relation
        == relation
    )


def test_dates_anatomy_numeric_and_units_are_separate():
    assert (
        compare(
            candidate(), candidate(), right_context=SourceContext(publication_date="2026-01-01")
        ).relation
        == "TIME"
    )
    assert (
        compare(candidate(), candidate(), right_context=SourceContext(anatomy=("retina",))).relation
        == "ANATOMY"
    )

    def numeric(value, unit):
        return [
            {
                "value": value,
                "unit": unit,
                "comparator": ">=",
                "quote": value + " " + unit,
                "span": {"start": 0, "end": 6},
            }
        ]

    left = candidate("Pressure 21 mmHg.", numbers=numeric("21", "mmHg"))
    assert (
        compare(left, candidate("Pressure 22 mmHg.", numbers=numeric("22", "mmHg"))).relation
        == "THRESHOLD"
    )
    assert (
        compare(left, candidate("Pressure 21 kPa.", numbers=numeric("21", "kPa"))).relation
        == "UNIT"
    )


def test_provider_counts_and_confidence_cannot_vote():
    left = candidate()
    right = candidate("OCT result is negative.", polarity="NEGATIVE", confidence=1.0)
    for _ in range(100):
        result = compare(left, right)
        assert result.relation == "CONTRADICTION" and result.same_source_version
    assert compare(left, right.model_copy(update={"confidence": 0.0})).relation == result.relation


def test_full_checklist_required():
    with pytest.raises(ValueError, match="complete offline"):
        ReviewAssessment(
            disposition="accept",
            rationale="synthetic",
            source_classes=("original_research",),
            authority_note="synthetic",
            currency_note="synthetic",
            semantic_note="synthetic",
            checked_dimensions=("polarity",),
        )
    assert len(DIMENSIONS) == 17


def test_schema_snapshots_and_legacy_checksum():
    for model in (
        m.ClinicalClaim,
        m.ClaimConflict,
        m.ExpertReview,
        m.Release,
        m.ReleasePolicy,
        ReviewAssessment,
    ):
        assert (
            json.loads(
                (
                    ROOT / "packages/knowledge_review/schemas" / (model.__name__ + ".schema.json")
                ).read_text()
            )
            == model.model_json_schema()
        )
    data = json.loads((ROOT / "docs/acceptance/04b5-policy/claim-demo.json").read_text())
    for snapshot in data["records"]:
        parsed = m.KnowledgeSnapshot.model_validate_json(json.dumps(snapshot))
        assert parsed.model_dump(mode="json") == snapshot
