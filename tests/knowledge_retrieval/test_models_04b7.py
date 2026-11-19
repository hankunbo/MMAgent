"""Strict, versioned request and scope boundaries; no medical inference."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from apps.knowledge_retrieval import PublishedKnowledgeService, SourceCandidateService
from db.knowledge import models as m
from packages.retrieval.knowledge.models import RetrievalContext
from packages.retrieval.knowledge.scope import ClinicalScope


@pytest.mark.parametrize(
    "values",
    [
        {"age_years": "30"},
        {"age_years": True},
        {"age_years": -1},
        {"age_years": 151},
        {"age_years": 30.5},
        {"patient_id": "forbidden"},
        {"run_id": "forbidden"},
        {"date_from": "2026-02-31"},
        {"date_from": "2026-06-01", "date_to": "2026-01-01"},
    ],
)
def test_invalid_context(values):
    with pytest.raises(ValidationError):
        RetrievalContext(**values)


@pytest.mark.parametrize(
    "values", [{"age_min": 80, "age_max": 18}, {"version": "future"}, {"age_min": True}]
)
def test_invalid_scope(values):
    with pytest.raises(ValidationError):
        ClinicalScope(**values)


def test_ports_separate_and_legacy_condition_bytes():
    assert not hasattr(PublishedKnowledgeService, "search_source_candidates")
    assert not hasattr(SourceCandidateService, "retrieve_published_knowledge")
    old = m.seal(
        m.ApplicabilityCondition,
        code="SOURCE:SYNTHETIC_B7",
        version="1.0.0",
        population=("synthetic",),
        scope_note="Synthetic",
        synthetic=True,
    )
    assert "retrieval_scope" not in old.model_dump(mode="json")
    assert m.revalidate_record(old) == old
    current = m.seal(
        m.ApplicabilityCondition,
        code="SOURCE:SYNTHETIC_B7",
        version="1.1.0",
        population=("synthetic",),
        scope_note="Synthetic",
        synthetic=True,
        retrieval_scope=ClinicalScope(age_min=18),
        supersedes=old.reference(),
    )
    assert current.checksum != old.checksum


def test_schema_snapshots():
    folder = Path(__file__).resolve().parents[2] / "packages/retrieval/knowledge/schemas"
    for model in (ClinicalScope, RetrievalContext, m.ApplicabilityCondition, m.Template):
        assert (
            json.loads((folder / (model.__name__ + ".schema.json")).read_text())
            == model.model_json_schema()
        )
