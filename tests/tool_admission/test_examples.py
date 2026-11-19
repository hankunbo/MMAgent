"""Saved synthetic JSON must remain strict, explanatory and non-authorizing."""

import json
from pathlib import Path

import pytest

from packages.grounding.models import validate
from packages.grounding.query import normalized_request
from packages.grounding.query_models import QueryRequest, QueryResult
from packages.tool_admission import AdmissionResult, AdmissionSelection, InputRecord

ROOT = Path(__file__).resolve().parents[2] / "docs/acceptance/06b/examples"


@pytest.mark.parametrize("name", ["retained", "rejected", "gaps", "research"])
def test_strict_saved_admission_examples(name):
    result = validate(AdmissionResult, json.loads((ROOT / f"{name}.synthetic.json").read_text()))
    assert not result.execution_authorized and not result.decisive_evidence
    if name == "retained":
        assert result.status == "preparation_only" and len(result.retained) == 3
    if name == "rejected":
        assert result.capability_gaps and result.status == "rejected"
    if name == "gaps":
        assert result.product_demands and not result.retained
    if name == "research":
        assert result.purpose == "shadow" and result.status == "preparation_only"


def test_strict_input_example():
    data = json.loads((ROOT / "input.synthetic.json").read_text())
    q = validate(QueryRequest, data["request"])
    result = validate(QueryResult, data["query_result"])
    selection = validate(AdmissionSelection, data["selection"])
    assert result.request == normalized_request(q) and selection.candidate_refs
    for value in data["inputs"]:
        record = validate(InputRecord, value)
        assert record.metadata in q.artifacts
