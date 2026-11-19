import importlib

import pytest

from packages.evaluation_preparation.evidence_tree_v3 import build
from packages.evaluation_preparation.generic_capability_projection import project_demands
from packages.evaluation_preparation.report_production import produce
from packages.grounding.models import digest


def fixture():
    original = importlib.import_module(
        "tests.evaluation_preparation.test_generic_capability_projection"
    )
    state, projection = project_demands(*original.fixture())
    projection["phase_input_checksum"] = digest(state)
    report = dict(
        schema_version="dynamic-case-report/1.0.0",
        result=dict(case_code=state.material.scope.case_code),
        scores=None,
        original_review=None,
        capability_revision=dict(
            schema_version="generic-capability-wholecase-report-scope/1.0.0",
            projection=projection,
            whole_case_aggregation_required=False,
        ),
        requirements=[
            dict(id=r.id, sources=[s.model_dump(mode="json") for s in r.definition.source_refs])
            for r in state.policy.requirements
        ],
    )
    return state, report


def test_new_missing_and_reused_demands_are_distinct_in_saved_six_sections():
    state, report = fixture()
    before = digest(report)
    produced = produce(report, state)
    rows = {row["id"]: row for row in produced["requirements"]}
    new = rows["active:NEW"]["presentation_sections"]["rationale"]["capability_selection"]
    missing = rows["active:MISSING"]["presentation_sections"]["rationale"]["capability_selection"]
    assert new["selected_for_acquisition"] and not missing["selected_for_acquisition"]
    assert new["denominator_after"] == new["denominator_before"] + 2
    assert new["observation_completion_not_inferred"]
    assert produced["scores"] is None and digest(report) == before
    assert build(produced, input_checksum=digest(produced))["checks"]


@pytest.mark.parametrize(
    "fault", ["version", "state", "case", "count", "duplicate", "retained", "scope", "dispatch"]
)
def test_foreign_or_inconsistent_selection_not_presented_as_current(fault):
    state, report = fixture()
    revision = report["capability_revision"]
    p = revision["projection"]
    if fault == "version":
        revision["schema_version"] = "unknown"
    if fault == "state":
        p["phase_input_checksum"] = digest("changed")
    if fault == "case":
        p["case_code"] = "CASE:OTHER"
    if fault == "count":
        p["requirements_after"] += 1
    if fault == "duplicate":
        p["added_requirement_ids"] *= 2
    if fault == "retained":
        p["retained_requirement_ids"] = []
    if fault == "scope":
        p["covered_by_existing_scope"][0]["existing"] = "foreign"
    if fault == "dispatch":
        p["dispatch_targets"].append("foreign")
    with pytest.raises(ValueError, match="report_capability_"):
        produce(report, state)
