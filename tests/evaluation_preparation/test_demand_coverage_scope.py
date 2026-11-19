"""Saved-report migration distinguishes missing inputs, definitions and aliases."""

import json
from copy import deepcopy

import pytest

from packages.multimodal_research.demand_coverage import (
    LEGACY_VERSION,
    VERSION,
    inspect_saved,
    legacy_project,
    project,
    render_summary,
)
from tests.evaluation_preparation.test_demand_coverage import report


def test_explicit_input_missing_is_not_inferred_from_method_absence():
    data = report()
    data["requirements"][1]["source_requirement"] = dict(
        component=None, missing_reason="patient_input_missing:VF"
    )
    data["requirements"][2]["source_requirement"] = dict(
        component=None, missing_reason="method_unavailable"
    )
    actual = project(data)
    assert actual["rows"][1]["state"] == "input_missing"
    assert actual["rows"][2]["state"] == "unresolved_requirement"
    assert actual["counts"]["input_missing"] == 1


def test_saved_definition_gap_does_not_create_review_or_change_score():
    data = report()
    data["requirements"][2].update(
        observation=dict(status="obtained"),
        source_requirement=dict(
            review_allowed=False, definition=dict(interpretation_status="missing")
        ),
    )
    before = deepcopy(data)
    actual = project(data)
    assert data == before
    assert actual["rows"][2]["state"] == "obtained_definition_pending"
    assert actual["rows"][2]["e"] is None
    assert actual["observation_accounting"]["current_obtained_without_review"] == ["three"]
    assert not actual["observation_accounting"]["all_current_obtained_have_review"]
    assert actual["observation_accounting"]["saved_completion_claim"] is True
    assert "定义或目标关系待补" in render_summary(actual)


def test_alias_only_comes_from_saved_plan_and_never_adds_a_vote():
    data = report()
    data["acquisition_observations"]["archived"] = dict(status="obtained")
    assert project(data)["observation_accounting"]["declared_shared_aliases"] == []
    data["active_acquisition"] = dict(
        plan=dict(
            measurement_fulfillment=dict(
                outcomes=[dict(requirement="archived", fulfilled_by="one", state="shared")]
            )
        )
    )
    actual = project(data)
    assert actual["counts"]["scored"] == 1
    assert actual["observation_accounting"]["archival_obtained_ids"] == ["archived"]
    assert actual["observation_accounting"]["declared_shared_aliases"] == [
        dict(
            archived_requirement_id="archived",
            current_requirement_id="one",
            current_requirement_reviewed=True,
        )
    ]
    assert actual["observation_accounting"]["independent_evidence_count"] is None


@pytest.mark.parametrize("version", [LEGACY_VERSION, VERSION])
def test_exact_old_or_new_projection_verified_without_rewriting_bytes(tmp_path, version):
    data = report()
    data["demand_coverage"] = (legacy_project if version == LEGACY_VERSION else project)(data)
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    result = inspect_saved(path)
    assert result["original_projection_verified"]
    assert result["saved_coverage_version"] == version
    assert result["coverage"]["schema_version"] == VERSION
    assert path.read_bytes() == before
    assert result["api_requests"] == result["database_writes"] == 0


@pytest.mark.parametrize("fault", ["score", "coverage", "version"])
def test_old_report_tampering_or_unknown_version_still_rejected(tmp_path, fault):
    data = report()
    data["demand_coverage"] = legacy_project(data)
    if fault == "score":
        data["scores"]["reviews"][0]["judgment"]["value"] = 0.2
    elif fault == "coverage":
        data["demand_coverage"]["counts"]["reviewed"] = 200
    else:
        data["demand_coverage"]["schema_version"] = "unknown/99"
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="demand_coverage_saved"):
        inspect_saved(path)
