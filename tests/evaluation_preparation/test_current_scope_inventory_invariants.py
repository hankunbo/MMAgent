"""Current catalog: input survey affects methods, not medical question inclusion.

Only the scope facts and public passages are synthetic. Projection, capability
selection and method planning use the production owners.
"""

import pytest

from apps.vectorengine_case_materials import ref
from packages.grounding.models import digest
from packages.llm_provider.models import Source
from packages.multimodal_research import scope_selection
from packages.multimodal_research.active_requirements import load
from tests.evaluation_preparation import test_input_scope_phase as scope_fixture
from tests.evaluation_preparation import test_scope_selection as fixture

PROFILE = "configs/research/multimodal_catalog_acquisition_v9.json"


def current_base(tmp_path, monkeypatch):
    profile = load(PROFILE)
    original = scope_fixture.source_fixture

    def sources():
        values = original()
        for key in {c["source"] for c in profile["checks"]}:
            text = "\n".join(c["quote"] for c in profile["checks"] if c["source"] == key)
            values[key] = Source(
                reference=ref("source", "SOURCE:" + key.upper(), digest(text)),
                locator="Synthetic exact current-catalog passage",
                purpose="research_material",
                state="research_only",
                text=text,
            )
        return values

    monkeypatch.setattr(fixture, "PROFILE", PROFILE)
    monkeypatch.setattr(scope_fixture, "source_fixture", sources)
    return fixture.setup(tmp_path, monkeypatch)[0]


@pytest.mark.parametrize("visibility", ["visible", "partly_visible", "uncertain", "not_identified"])
def test_current_scope_preserves_questions_and_changes_only_method_plan(
    tmp_path, monkeypatch, visibility
):
    base = current_base(tmp_path, monkeypatch)
    before = digest(base.ledger)
    records = fixture.facts(base, visibility)
    for record in records:
        for region in record["regions"]:
            region["finding"] = "Synthetic survey: no obvious abnormality in the inspected portion."
            region["clarity"] = 0.2 if visibility == "partly_visible" else region["clarity"]
    plan = scope_selection.project(base, records)
    assert digest(base.ledger) == before
    assert {r["id"] for r in plan["rows"]} == {r["id"] for r in base.ledger}
    assert plan["removed_requirements"] == [] and plan["added_disease_votes"] == 0
    assert not plan["physical_scale_authorized"]
    by_id = {r["id"]: r for r in plan["rows"]}
    assert {"active:rim_diffuse", "active:cup_geometry", "active:oct_rnfl"} <= set(by_id)
    assert by_id["active:cup_geometry"]["action"] == (
        "current_native_owner_then_qualitative_fallback"
        if visibility in {"visible", "partly_visible"}
        else "vision_scope_clarification"
    )


def test_current_unassessed_scope_retains_questions_without_inventing_observations(
    tmp_path, monkeypatch
):
    base = current_base(tmp_path, monkeypatch)
    plan = scope_selection.project(base, [])
    assert plan["removed_requirements"] == [] and plan["added_disease_votes"] == 0
    assert len(plan["rows"]) == len(base.ledger)
    cup = next(row for row in plan["rows"] if row["id"] == "active:cup_geometry")
    assert cup["state"] == "scope_not_assessed" and not cup["native_candidate"]
    assert cup["action"] == "vision_scope_clarification"
