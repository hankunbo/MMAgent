import copy
from pathlib import Path

import pytest

from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.evaluation_preparation.generic_capability_selection import (
    inspect,
    observation_inventory,
    selection_profile,
    task_alias,
)
from packages.grounding.models import digest


def public():
    return dict(
        schema_version="generic-original-public-proposition/1.0.0",
        original_root_version="generic-source-case-root/2.0.0",
        original_root_key="glaucoma",
        title="glaucoma",
        target="GOAL:OID_EXPLICIT_ORIGINAL_OWNER",
        scope="declared_research_question_not_diagnosis_or_dataset_label",
        original_catalog_checksum=digest("original-source-catalog"),
        root_builder_checksum=digest("original-source-root-builder"),
    )


def intent():
    return load_intent("glaucoma_detection", Path("configs/research/multimodal_tasks_v2.json"))[0]


def test_alias_binds_vocabulary_without_relabeling_original_goal_or_scores():
    value = public()
    before = copy.deepcopy(value)
    result = task_alias(value, intent())
    assert result["canonical_target"] == value["target"]
    assert result["selection_target"] == "GOAL:GLAUCOMA_DETECTION"
    assert not result["old_scores_relabelled"]
    assert value == before


def test_actual_projection_consumes_derived_detection_profile_not_staging_template():
    task = intent()
    raw, binding, profile = selection_profile(task)
    assert raw["definition_expansion"]["public_target"] == task.target_code
    assert profile["goals"] == [task.target_code]
    assert binding["task_checksum"] == task.checksum
    assert all(task.target_code in check["goals"] for check in profile["checks"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("original_root_version", "unregistered-root/1.0.0"),
        ("original_root_key", "advanced_glaucoma"),
        ("title", "glaucoma with unknown additional goal"),
        ("scope", "patient_ground_truth"),
        ("target", "GOAL:GLAUCOMA_DETECTION"),
        ("original_catalog_checksum", None),
    ],
)
def test_no_fuzzy_goal_or_dataset_label_alias(field, value):
    origin = public()
    origin[field] = value
    with pytest.raises(ValueError, match="original_question_not_supported"):
        task_alias(origin, intent())


def test_original_inventory_keeps_missing_requirements_and_exact_observation_references():
    from tests.evaluation_preparation.test_generic_definition_projection import fixture

    _, _, state, _ = fixture()
    before = state.model_dump_json()
    rows = observation_inventory(state)
    assert {r["id"] for r in rows} == {r.id for r in state.policy.requirements}
    observed = {o.reference.code: o for o in state.material.observations}
    for row in rows:
        assert not row["score_reuse_authorized"]
        if row["observation"]:
            assert row["observation"] == observed[row["observation"]["code"]].reference.model_dump(
                mode="json"
            )
        else:
            assert row["status"] == "not_obtained"
    assert state.model_dump_json() == before


def test_inspection_rejects_caller_supplied_case_or_snapshot():
    with pytest.raises(ValueError, match="original_campaign_required"):
        inspect(object())
