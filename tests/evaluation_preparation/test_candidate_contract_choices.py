"""Source format guidance is owner-bound; it never repairs a candidate answer."""

import copy
import json
from pathlib import Path

import pytest

from packages.multimodal_research.definition_context import candidate_contract_choices
from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.evaluation_preparation.test_confirmed_interpretation_definition import (
    exercise_source_proposal,
)


def payload():
    return dict(
        input_capabilities=[
            dict(
                input_alias="input:0",
                modality="OCT",
                concept="volume",
                installed_methods=["live_image_observation"],
                unit=None,
            ),
            dict(
                input_alias="input:1",
                modality="clinical",
                concept="thickness",
                installed_methods=["source_numeric_read"],
                unit="um",
            ),
            dict(
                input_alias="input:2",
                modality="fundus",
                concept="image",
                installed_methods=[],
                unit=None,
            ),
        ],
        confirmed_observation_capabilities=[],
        existing_requirement_context=dict(requirements=[]),
    )


def test_raw_choices_do_not_invent_confirmed_observations_or_medical_answers():
    source = payload()
    before = copy.deepcopy(source)
    choices = candidate_contract_choices(source)["candidate_contract_choices"]
    assert source == before
    assert choices["confirmed_observation"] == []
    assert len(choices["raw_input"]) == 2
    image, numeric = choices["raw_input"]
    assert image["copy_fields"]["method"] == "live_image_observation"
    assert image["copy_fields"]["unit"] == "qualitative"
    assert image["input_condition_selectors"]["method"] is None
    assert numeric["copy_fields"]["unit"] == "um"
    assert numeric["input_condition_selectors"]["predicate"] == "source_numeric_available"
    assert all("interpretation" not in item["copy_fields"] for item in choices["raw_input"])
    assert choices["missing_input_definitions_remain_allowed_with_explicit_unresolved_conditions"]


@pytest.mark.parametrize(
    "method,unit",
    [(None, "ratio"), ({"name": "measure"}, "ratio"), ("measure", None), ("wrong", "ratio")],
)
def test_confirmed_choices_reject_missing_or_mismatched_owner_metadata(method, unit):
    source = payload()
    source["existing_requirement_context"]["requirements"] = [
        dict(
            requirement_id="requirement:0",
            method="measure",
            unit="ratio",
            examination="Synthetic measurement",
            input_modalities=["OCT"],
        )
    ]
    source["confirmed_observation_capabilities"] = [
        dict(
            requirement_alias="requirement:0",
            method=method,
            unit=unit,
        )
    ]
    with pytest.raises(ValueError, match="definition_contract_choice_confirmed_binding"):
        candidate_contract_choices(source)


def test_actual_source_wire_owner_revision_and_idempotent_recovery(tmp_path):
    exercise_source_proposal(tmp_path, typed=True, candidate_choices=True)


@pytest.mark.parametrize(
    "old,new",
    [
        (
            "multimodal_harvard_visual_priority_v3.json",
            "multimodal_harvard_visual_priority_v4.json",
        ),
        ("multimodal_structural_coverage_v1.json", "multimodal_structural_coverage_v2.json"),
        (
            "multimodal_grape_active_requirements_v4.json",
            "multimodal_grape_active_requirements_v5.json",
        ),
    ],
)
def test_new_profiles_change_only_explicit_request_contract_not_budgets(old, new):
    root = Path("configs/research")
    previous = json.loads((root / old).read_bytes())
    current = json.loads((root / new).read_bytes())
    # The profile name and Source request contract are the only deliberate delta.
    assert current.pop("id") != previous.pop("id")
    policy = current["definition_expansion"]
    assert read_expansion_policy(policy).prompt_version == "definition-draft-6.4.0"
    assert policy["schema_version"] == "campaign-definition-expansion/1.28.0"
    policy["schema_version"] = previous["definition_expansion"]["schema_version"]
    policy["prompt_version"] = previous["definition_expansion"]["prompt_version"]
    assert current == previous
