"""Public per-requirement semantics and explicit new Provider/trial versions."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest
from packages.llm_provider import DeterministicFakeProvider
from packages.multimodal_research.definition_context import (
    public_examination,
    public_gap_context,
    requirement_context,
    same_examination,
)
from packages.multimodal_research.definition_expansion import (
    definition_config,
    read_expansion_policy,
)
from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture
from tests.evaluation_preparation.test_directional_review import fixture
from tests.evaluation_preparation.test_trial_entry_window import prepare_entry


def rows():
    definition = project_review_tasks(fixture().owner.current()).tasks[0].definition
    return [
        dict(
            id="requirement:a",
            component="PRIVATE_COMPONENT",
            definition=definition,
            input_modalities=["synthetic-image"],
            acquisition_method="vision",
            note="PRIVATE_NOTE",
            patient_value=123456.7,
        )
    ]


def test_public_mapping_excludes_values_and_actual_component_identity():
    original = rows()
    other = copy.deepcopy(original[0])
    other.update(id="requirement:b", component="OTHER_PRIVATE_COMPONENT")
    result = requirement_context(
        original + [other], [{"target_relation_gap_ids": ["requirement:a"]}], {}
    )
    first, second = result["requirements"]
    assert first["component"] != second["component"]
    assert first["examination"] == second["examination"]
    assert first["applicability_conditions"] == original[0]["definition"].applicability_conditions
    assert first["requested_revision"] == "target_relation"
    assert first["method"] == "live_image_observation" and first["unit"] == "qualitative"
    assert "PRIVATE" not in json.dumps(result) and "123456.7" not in json.dumps(result)
    assert result["empty_proposal_allowed"] and not result["duplicate_vote_allowed"]


def test_numeric_owner_name_is_public_without_losing_component_equivalence():
    from packages.multimodal_research.material import RULES, definition

    values = rows()
    component = "PRIVATE_COMPONENT_PATIENT"
    name, _, interpretation, _ = RULES["age"]
    source = SimpleNamespace(reference=values[0]["definition"].source_refs[0])
    original = definition("check:" + component, name + " · " + component, interpretation, (source,))
    values[0].update(
        id="check:" + component,
        component=component,
        definition=original,
        acquisition_method="source_measurement",
    )
    context = requirement_context(values, [], {})
    assert context["requirements"][0]["examination"] == name
    assert component not in json.dumps(context)
    projected = dict(values[0], definition=definition("new", name, interpretation, (source,)))
    assert same_examination(values[0], projected)
    assert not same_examination(values[0], dict(projected, component="different"))
    assert public_examination(values[0]) == name
    assert original.examination == name + " · " + component


def test_exact_method_metadata_does_not_relabel_measurement():
    values = rows()
    values[0]["acquisition_method"] = "disc_cup_measurement"
    result = requirement_context(
        values,
        [],
        {
            "requirement:a": SimpleNamespace(
                method="confirmed_fundus_disc_cup_geometry", unit="ratio", value="PRIVATE"
            )
        },
    )
    assert result["requirements"][0]["method"] == "confirmed_fundus_disc_cup_geometry"
    assert result["requirements"][0]["acquisition_method"] == "disc_cup_measurement"
    assert "PRIVATE" not in json.dumps(result)


def test_public_gap_ids_match_context_aliases_without_component_suffix():
    values = rows()
    values[0]["id"] = "active:feature:PRIVATE_COMPONENT_PATIENT"
    plans = [
        dict(
            query="Public question",
            selected=True,
            target_relation_gap_ids=[values[0]["id"]],
            unbound_interpretation_ids=[values[0]["id"]],
            missing_definition_ids=["PRIVATE_MISSING"],
            private_note="PRIVATE_NOTE",
        )
    ]
    before = copy.deepcopy(plans)
    context = requirement_context(values, plans, {})
    gaps = public_gap_context(values, plans)
    assert plans == before
    alias = context["requirements"][0]["requirement_id"]
    assert gaps[0]["target_relation_gap_ids"] == gaps[0]["unbound_interpretation_ids"] == [alias]
    assert gaps[0]["missing_definition_ids"] == ["missing-definition:0"]
    assert "PRIVATE" not in json.dumps(dict(context=context, gaps=gaps))
    with pytest.raises(ValueError, match="unknown_requirement"):
        public_gap_context(values, [{"target_relation_gap_ids": ["unknown"]}])


@pytest.mark.parametrize(
    "field,expected", [("expected_method", "method"), ("expected_unit", "unit")]
)
def test_method_and_unit_mismatch_reject(field, expected):
    values = rows()
    values[0][field] = "wrong"
    with pytest.raises(ValueError, match="definition_context_" + expected + "_binding"):
        requirement_context(
            values, [], {"requirement:a": SimpleNamespace(method="real", unit="ratio")}
        )


def test_duplicate_unknown_gap_and_tampered_definition_reject():
    values = rows()
    with pytest.raises(ValueError, match="duplicate_requirement"):
        requirement_context(values + values, [], {})
    with pytest.raises(ValueError, match="unknown_requirement"):
        requirement_context(values, [{"unbound_interpretation_ids": ["not-an-id"]}], {})
    values[0]["definition"] = values[0]["definition"].model_copy(
        update={"interpretation": "altered"}
    )
    with pytest.raises(ValueError, match="task_definition_checksum"):
        requirement_context(values, [], {})


def test_new_provider_wire_selects_original_evidence_schema(tmp_path):
    e = expansion_fixture(tmp_path)
    config = definition_config(
        e.provider.config, prompt_version="definition-draft-4.1.0", source_output_tokens=8192
    )
    context = requirement_context(rows(), [], {})
    seen = []

    def response(body):
        seen.append(body)
        assert "existing_requirement_context" in json.dumps(body)
        assert "input-component:0" in json.dumps(body)
        assert "PRIVATE" not in json.dumps(body)
        return dict(
            schema_version="research-definition-proposal/3.0.0",
            definitions=[],
            no_addition_reason="No new supported relationship in this synthetic input",
        )

    provider = DeterministicFakeProvider(config, [response])
    result = provider.generate_definitions(
        payload=dict(existing_requirement_context=context),
        scope=e.scope,
        budget=e.budget,
        before_request=lambda: None,
    )
    assert len(seen) == 1 and not result.definitions
    assert e.budget.used == 1


def test_new_policy_trial_is_explicit_and_parent_bounded(tmp_path, monkeypatch):
    from packages.multimodal_research import definition_trial

    _, spec = prepare_entry(tmp_path, monkeypatch)
    policy = "configs/research/multimodal_definition_gate_v19.json"
    value = definition_trial.prepare(
        tmp_path,
        expansion_policy=policy,
        authorization="synthetic explicit context contract test",
        seconds=5400,
        entry_time_window=True,
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.15.0"
    assert definition_trial.read(tmp_path) == value
    spec["seconds"] = 3600
    value["entry_checksum"] = digest(spec)
    with pytest.raises(ValueError, match="entry_time_window"):
        definition_trial.checked(value, spec)
    old = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v18.json").read_bytes())
    )
    new = read_expansion_policy(json.loads(Path(policy).read_bytes()))
    assert old.prompt_version == "definition-draft-4.0.0"
    assert new.prompt_version == "definition-draft-4.1.0"
