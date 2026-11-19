"""Selected measured facts remain exact through the existing definition owner."""

import copy
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.evaluation_preparation.generic_definition_facts import (
    ConfirmedGenericMeasurements,
    measurement_basis,
)
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ObservationInput
from tests.evaluation_preparation.test_confirmed_numeric_definition import setup


def fixture():
    domain, numeric, candidate, docs, sources = setup()
    target = numeric.base.ledger[0]["id"]
    # Independent synthetic tool result; the measurement projector supplies all
    # values and names. This input is not a saved medical answer.
    target = "descriptive:CUP_AREA_CONTEXT"
    definition = numeric.base.ledger[0]["definition"].model_dump(mode="json")
    row = dict(id=target, definition=definition, input_modalities=["new_image"])
    tools = dict(results={}, original_measurements=dict(measurements=dict(area_cdr=0.25)))
    artifact = SimpleNamespace(
        asset_code="ASSET:SYNTHETIC_TOOL",
        model_dump=lambda **_: dict(asset_code="ASSET:SYNTHETIC_TOOL", checksum=digest(tools)),
    )
    image = SimpleNamespace(
        asset_code="ASSET:SYNTHETIC_IMAGE",
        model_dump=lambda **_: dict(asset_code="ASSET:SYNTHETIC_IMAGE", checksum=digest("image")),
    )
    from packages.evaluation_preparation.comprehensive_materials import measured_projection

    measured = measured_projection(tools)[target]
    observed = dict(
        method=measured["method"],
        unit=measured["unit"],
        value=canonical(dict(measured, full_record=artifact.asset_code)).decode(),
        status="obtained",
        image=None,
    )
    observed["reference"] = ref("observation", "OBS:SYNTHETIC_AREA", digest(observed))
    observed = validate(ObservationInput, observed)
    base = SimpleNamespace(tools=tools, extra_artifact=artifact, image_artifact=image)
    state = SimpleNamespace(
        policy=SimpleNamespace(
            requirements=(SimpleNamespace(id=target, model_dump=lambda **_: copy.deepcopy(row)),),
            items=(SimpleNamespace(id=target, observation_code=observed.reference.code),),
        ),
        material=SimpleNamespace(
            observations=(observed,),
            scope=SimpleNamespace(input_checksum=domain.binding["input_checksum"]),
            goal_code=domain.binding["target"],
        ),
    )
    basis = measurement_basis(state, base, target)
    owner = object.__new__(ConfirmedGenericMeasurements)
    current = dict(
        input_checksum=domain.binding["input_checksum"],
        target=domain.binding["target"],
        facts={target: basis},
    )
    owner.selected = (target,)
    owner.snapshot = copy.deepcopy(current)
    owner.snapshot_checksum = digest(owner.snapshot)
    owner._read = lambda: copy.deepcopy(current)
    visible = SimpleNamespace(checksum=domain.binding["input_checksum"])
    base.visible_value = visible
    base.visible = SimpleNamespace(read=lambda _: visible)
    base.binding = dict(case_code="CASE:SYNTHETIC")
    owner.campaign = SimpleNamespace(base=base)
    domain.confirmed_observations = owner
    domain.requirement_ids = (target,)
    domain.input_reader = owner.input_view
    domain.fact_reader = owner.dependency_fact
    candidate.update(method=measured["method"], unit=measured["unit"])
    return SimpleNamespace(
        domain=domain,
        owner=owner,
        candidate=candidate,
        docs=docs,
        sources=sources,
        current=current,
        state=state,
        base=base,
        target=target,
    )


def test_original_definition_acceptance_and_historical_read_keep_measurement_exact():
    f = fixture()
    original = copy.deepcopy(f.state.material.observations)
    receipt = f.domain.accept("same-measurement-new-interpretation", f.candidate)
    assert receipt["confirmed_basis"]["method"] == "same_mask_area_cdr"
    assert receipt["confirmed_basis"]["unit"] == "ratio"
    assert receipt["confirmed_basis"]["origin"]["projection_component"] == "original_measurements"
    assert f.domain.read("same-measurement-new-interpretation") == receipt
    assert f.state.material.observations == original
    assert f.owner.input_view().components == ()
    assert f.owner.dependency_basis(f.target)["technical_checks"] == {}
    f.current["facts"][f.target]["origin"]["tool_result_checksum"] = digest("changed")
    with pytest.raises(ValueError, match="generic_definition_original_changed"):
        f.domain.read("same-measurement-new-interpretation")
    assert f.domain.read("same-measurement-new-interpretation", historical=True) == receipt


@pytest.mark.parametrize(
    "field,value", [("method", "vertical_cdr"), ("unit", "mm"), ("target", "GOAL:OTHER")]
)
def test_unrelated_target_method_or_unit_cannot_become_current_definition(field, value):
    f = fixture()
    f.candidate[field] = value
    with pytest.raises(
        ValueError, match="definition_confirmed_requirement_method|definition_draft_target"
    ):
        f.domain.accept("bad", f.candidate)
    assert f.docs.read("definition:bad") is None


@pytest.mark.parametrize("change", ["value", "method", "unit", "missing_tool"])
def test_projection_cannot_substitute_a_different_measurement(change):
    f = fixture()
    if change == "missing_tool":
        f.base.extra_artifact = None
    else:
        old = f.state.material.observations[0]
        value = dict(value="0.1", method="another_method", unit="mm")[change]
        f.state.material.observations = (old.model_copy(update={change: value}),)
    with pytest.raises(ValueError, match="generic_definition_.*"):
        measurement_basis(f.state, f.base, f.target)


def test_caller_object_and_unselected_requirement_do_not_grant_method_authority():
    with pytest.raises(ValueError, match="original_campaign_owner_required"):
        ConfirmedGenericMeasurements(SimpleNamespace(), ("some-item",))
    f = fixture()
    with pytest.raises(ValueError, match="requirement_scope"):
        f.owner.interpretation_basis("not-selected")
    with pytest.raises(ValueError, match="additional_input_owner_required"):
        f.owner.dependency_fact(None)
    f.owner.snapshot["target"] = "GOAL:OTHER"
    with pytest.raises(ValueError, match="snapshot_changed"):
        f.owner.current()


def test_actual_source_wire_uses_exact_method_without_case_measurement_or_identifiers():
    import json

    from packages.llm_provider.config import prompt
    from packages.llm_provider.definition_passages import SelectedDefinitionProposal
    from packages.llm_provider.transport import wire_request
    from packages.multimodal_research.campaign_renewal import compact_configuration
    from packages.multimodal_research.definition_expansion import definition_config

    f = fixture()
    payload = f.owner.draft_context(materials=[], public_queries=["cup area measurement relation"])
    choice = payload["candidate_contract_choices"]["confirmed_observation"][0]["copy_fields"]
    assert choice["method"] == "same_mask_area_cdr" and choice["unit"] == "ratio"
    assert payload["candidate_contract_choices"]["raw_input"] == []
    config = definition_config(compact_configuration(), prompt_version="definition-draft-6.8.0")
    role = next(r for r in config.roles if r.role == "source")
    body = wire_request(
        config.transport,
        role,
        prompt(config, "source")[1],
        payload,
        SelectedDefinitionProposal.model_json_schema(),
        (),
        wire_policy=config.wire_policy,
        wire_mode=config.wire_mode,
    )
    text = json.dumps(body)
    assert "same_mask_area_cdr" in text
    assert "same_cup_description_not_independent_disease_evidence" in text
    for forbidden in [
        "0.25",
        "ASSET:SYNTHETIC_TOOL",
        "ASSET:SYNTHETIC_IMAGE",
        "OBS:SYNTHETIC_AREA",
    ]:
        assert forbidden not in text
    assert "new acquisition" in text
