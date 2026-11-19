"""Original geometry identity and missing-input boundaries, without model execution."""

import copy
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.evaluation_preparation import generic_ports
from packages.evaluation_preparation.generic_definition_facts import measurement_basis
from packages.evaluation_preparation.quality import mask_quality
from packages.evaluation_preparation.quality_materials import quality_observation_value
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ObservationInput
from tests.evaluation_preparation.test_generic_definition_facts import fixture
from tests.evaluation_preparation.test_quality_weighting import png


def setup(monkeypatch, target="criterion:VCDR_GE_07"):
    f = fixture()
    field = {
        "criterion:VCDR_GE_07": "vertical_cdr",
        "criterion:RIM_RADIAL_LT_01": "radial_rim_minimum_ratio",
    }[target]
    row = f.state.policy.requirements[0].model_dump(mode="json")
    row["id"] = target
    image = f.base.image_artifact
    image.checksum = digest("image")
    value = dict(value=0.5, input_checksum=image.checksum)
    body = dict(
        method="registered_segman_geometry",
        unit="ratio",
        image=None,
        status="obtained",
        value=canonical(value).decode(),
    )
    body["reference"] = ref("observation", "OBS:DYNAMIC:" + target, digest(body))
    observed = validate(ObservationInput, body)
    f.state.policy.requirements = (
        SimpleNamespace(id=target, model_dump=lambda **_: copy.deepcopy(row)),
    )
    f.state.policy.items = (
        SimpleNamespace(
            id=target, method=observed.method, observation_code=observed.reference.code
        ),
    )
    f.state.material.observations = (observed,)
    f.base.directory = "original-directory"
    f.base.binding = {"case_code": "CASE:SYNTHETIC"}
    f.base.runtime, f.base.objects = object(), object()
    f.base.measurement = {"measurements": {field: 0.5}}
    cup = [0] * 36
    cup[7] = cup[8] = cup[35] = 255  # Low quality remains a usable observation.
    raw = {"disc_mask": png([255] * 36), "cup_mask": png(cup)}
    observed = observed.model_copy(
        update={
            "value": quality_observation_value(
                observed.value, mask_quality(raw["disc_mask"], raw["cup_mask"])
            )
        }
    )
    f.state.material.observations = (observed,)
    result = [{"measurements": f.base.extra_artifact}, raw, copy.deepcopy(f.base.measurement)]
    calls = []

    def read(*args):
        calls.append(args)
        assert args == (f.base.directory, f.base.binding, f.base.runtime, f.base.objects)
        return result

    monkeypatch.setattr(generic_ports, "read", read)
    return f, target, result, calls


@pytest.mark.parametrize("target", ["criterion:VCDR_GE_07", "criterion:RIM_RADIAL_LT_01"])
def test_exact_original_geometry_is_reusable_without_new_measurement_or_score(monkeypatch, target):
    f, target, result, calls = setup(monkeypatch, target)
    before = copy.deepcopy(f.state.material.observations)
    basis = measurement_basis(f.state, f.base, target)
    assert len(calls) == 1
    assert basis["observation"] == before[0].reference.model_dump(mode="json")
    assert basis["method"] == "registered_segman_geometry" and basis["unit"] == "ratio"
    assert 0 < basis["origin"]["quality"]["quality_factor"] < 1
    assert basis["origin"]["quality"]["masks_modified"] is False
    assert "e" not in basis and "scores" not in basis
    assert f.state.material.observations == before
    assert measurement_basis(f.state, f.base, target) == basis


@pytest.mark.parametrize(
    "change", ["value", "method", "unit", "missing", "changed_port", "empty_mask"]
)
def test_current_ports_and_exact_observation_are_required(monkeypatch, change):
    f, target, result, _ = setup(monkeypatch)
    if change in {"value", "method", "unit"}:
        value = {"value": "0.5", "method": "color_only", "unit": "mm"}[change]
        f.state.material.observations = (
            f.state.material.observations[0].model_copy(update={change: value}),
        )
    elif change == "missing":
        monkeypatch.setattr(generic_ports, "read", lambda *_: None)
    elif change == "changed_port":
        result[2]["measurements"]["vertical_cdr"] = 0.6
    else:
        result[1]["cup_mask"] = png([0] * 36)
    with pytest.raises(ValueError, match="generic_definition_"):
        measurement_basis(f.state, f.base, target)


def test_original_port_owner_rejection_is_not_hidden(monkeypatch):
    f, target, _, _ = setup(monkeypatch)

    def reject(*_):
        raise ValueError("generic_output_owner_binding")

    monkeypatch.setattr(generic_ports, "read", reject)
    with pytest.raises(ValueError, match="generic_output_owner_binding"):
        measurement_basis(f.state, f.base, target)


def test_geometry_fact_passes_existing_definition_owner_without_rewriting_observation(monkeypatch):
    f, target, _, _ = setup(monkeypatch)
    basis = measurement_basis(f.state, f.base, target)
    f.current["facts"] = {target: basis}
    f.owner.selected = (target,)
    f.owner.snapshot = copy.deepcopy(f.current)
    f.owner.snapshot_checksum = digest(f.current)
    f.domain.requirement_ids = (target,)
    f.candidate.update(method=basis["method"], unit=basis["unit"])
    before = copy.deepcopy(f.state.material.observations)
    receipt = f.domain.accept("geometry-interpretation-basis", f.candidate)
    assert receipt["confirmed_basis"] == basis
    assert f.domain.read("geometry-interpretation-basis") == receipt
    assert f.state.material.observations == before
    f.current["facts"][target]["origin"]["quality"]["quality_factor"] = 0.9
    with pytest.raises(ValueError, match="generic_definition_original_changed"):
        f.domain.read("geometry-interpretation-basis")
    assert f.domain.read("geometry-interpretation-basis", historical=True) == receipt
