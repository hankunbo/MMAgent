"""Visible component projection is input identity, never tool qualification."""

import importlib

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.grounding.models import digest, seal
from packages.grounding.synthetic import ref
from packages.tool_admission.models import ArtifactTypeBinding
from tests.dataset_adapters.test_multimodal import gamma, gdp


def policy(**changes):
    module = importlib.import_module("packages.research_runtime.component_inputs")
    return seal(
        module.ComponentInputPolicy,
        **dict(
            dict(
                source_kind="image",
                source_concept="color_fundus",
                source_modality="color_fundus",
                source_dtype="uint8_encoded_jpeg",
                input_binding=seal(
                    ArtifactTypeBinding,
                    version="1.0.0",
                    artifact_ref=ref("artifact", "COLOR_INPUT"),
                    contract_type_code="ARTIFACT:COLOR_FUNDUS_INPUT",
                    unit_term=dict(state="unknown", code=None),
                    contract_unit=None,
                    source_ref=ref("source_version", "SOURCE:INPUT_MAPPING"),
                ),
                modality=dict(state="known", code="CONCEPT:COLOR_FUNDUS"),
                scope_terms=("CONCEPT:RETINA",),
            ),
            **changes,
        ),
    )


def test_component_policy_and_exact_jpeg(tmp_path):
    p = policy()
    module = importlib.import_module("packages.research_runtime.component_inputs")
    sample = prepare("gamma", gamma(tmp_path / "gamma"), "0001")
    component = next(c for c in sample.manifest.components if c.kind == "image")
    payload = next(p for p in sample.manifest.payloads if p.id == component.payload_ids[0])
    raw = dict(sample.payload_bytes)[payload.id]
    assert module.check_component(component, payload, raw, p) == ("image", "PIXEL_2D")
    assert p.schema_version == "research-component-input-policy/1.0.0"
    assert p.input_binding.unit_term.state == "unknown"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"status": "missing", "reason": "unavailable", "payload_ids": ()}, "unavailable"),
        ({"status": "partial", "reason": "partial"}, "unavailable"),
        ({"kind": "image_series", "expected_members": 1, "member_indices": (0,)}, "mapping"),
        ({"modality": "SLO"}, "mapping"),
        ({"concept": "different"}, "mapping"),
        ({"dtype": "float32"}, "mapping"),
        ({"shape": (1, 1)}, "shape"),
        ({"unit": "mm", "unit_basis": "synthetic"}, "unit"),
        ({"payload_ids": ("other",)}, "payload"),
    ],
)
def test_component_mismatch_not_cast(tmp_path, change, reason):
    p = policy()
    module = importlib.import_module("packages.research_runtime.component_inputs")
    sample = prepare("gamma", gamma(tmp_path / "gamma"), "0001")
    c = next(c for c in sample.manifest.components if c.kind == "image")
    payload = next(p for p in sample.manifest.payloads if p.id == c.payload_ids[0])
    raw = dict(sample.payload_bytes)[payload.id]
    with pytest.raises(ValueError, match="component_.*" + reason):
        module.check_component(c.model_copy(update=change), payload, raw, p)


def test_component_bytes_media_and_size(tmp_path):
    p = policy()
    module = importlib.import_module("packages.research_runtime.component_inputs")
    sample = prepare("gamma", gamma(tmp_path / "gamma"), "0001")
    c = next(c for c in sample.manifest.components if c.kind == "image")
    payload = next(p for p in sample.manifest.payloads if p.id == c.payload_ids[0])
    raw = dict(sample.payload_bytes)[payload.id]
    for changed, data, reason in (
        (payload, raw + b"x", "bytes"),
        (payload.model_copy(update={"byte_size": 4194305}), raw, "size"),
        (payload.model_copy(update={"media_type": "application/json"}), raw, "media"),
    ):
        with pytest.raises(ValueError, match="component_.*" + reason):
            module.check_component(c, changed, data, p)
    with pytest.raises(ValueError):
        policy().model_copy(update={"checksum": digest("forged")}).__class__.model_validate_json(
            policy().model_copy(update={"checksum": digest("forged")}).model_dump_json()
        )


def test_array_projection_keeps_unknown_units(tmp_path):
    module = importlib.import_module("packages.research_runtime.component_inputs")
    sample = prepare("harvard_gdp", gdp(tmp_path / "gdp"), "data_0001.npz")
    c = next(c for c in sample.manifest.components if c.kind == "map")
    p = policy(
        source_kind=c.kind,
        source_concept=c.concept,
        source_modality=c.modality,
        source_dtype=c.dtype,
    )
    payload = next(p for p in sample.manifest.payloads if p.id == c.payload_ids[0])
    assert module.check_component(c, payload, dict(sample.payload_bytes)[payload.id], p) == (
        "array",
        "PIXEL_2D",
    )
    assert p.input_binding.contract_unit is None
