"""Local quote coordinates and available method input, without leaking patient values."""

from types import SimpleNamespace

from packages.multimodal_research.definition_expansion import located_proposal_context


def component(kind, *, status="obtained", value=None, unit=None):
    return SimpleNamespace(
        kind=kind,
        modality="OCT" if kind == "volume" else "clinical",
        concept="raw_volume" if kind == "volume" else "thickness",
        status=status,
        value=value,
        unit=unit,
        id="PRIVATE_PATIENT_ID",
        source_locator="PRIVATE_PATH",
        payload_ids=(),
    )


def test_coordinates_are_local_codepoints_and_document_offsets_are_not_sent():
    text = "真实原文 αβ。"
    original = dict(
        code="SOURCE:X",
        checksum="sha256:exact",
        text=text,
        position=dict(span_start=5550, span_end=5550 + len(text)),
        hit={"private_internal": "not_for_model"},
    )
    payload = located_proposal_context(
        dict(source_materials=[original]),
        SimpleNamespace(
            manifest=SimpleNamespace(components=()), prepared=SimpleNamespace(payload_bytes=())
        ),
    )
    material = payload["source_materials"][0]
    assert material["text"] == text and "position" not in material and "hit" not in material
    assert material["whole_text_excerpt"] == dict(
        source_code="SOURCE:X", source_checksum="sha256:exact", start=0, end=len(text), text=text
    )
    assert original["position"]["span_start"] == 5550
    assert payload["citation_contract"]["coordinate_space"] == "source_text_unicode_codepoints"


def test_modalities_do_not_imply_numeric_measurements_and_no_patient_values_are_sent():
    base = SimpleNamespace(
        manifest=SimpleNamespace(
            components=(component("volume"), component("scalar", value=73.2, unit="um"))
        ),
        prepared=SimpleNamespace(payload_bytes=()),
    )
    payload = located_proposal_context(dict(source_materials=[]), base)
    image, scalar = payload["input_capabilities"]
    assert image["installed_methods"] == ["live_image_observation"]
    assert scalar["installed_methods"] == ["source_numeric_read"] and scalar["unit"] == "um"
    assert "PRIVATE" not in str(payload) and "73.2" not in str(payload)


def test_missing_or_withheld_component_does_not_grant_an_acquisition_method():
    base = SimpleNamespace(
        manifest=SimpleNamespace(
            components=(
                component("volume", status="missing"),
                component("scalar", status="withheld"),
            )
        ),
        prepared=SimpleNamespace(payload_bytes=()),
    )
    payload = located_proposal_context(dict(source_materials=[]), base)
    assert len(payload["input_capabilities"]) == 1
    assert payload["input_capabilities"][0]["installed_methods"] == []


def test_new_policy_selects_explicit_prompt_without_changing_old_default():
    import json
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import read_expansion_policy

    old = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v3.json").read_bytes())
    )
    new = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v4.json").read_bytes())
    )
    assert old.prompt_version == "definition-draft-2.1.0"
    assert new.prompt_version == "definition-draft-2.2.0"
    assert old.max_new_observations == new.max_new_observations == 1
    assert old.max_rounds == new.max_rounds == 1
