"""Representation facts do not authorize medical observations or measurements."""

import json
import sys
from dataclasses import replace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.analysis import InputPlan, build
from packages.dataset_adapters.multimodal.capabilities import (
    InputCapabilities,
    from_preparation,
    project,
    read_snapshot,
    write_snapshot,
)
from packages.dataset_adapters.multimodal.models import CaseComponents
from packages.dataset_adapters.multimodal.preparation import persist
from packages.disease_packs.models import seal
from tests.dataset_adapters.test_multimodal import gamma, gdp, grape


def projected(prepared):
    return project(prepared.manifest, build(prepared.manifest, dict(prepared.payload_bytes)))


def abilities(component):
    return {a.name: a for a in component.abilities}


def reseal(model, value, **changes):
    payload = dict(vars(value))
    payload.pop("checksum")
    payload.update(changes)
    return seal(model, **payload)


def test_gamma_raw_series_is_not_thickness_or_visible_region(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    value = projected(p)
    image, oct_volume = value.components
    assert image.modality == "color_fundus"
    assert abilities(image)["read_image"].status == "available"
    assert oct_volume.representation == "image_series"
    assert oct_volume.acquisition_fraction == 3 / 256
    assert oct_volume.acquisition_basis == "provided_slice_count_not_tissue_visibility"
    assert oct_volume.anatomical_region == "unknown"
    assert oct_volume.visual_scope == "not_assessed"
    assert "read_source_measurement" not in abilities(oct_volume)
    assert abilities(oct_volume)["absolute_image_measurement"].status == "conditional"
    assert abilities(oct_volume)["qualitative_visual_observation"].status == "conditional"
    assert not value.execution_authorized and value.api_requests == 0


def test_missing_photo_does_not_discard_available_oct(tmp_path):
    p = prepare("gamma", gamma(tmp_path, photo=False), "0001")
    image, series = projected(p).components
    assert image.input_status == "missing" and not image.payload_checksums
    assert all(a.status == "unavailable" for a in image.abilities)
    assert abilities(series)["read_image_series"].status == "available"


def test_source_units_do_not_become_image_calibration(tmp_path):
    root = gdp(tmp_path)
    original = projected(prepare("harvard_gdp", root, "data_0001.npz"))
    qualified = projected(
        prepare("harvard_gdp", root, "data_0001.npz", semantic_profile="harvard-gdp-v1")
    )
    before = next(c for c in original.components if c.concept == "rnflt")
    after = next(c for c in qualified.components if c.concept == "rnflt")
    assert before.representation == after.representation == "map"
    assert abilities(before)["read_source_measurement"].status == "conditional"
    assert abilities(after)["read_source_measurement"].status == "available"
    assert after.unit == "um" and after.unit_basis
    assert after.physical_image_calibration == "not_established"
    assert "read_volume" not in abilities(after)
    assert "qualitative_visual_observation" not in abilities(after)


def test_case_identity_not_a_capability_selector(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    first = projected(p)
    # Synthetic boundary test, not a relabelled real patient or authorized run.
    other = reseal(
        CaseComponents,
        p.manifest,
        dataset="harvard_gdp",
        task="glaucoma_detection",
        selection="development_preparation",
        sample="synthetic_alias",
    )
    second = projected(replace(p, manifest=other))
    assert first.input_checksum != second.input_checksum
    assert first.components == second.components


def test_zero_patient_value_is_available_and_not_a_negative_diagnosis(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path, age=0.0), "data_0001.npz")
    age = next(c for c in projected(p).components if c.concept == "age")
    assert abilities(age)["read_scalar"].status == "available"
    assert "value" not in age.model_dump()


def test_grape_laterality_and_visits_are_preserved(tmp_path):
    p = prepare("grape", grape(tmp_path), "1:OD", cutoff_visit=1)
    result = projected(p)
    assert {(c.component_id, c.laterality, c.visit) for c in result.components} == {
        (c.id, c.laterality, c.visit) for c in p.manifest.components
    }
    assert len({c.visit for c in result.components}) > 1


def test_original_owner_rechecks_payload_and_never_overwrites(tmp_path):
    p = prepare("gamma", gamma(tmp_path / "data"), "0001")
    path = persist(p, tmp_path / "prepared")
    output = tmp_path / "capabilities.json"
    first = write_snapshot(path, output)
    before = output.read_bytes()
    assert InputCapabilities.model_validate_json(before) == first
    assert read_snapshot(path, output) == first
    with pytest.raises(FileExistsError):
        write_snapshot(path, output)
    assert output.read_bytes() == before
    payload = next((path.parent / "payloads").iterdir())
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError):
        from_preparation(path)


def test_valid_checksum_does_not_authorize_invented_capabilities(tmp_path):
    p = prepare("gamma", gamma(tmp_path / "data"), "0001")
    path = persist(p, tmp_path / "prepared")
    output = tmp_path / "capabilities.json"
    first = write_snapshot(path, output)
    component = first.components[0]
    altered = component.model_copy(
        update={
            "abilities": tuple(
                a.model_copy(update={"status": "available"}) for a in component.abilities
            )
        }
    )
    fake = reseal(InputCapabilities, first, components=(altered, *first.components[1:]))
    # An internally consistent checksum remains only integrity, never source truth.
    output.write_text(fake.model_dump_json())
    with pytest.raises(ValueError, match="capability_snapshot_changed"):
        read_snapshot(path, output)


def test_audit_binding_and_constructed_objects_rejected(tmp_path):
    p = prepare("gamma", gamma(tmp_path), "0001")
    audit = build(p.manifest, dict(p.payload_bytes))
    changed = reseal(InputPlan, audit, input_checksum="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="capability_audit_input_binding"):
        project(p.manifest, changed)
    with pytest.raises(ValueError, match="checksum"):
        project(p.manifest.model_copy(update={"sample": "forged"}), audit)
    with pytest.raises(ValueError, match="capability_audit_inventory"):
        project(p.manifest, reseal(InputPlan, audit, components=audit.components[:-1]))


def test_public_command_emits_no_authorization(tmp_path, monkeypatch, capsys):
    from apps.multimodal_case import main

    p = prepare("gamma", gamma(tmp_path / "data"), "0001")
    path = persist(p, tmp_path / "prepared")
    output = tmp_path / "capabilities.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["multimodal_case", "capability-plan", "--input", str(path), "--output", str(output)],
    )
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["api_requests"] == summary["database_writes"] == summary["model_runs"] == 0
    assert summary["qualification"] == "representation_only_visual_scope_pending"
    assert (
        InputCapabilities.model_validate_json(output.read_bytes()).input_checksum
        == p.manifest.checksum
    )


def test_public_prepare_uses_original_explicit_semantics(tmp_path, monkeypatch, capsys):
    from apps.multimodal_case import main
    from packages.dataset_adapters.multimodal.preparation import verify

    root = gdp(tmp_path / "data")
    out = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "prepare",
            "--dataset",
            "harvard_gdp",
            "--root",
            str(root),
            "--sample",
            "data_0001.npz",
            "--input-semantics",
            "harvard-gdp-v1",
            "--output",
            str(out),
        ],
    )
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    saved = verify(summary["output"])
    assert saved.adapter_version == "multimodal-adapters/1.1.0"
    assert next(c for c in saved.components if c.concept == "rnflt").unit == "um"


@pytest.mark.parametrize("action", ["check", "prepare", "pilot-prepare"])
def test_original_input_cli_no_local_prepare_shadowing(tmp_path, monkeypatch, capsys, action):
    from apps.multimodal_case import main

    root = gamma(tmp_path / "data")
    if action == "pilot-prepare":
        manifest = tmp_path / "pilot.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "multimodal-pilot-selection/1.0.0",
                    "purpose": "input_preparation_only",
                    "real_api_authorization": False,
                    "cases": [{"dataset": "gamma", "root": str(root), "sample": "0001"}],
                }
            )
        )
        args = ["--manifest", str(manifest), "--output", str(tmp_path / "pilot")]
    else:
        args = ["--dataset", "gamma", "--root", str(root), "--sample", "0001"]
        if action == "prepare":
            args += ["--output", str(tmp_path / "prepared")]
    monkeypatch.setattr(sys, "argv", ["multimodal_case", action, *args])
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset"] == "gamma" and summary["llm_requests"] == 0
