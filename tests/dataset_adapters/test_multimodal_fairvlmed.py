"""FairVLMed original selected-input owner -> shared capabilities, no model calls."""

import json

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.analysis import build
from packages.dataset_adapters.multimodal.capabilities import project
from packages.dataset_adapters.multimodal.models import CaseComponents
from packages.dataset_adapters.reading import AdapterError
from packages.dataset_adapters.runtime_input import factory
from tests.dataset_adapters.support import FAIR_FIELDS, csv_write, fair, npz


def test_original_selected_reader_projected_without_hidden_values(tmp_path):
    fair(tmp_path)
    original = factory("fairvlmed", tmp_path).prepare("data_00000.npz")
    result = prepare("fairvlmed", tmp_path, "data_00000.npz", selection="training")
    assert result.recheck() == result
    assert result.manifest.adapter_version == "multimodal-adapters/1.3.0"
    assert CaseComponents.model_validate_json(result.manifest.model_dump_json()) == result.manifest
    image, age = result.manifest.components
    assert image.modality == "SLO" and image.kind == "image" and image.shape == (2, 3)
    assert dict(result.payload_bytes)[image.payload_ids[0]] == original.images[0][0]
    assert age.value == 42 and age.unit == "year"
    assert age.laterality == "UNKNOWN" and result.manifest.patient_group is None
    assert all(p.media_type == "image/png" for p in result.manifest.payloads)
    audit = build(result.manifest, dict(result.payload_bytes))
    capabilities = project(result.manifest, audit)
    assert capabilities.components[0].modality == "SLO"
    assert capabilities.components[0].physical_image_calibration == "not_established"
    assert "jpeg_header_only_not_pixel_decode_or_medical_QC" not in audit.components[0].issues
    value = result.manifest.model_dump_json()
    assert "SECRET_DIAGNOSIS" not in value and "SECRET_SUMMARY" not in value
    assert "glaucoma" in result.manifest.excluded_fields
    assert all(not raw.startswith(b"PK") for _, raw in result.payload_bytes)


@pytest.mark.parametrize("split", ["test", "validation"])
def test_split_gate_uses_original_csv_and_does_not_allow_override(tmp_path, split):
    _, rows = fair(tmp_path)
    csv_write(tmp_path / "ReadMe/data_summary.csv", FAIR_FIELDS, [dict(rows[0], use=split)])
    with pytest.raises(AdapterError, match="training_sample_not_found"):
        prepare("fairvlmed", tmp_path, "data_00000.npz", selection="training")
    with pytest.raises(AdapterError, match="split_not_allowed"):
        prepare("fairvlmed", tmp_path, "data_00000.npz", selection=split)


def test_original_selected_optional_fields_and_source_recheck(tmp_path):
    import io
    import zipfile

    fair(tmp_path)
    raw = io.BytesIO(npz())
    out = io.BytesIO()
    with zipfile.ZipFile(raw) as old, zipfile.ZipFile(out, "w") as new:
        new.writestr("slo_fundus.npy", old.read("slo_fundus.npy"))
    path = tmp_path / "Dataset/Training/data_00000.npz"
    path.write_bytes(out.getvalue())
    sample = prepare("fairvlmed", tmp_path, "data_00000.npz")
    assert sample.manifest.components[1].status == "missing"
    assert sample.manifest.components[1].value is None
    path.write_bytes(npz())
    with pytest.raises(AdapterError, match="preparation_changed"):
        sample.recheck()


def test_bridge_version_cannot_be_silently_old_or_other_dataset(tmp_path):
    fair(tmp_path)
    value = prepare("fairvlmed", tmp_path, "data_00000.npz").manifest.model_dump(mode="json")
    from packages.disease_packs.models import seal

    value.pop("checksum")
    with pytest.raises(ValueError, match="fairvlmed_adapter_binding"):
        seal(CaseComponents, **dict(value, adapter_version="multimodal-adapters/1.0.0"))
    with pytest.raises(ValueError, match="fairvlmed_adapter_binding"):
        seal(CaseComponents, **dict(value, dataset="gamma", task="glaucoma_grading"))


def test_actual_cli_prepare_preflight_and_scope_preserve_slo(tmp_path, monkeypatch):
    import sys

    from apps.multimodal_case import main
    from packages.multimodal_research import entry

    def invoke(arguments):
        monkeypatch.setattr(sys, "argv", ["multimodal_case", *arguments])
        return main()

    fair(tmp_path / "data")
    out = tmp_path / "run"
    monkeypatch.setattr(entry, "owners", lambda: pytest.fail("local preflight opened DB"))
    assert (
        invoke(
            [
                "live-prepare",
                "--dataset",
                "fairvlmed",
                "--root",
                str(tmp_path / "data"),
                "--sample",
                "data_00000.npz",
                "--selection",
                "training",
                "--task",
                "advanced_glaucoma",
                "--registry",
                "configs/research/multimodal_propositions_v2.json",
                "--automatic-requirements",
                "--reuse-procedures",
                "--resilient-vision",
                "--view-profile",
                "detail-v1",
                "--execution-profile",
                "relay-root-question-v1",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    result = entry.assemble_entry(out, preflight_only=True)
    assert result["api_requests"] == 0
    from packages.multimodal_research import entry_preflight, execution_lease

    monkeypatch.setattr(
        execution_lease, "entry_session", lambda _: pytest.fail("local preflight acquired lease")
    )
    assert entry_preflight.command(out, preflight_only=True) == result
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["procedure_reuse_contract"]["policy"] == "proposition-input-method-reuse/1.0.0"
    assert "SLO" in spec["intent"]["allowed_modalities"]
    view = json.loads((out / "views/views.json").read_bytes())
    assert len(view["views"]) == 1 and view["views"][0]["modality"] == "SLO"
    assert view["views"][0]["transform"]["source_shapes"] == [[2, 3]]
    assert view["views"][0]["transform"]["anatomical_registration"] is False
    assert view["views"][0]["inspected_fraction"] == 1


def test_base_observation_uses_actual_slo_without_color_method_or_extra_proposition(tmp_path):
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.material import requirements
    from tests.evaluation_preparation.test_active_acquisition import source_fixture

    fair(tmp_path)
    sample = prepare("fairvlmed", tmp_path, "data_00000.npz")
    task, _ = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v2.json")
    rows = requirements(sample.manifest, source_fixture(), task)
    assert {r["component"] for r in rows if r["component"] is not None} == {"slo_fundus", "age"}
    assert {r["id"] for r in rows if r["component"] is None} == {"gap:functional_stage"}
    image = next(r for r in rows if r["component"] == "slo_fundus")
    assert image["input_modalities"] == ("SLO",)
    assert image["missing_reason"] == "not_yet_observed"
    assert "灰度SLO" in image["definition"].interpretation
    registry = json.loads(open("configs/research/multimodal_propositions_v2.json").read())
    assert all(r["id"] != "non_glaucoma" for r in registry["profiles"])


def test_proposition_versions_preserve_old_binding_and_reject_forged_modalities():
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.single_proposition import selected

    old, _ = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")
    new, _ = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v2.json")
    assert selected(old) == selected(new) == ("advanced_glaucoma", "晚期青光眼")
    assert "SLO" not in old.allowed_modalities and "SLO" in new.allowed_modalities
    for forged in [
        old.model_copy(update={"allowed_modalities": new.allowed_modalities}),
        new.model_copy(update={"version": "999.0.0"}),
    ]:
        with pytest.raises(ValueError, match="single_proposition_task_binding"):
            selected(forged)
