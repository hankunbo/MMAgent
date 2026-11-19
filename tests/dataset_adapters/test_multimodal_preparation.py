import json
import zipfile

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.analysis import build, from_preparation
from packages.dataset_adapters.multimodal.formats import sheets
from packages.dataset_adapters.multimodal.preparation import persist, verify
from packages.dataset_adapters.reading import AdapterError
from tests.dataset_adapters.test_multimodal import archive, gamma, gdp, grape


def rewrite_workbook(path, transform):
    with zipfile.ZipFile(path) as z:
        values = {i.filename: z.read(i) for i in z.infolist()}
    archive(path, {k: transform(k, v) for k, v in values.items()})


def test_preparation_roundtrip_excludes_labels_and_nonfinite_not_imputed(tmp_path):
    sample = prepare("harvard_gdp", gdp(tmp_path / "source", nonfinite=True), "data_0001.npz")
    path = persist(sample, tmp_path / "out")
    assert verify(path) == sample.manifest
    plan = from_preparation(path)
    assert plan == build(sample.manifest, dict(sample.payload_bytes))
    td = next(c for c in plan.components if c.component_id == "rnflt:tds")
    assert td.total_values == 52 and td.finite_values == 51
    assert td.acquisition == 51 / 52 and td.input_status == "partial"
    assert all(c.disease_direction_score is None for c in plan.components)
    assert plan.inference_ready is False and plan.score is None
    assert b"99123" not in path.read_bytes()
    assert all(b"99123" not in p.read_bytes() for p in (path.parent / "payloads").iterdir())
    with pytest.raises(FileExistsError):
        persist(sample, path.parent)


@pytest.mark.parametrize("change", ["extra", "scope", "payload", "duplicate", "path"])
def test_export_rejects_tamper(tmp_path, change):
    sample = prepare("harvard_gdp", gdp(tmp_path / "source"), "data_0001.npz")
    path = persist(sample, tmp_path / "out")
    value = json.loads(path.read_bytes())
    if change == "extra":
        value["prediction"] = "not_allowed"
    elif change == "scope":
        value["inference_ready"] = True
    elif change == "payload":
        f = path.parent / value["payload_files"][0]["path"]
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(data)
    elif change == "duplicate":
        value["payload_files"].append(value["payload_files"][0])
    else:
        value["payload_files"][0]["path"] = "../arbitrary"
    path.write_text(json.dumps(value))
    with pytest.raises(AdapterError):
        verify(path)


def test_preparation_refuses_symlink_output_parent(tmp_path):
    sample = prepare("harvard_gdp", gdp(tmp_path / "source"), "data_0001.npz")
    (tmp_path / "destination").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "destination", target_is_directory=True)
    with pytest.raises(AdapterError, match="output_symlink"):
        persist(sample, tmp_path / "alias/out")
    assert not list((tmp_path / "destination").iterdir())


def test_series_coverage_does_not_claim_volume_QC_or_binary_classification(tmp_path):
    sample = prepare("gamma", gamma(tmp_path), "0001")
    value = build(sample.manifest, dict(sample.payload_bytes))
    oct = next(c for c in value.components if c.component_id == "oct")
    assert oct.acquisition == 3 / 256 and oct.clinical_reliability is None
    assert value.task == "glaucoma_grading"
    assert value.medical_requirement_snapshot is None
    assert value.inference_ready is False
    assert "ordered_grades" in value.task_meaning


def test_longitudinal_single_visit_remains_usable_without_fake_change(tmp_path):
    sample = prepare("grape", grape(tmp_path), "1:OD", cutoff_visit=0)
    value = build(sample.manifest, dict(sample.payload_bytes))
    assert set(value.visit_groups) == {"0"}
    assert "fewer_than_two_visible_visits_no_observed_change" in value.task_gaps
    assert next(c for c in value.components if c.component_id == "visit:0:iop").acquisition == 1
    assert next(c for c in value.components if c.component_id == "visit:0:cct").acquisition == 0
    assert value.task == "observed_change" and value.score is None


def test_current_owner_rebuild_after_export_header_change(tmp_path):
    sample = prepare("grape", grape(tmp_path), "1:OD", cutoff_visit=1)
    path = tmp_path / "VF_and_clinical_information.xlsx"
    rewrite_workbook(path, lambda k, v: v.replace(b">Mean<", b">Unknown<"))
    with pytest.raises(AdapterError, match="headers"):
        sample.recheck()


@pytest.mark.parametrize("duplicate", ["sheet", "relationship"])
def test_workbook_duplicate_identity_rejected(tmp_path, duplicate):
    grape(tmp_path)
    path = tmp_path / "VF_and_clinical_information.xlsx"

    def transform(k, v):
        if duplicate == "sheet" and k == "xl/workbook.xml":
            return v.replace(b'name="Follow-up"', b'name="Baseline"')
        if duplicate == "relationship" and k == "xl/_rels/workbook.xml.rels":
            return v.replace(b'Id="r1"', b'Id="r0"')
        return v

    rewrite_workbook(path, transform)
    with pytest.raises(AdapterError, match="duplicate"):
        list(sheets(path.read_bytes()))
