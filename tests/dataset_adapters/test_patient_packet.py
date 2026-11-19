"""Patient-wide archive boundaries, using real file owners and no runtime campaign."""

import io
import json

import pytest
from PIL import Image

from apps.dataset_adapter import main
from packages.dataset_adapters.patient_decoders import candidates
from packages.dataset_adapters.patient_packet import prepare_patient
from packages.dataset_adapters.patient_storage import persist, verify
from packages.dataset_adapters.reading import AdapterError
from tests.dataset_adapters.support import fair, odir
from tests.dataset_adapters.test_multimodal import gamma, gdp, grape
from tests.dataset_adapters.test_papila import dataset as papila


def test_grape_whole_subject_keeps_future_and_both_eyes_without_task_scope(tmp_path):
    grape(tmp_path)
    p = prepare_patient("grape", tmp_path, "1")
    assert p.packet.collection_scope == "all_recorded_eyes_and_visits"
    assert {c.laterality for c in p.packet.components} == {"LEFT", "RIGHT"}
    assert {c.visit for c in p.packet.components} == {"0", "1", "2"}
    assert any(c.value == 98765 for c in p.packet.components)
    assert len(p.packet.visits) == 4  # 3 OD groups and 1 OS group, not 3 asserted shared visits
    assert not p.packet.task_scope_selected and not p.packet.runtime_ready
    assert "SECRET_PROGRESSION" not in p.packet.model_dump_json()
    assert "SECRET_FUTURE_IMAGE" not in p.packet.model_dump_json()
    assert "dataset" not in p.packet.model_dump() and "task" not in p.packet.model_dump()
    assert all(not c.source_paths for c in p.packet.components)


@pytest.mark.parametrize(
    "adapter,subject,fixture",
    [
        ("gamma", "0001", gamma),
        ("harvard_gdp", "data_0001.npz", gdp),
        ("fairvlmed", "data_00000.npz", fair),
        ("odir", "11", odir),
        ("papila", "003", papila),
        ("grape", "1", grape),
    ],
)
def test_all_owners_persist_readback_and_do_not_overwrite(tmp_path, adapter, subject, fixture):
    root = tmp_path / "input"
    root.mkdir()
    fixture(root)
    prepared = prepare_patient(adapter, root, subject)
    out = tmp_path / "packet"
    persist(prepared, out, source_root=root)
    assert verify(out, source_root=root) == prepared.packet
    with pytest.raises(FileExistsError):
        persist(prepared, out, source_root=root)
    assert prepared.validation["api_requests"] == prepared.validation["model_runs"] == 0
    for word in ("SECRET_DIAGNOSIS", "SECRET_LEFT", "SECRET_SUMMARY", "99123"):
        assert word not in (out / "case-input.json").read_text()
    if adapter in {"gamma", "harvard_gdp", "fairvlmed"}:
        assert prepared.packet.identity_basis == "sample_only_unproven_patient"
        assert all(c.laterality == "UNKNOWN" for c in prepared.packet.components)


def test_pixel_decode_failure_is_reported_but_preserved(tmp_path):
    gamma(tmp_path)
    p = prepare_patient("gamma", tmp_path, "0001")
    assert p.validation["image_decode_errors"]
    assert all(c.parse_status == "invalid_payload" for c in p.packet.components)
    assert p.packet.payloads  # Original bytes still available for diagnosis, not medical use.


def test_real_encoded_pixels_decode(tmp_path):
    gamma(tmp_path)
    raw = io.BytesIO()
    Image.new("RGB", (3, 2)).save(raw, format="JPEG")
    for p in tmp_path.rglob("*.jpg"):
        p.write_bytes(raw.getvalue())
    prepared = prepare_patient("gamma", tmp_path, "0001")
    assert not prepared.validation["image_decode_errors"]
    assert all(c.parse_status == "readable" for c in prepared.packet.components)
    assert (
        next(c for c in prepared.packet.components if c.kind == "image_series").status == "partial"
    )


def test_partial_and_zero_are_not_rewritten(tmp_path):
    gdp(tmp_path, nonfinite=True)
    p = prepare_patient("harvard_gdp", tmp_path, "data_0001.npz")
    c = {c.concept: c for c in p.packet.components}
    assert c["age"].value == 0 and c["tds"].status == "partial"
    assert c["bscans"].status == "missing" and c["bscans"].parse_status == "no_payload"


@pytest.mark.parametrize("target", ["payload", "manifest", "incomplete", "source"])
def test_changed_or_incomplete_packet_cannot_verify(tmp_path, target):
    root = tmp_path / "source"
    gdp(root)
    p = prepare_patient("harvard_gdp", root, "data_0001.npz")
    out = tmp_path / "packet"
    persist(p, out, source_root=root)
    if target == "payload":
        (out / p.packet.payloads[0].path).write_bytes(b"wrong")
    elif target == "manifest":
        value = json.loads((out / "case-input.json").read_text())
        value["runtime_ready"] = True
        (out / "case-input.json").write_text(json.dumps(value))
    elif target == "source":
        gdp(root, age=99)
    else:
        (out / "complete.json").unlink()
    with pytest.raises((AdapterError, FileNotFoundError)):
        verify(out, source_root=root)


def test_source_change_before_save_rejected(tmp_path):
    gdp(tmp_path / "source")
    p = prepare_patient("harvard_gdp", tmp_path / "source", "data_0001.npz")
    gdp(tmp_path / "source", age=99)
    with pytest.raises(AdapterError, match="source_changed_before_save"):
        persist(p, tmp_path / "packet", source_root=tmp_path / "source")
    assert not (tmp_path / "packet").exists()


def test_cli_and_restricted_selector(tmp_path, capsys):
    grape(tmp_path / "source")
    args = ["--adapter", "grape", "--root", str(tmp_path / "source")]
    assert main(["list", *args]) == 0
    assert json.loads(capsys.readouterr().out)["subjects"] == ["1"]
    assert main(["prepare", *args, "--subject", "1:OS", "--output", str(tmp_path / "bad")]) == 2
    assert main(["prepare", *args, "--subject", "1", "--output", str(tmp_path / "good")]) == 0
    assert main(["verify", "--input", str(tmp_path / "good")]) == 0
    with pytest.raises(AdapterError, match="selection_not_supported"):
        candidates("gamma", tmp_path, selection="test")


def test_output_symlink_rejected(tmp_path):
    gdp(tmp_path / "source")
    (tmp_path / "target").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)
    p = prepare_patient("harvard_gdp", tmp_path / "source", "data_0001.npz")
    with pytest.raises(AdapterError, match="output_symlink"):
        persist(p, tmp_path / "link" / "out", source_root=tmp_path / "source")


def test_grape_source_background_follows_its_eye_and_visit(tmp_path):
    from packages.dataset_adapters.multimodal.formats import sheets
    from tests.dataset_adapters.test_multimodal import workbook

    grape(tmp_path)
    path = tmp_path / "VF_and_clinical_information.xlsx"
    tables = {name: [dict(c) for _, c in rows] for name, rows in sheets(path.read_bytes())}
    tables["Baseline"][0].update(D="Gender", R="Acquisition Device", S="Resolusion")
    tables["Baseline"][2].update(D="male", R="camera-A", S="1000x1000")
    tables["Follow-up"][0].update(G="Acquisition Device", H="Resolusion")
    tables["Follow-up"][2].update(G="camera-B", H="2000x2000")
    workbook(path, tables)
    p = prepare_patient("grape", tmp_path, "1").packet
    b = next(b for b in p.background if b.value == "camera-B")
    assert b.laterality == "RIGHT" and b.visit == "1"
    group = next(v for v in p.visits if b.id in v.component_ids)
    assert group.interval_years == 0.5
    assert len(p.visits) == 4


def test_wrong_eye_or_visit_membership_cannot_be_sealed(tmp_path):
    from packages.disease_packs.models import seal

    grape(tmp_path)
    p = prepare_patient("grape", tmp_path, "1").packet
    data = p.model_dump(mode="json", exclude={"checksum"})
    data["visits"][0]["laterality"] = "UNKNOWN"
    with pytest.raises(ValueError, match="visit_eye_binding"):
        seal(type(p), **data)
