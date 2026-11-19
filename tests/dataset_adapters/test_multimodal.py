"""New component preparation is not a new inference/clinical authorization."""

import io
import json
import struct
import zipfile
from dataclasses import replace
from xml.sax.saxutils import escape

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.formats import npy, npz, sheets
from packages.dataset_adapters.multimodal.models import CaseComponents
from packages.dataset_adapters.reading import AdapterError
from tests.dataset_adapters.support import npy as old_npy


def archive(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in entries.items():
            z.writestr(name, raw)


def gdp(root, *, age=0.0, label=99123, nonfinite=False):
    values = [-1.0] * 52
    if nonfinite:
        values[10] = float("nan")
    archive(
        root / "RNFLT/data_0001.npz",
        {
            "age.npy": old_npy((), "<f8", struct.pack("<d", age)),
            "md.npy": old_npy((), "<f8", struct.pack("<d", -2.5)),
            "tds.npy": old_npy((52,), "<f8", struct.pack("<52d", *values)),
            "rnflt.npy": old_npy((225, 225), "<f8", struct.pack("<d", 80.0) * 50625),
            "glaucoma.npy": old_npy((), "<i8", struct.pack("<q", label)),
            "progression.npy": old_npy((0,), "<f8", b""),
        },
    )
    return root


def jpeg():
    return b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x02\x00\x03\x01\x01\x11\x00\xff\xd9"


def gamma(root, indices=(0, 2, 10), photo=True):
    base = root / "training/multi-modality_images/0001"
    (base / "0001").mkdir(parents=True)
    if photo:
        (base / "0001.jpg").write_bytes(jpeg())
    for i in indices:
        (base / "0001" / f"{i}_image.jpg").write_bytes(jpeg())
    return root


def workbook(path, tables):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    entries = {
        "xl/workbook.xml": (
            f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            + "".join(f'<sheet name="{name}" r:id="r{i}"/>' for i, name in enumerate(tables))
            + "</sheets></workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="r{i}" Target="worksheets/s{i}.xml"/>'
                for i in range(len(tables))
            )
            + "</Relationships>"
        ),
    }
    for i, rows in enumerate(tables.values()):
        content = ""
        for n, row in enumerate(rows, 1):
            content += (
                f'<row r="{n}">'
                + "".join(
                    f'<c r="{col}{n}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                    for col, value in row.items()
                )
                + "</row>"
            )
        entries[f"xl/worksheets/s{i}.xml"] = (
            f'<worksheet xmlns="{ns}"><sheetData>{content}</sheetData></worksheet>'
        )
    archive(path, entries)


def grape(root, *, with_images=False, vf_missing=False):
    from packages.dataset_adapters.multimodal.adapters import column

    baseline = {
        "A": "1",
        "B": "OD",
        "C": "50",
        "E": "0",
        "F": "/",
        "G": "SECRET_FUTURE_COUNT",
        "H": "SECRET_PROGRESSION",
        "K": "SECRET_DIAGNOSIS",
        "L": "80",
        "Q": "baseline.jpg" if with_images else "/",
        **{column(20 + i): "" if vf_missing else str(i) for i in range(61)},
    }
    headers_b = {
        "A": "Subject Number",
        "B": "Laterality",
        "C": "Age",
        "E": "IOP",
        "F": "CCT",
        "L": "OCT RNFL thickness",
        "Q": "Corresponding CFP",
        "T": "VF",
    }
    headers_f = {
        "A": "Subject Number",
        "B": "Laterality",
        "C": "Visit Number",
        "D": "Interval Years",
        "E": "IOP",
        "F": "Corresponding CFP",
        "I": "VF",
    }
    workbook(
        root / "VF_and_clinical_information.xlsx",
        {
            "Baseline": [
                headers_b,
                {
                    **{column(20 + i): str(i) for i in range(61)},
                    "L": "Mean",
                    "M": "S",
                    "N": "N",
                    "O": "I",
                    "P": "T",
                },
                baseline,
                dict(baseline, B="OS"),
            ],
            "Follow-up": [
                headers_f,
                {column(9 + i): str(i) for i in range(61)},
                {
                    "A": "1",
                    "B": "OD",
                    "C": "1",
                    "D": "0.5",
                    "E": "18",
                    "F": "followup.jpg" if with_images else "/",
                    "I": "" if vf_missing else "-3",
                },
                {"A": "1", "B": "OD", "C": "2", "D": "1", "E": "98765", "F": "SECRET_FUTURE_IMAGE"},
            ],
        },
    )
    if with_images:
        images = root / "extracted/CFPs"
        images.mkdir(parents=True)
        for name in ("baseline.jpg", "followup.jpg"):
            (images / name).write_bytes(jpeg())
    return root


def test_gdp_preserves_zero_negative_missing_without_label_values(tmp_path):
    value = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    c = {c.concept: c for c in value.manifest.components}
    assert c["age"].value == 0 and c["md"].value == -2.5
    assert c["bscans"].status == "missing"
    assert c["rnflt"].kind == "map" and c["rnflt"].unit is None
    assert "99123" not in value.manifest.model_dump_json()
    assert all(b"SECRET" not in data for _, data in value.payload_bytes)
    assert value.manifest.runtime_ready is False
    assert value.recheck() == value
    assert CaseComponents.model_validate_json(value.manifest.model_dump_json()) == value.manifest


def test_nonfinite_vector_is_partial_not_false_zero(tmp_path):
    value = prepare("harvard_gdp", gdp(tmp_path, nonfinite=True), "data_0001.npz")
    v = next(c for c in value.manifest.components if c.concept == "tds")
    assert v.status == "partial" and v.reason == "nonfinite_values_preserved"


def test_prepare_recheck_rejects_changed_source_and_forged_payload(tmp_path):
    value = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    with pytest.raises(AdapterError, match="preparation_changed"):
        replace(value, payload_bytes=()).recheck()
    gdp(tmp_path, age=99)
    with pytest.raises(AdapterError, match="preparation_changed"):
        value.recheck()


@pytest.mark.parametrize(
    "dataset,sample,split",
    [
        ("unregistered", "x", None),
        ("gamma", "0001", "test"),
        ("harvard_gdp", "../data_0001.npz", None),
    ],
)
def test_unknown_split_and_path_not_admitted(tmp_path, dataset, sample, split):
    with pytest.raises(AdapterError):
        prepare(dataset, tmp_path, sample, selection=split)


def test_gamma_numeric_order_partial_and_no_slo_claim(tmp_path):
    value = prepare("gamma", gamma(tmp_path), "0001")
    c = next(c for c in value.manifest.components if c.id == "oct")
    assert c.member_indices == (0, 2, 10) and c.status == "partial"
    assert c.expected_members == 256
    assert value.manifest.task == "glaucoma_grading"
    assert {c.modality for c in value.manifest.components} == {"OCT", "color_fundus"}
    assert len(value.manifest.components) == 2  # Not 256 artificial independent observations.


def test_gamma_full_series_obtained_and_missing_photo_retained(tmp_path):
    value = prepare("gamma", gamma(tmp_path, range(256), photo=False), "0001")
    assert value.manifest.components[0].status == "missing"
    assert value.manifest.components[1].status == "obtained"


def test_gamma_duplicate_index_rejected(tmp_path):
    gamma(tmp_path)
    (tmp_path / "training/multi-modality_images/0001/0001/00_image.jpg").write_bytes(jpeg())
    with pytest.raises(AdapterError, match="slice_index"):
        prepare("gamma", tmp_path, "0001")


def test_source_symlink_rejected(tmp_path):
    gamma(tmp_path)
    photo = tmp_path / "training/multi-modality_images/0001/0001.jpg"
    raw = photo.read_bytes()
    photo.unlink()
    other = tmp_path / "other.jpg"
    other.write_bytes(raw)
    photo.symlink_to(other)
    with pytest.raises(ValueError):
        prepare("gamma", tmp_path, "0001")


def test_grape_cutoff_eye_group_zero_and_no_future_values(tmp_path):
    root = grape(tmp_path)
    right = prepare("grape", root, "1:OD", cutoff_visit=1)
    left = prepare("grape", root, "1:OS", cutoff_visit=0)
    assert right.manifest.patient_group == left.manifest.patient_group
    assert {c.visit for c in right.manifest.components} == {"0", "1"}
    assert {c.laterality for c in right.manifest.components} == {"RIGHT"}
    wire = right.manifest.model_dump_json() + str(right.payload_bytes)
    assert "SECRET" not in wire and "98765" not in wire
    assert next(c for c in right.manifest.components if c.id == "visit:0:iop").value == 0
    assert next(c for c in right.manifest.components if c.id == "visit:0:cct").value is None
    assert all(c.status == "missing" for c in right.manifest.components if c.kind == "image")
    assert right.recheck() == right


def test_grape_cutoff_required_and_no_prediction_claim(tmp_path):
    with pytest.raises(AdapterError, match="cutoff"):
        prepare("grape", grape(tmp_path), "1:OD")
    value = prepare("grape", tmp_path, "1:OD", cutoff_visit=0)
    assert value.manifest.task == "observed_change"


@pytest.mark.parametrize("kind", ["object", "path", "duplicate", "length", "fortran"])
def test_unsafe_npz_rejected(kind):
    raw = old_npy((), "<f8", struct.pack("<d", 0.0))
    name = "age.npy"
    if kind == "object":
        raw = old_npy((), "|O8", b"unsafe!!")
    if kind == "path":
        name = "../age.npy"
    if kind == "length":
        raw += b"extra"
    if kind == "fortran":
        raw = raw.replace(b"False", b"True ")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(name, raw)
        if kind == "duplicate":
            with pytest.warns(UserWarning):
                z.writestr(name, raw)
    with pytest.raises(AdapterError):
        npz(out.getvalue(), {"age"})


def test_unknown_safe_field_values_not_exposed():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("extra.npy", old_npy((), "<i8", struct.pack("<q", 99123)))
    arrays, names = npz(out.getvalue(), {"age"})
    assert arrays == {} and names == ("extra",)


def test_array_endian_float_and_empty_reference():
    assert npy(old_npy((), ">f8", struct.pack(">d", -2))).scalar() == -2
    assert npy(old_npy((0,), "<f8", b"")).data == b""


def test_xlsx_actual_cell_coordinates(tmp_path):
    p = tmp_path / "table.xlsx"
    workbook(p, {"Sheet": [{"A": "x", "AA": 0}]})
    assert list(sheets(p.read_bytes())) == [("Sheet", [(1, {"A": "x", "AA": "0"})])]


def test_model_rejects_false_runtime_ready(tmp_path):
    value = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz").manifest.model_dump(mode="json")
    value["runtime_ready"] = True
    with pytest.raises(ValueError):
        CaseComponents.model_validate_json(json.dumps(value))
