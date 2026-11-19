import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from packages.dataset_adapters.papila import HEADERS, PapilaClinicalPreparation, number, workbook
from packages.dataset_adapters.papila_models import PapilaClinicalInput
from packages.dataset_adapters.reading import AdapterError


def xlsx(*, ids=("#002", "#003", "#004"), replace=None, formula=False, extra=False):
    rows = ['<row r="1"/>']

    def cells(values, n):
        return "".join(
            f'<c r="{col}{n}" t="inlineStr"><is><t>{value}</t></is></c>'
            for col, value in values.items()
        )

    rows.append('<row r="2">' + cells(HEADERS, 2) + "</row>")
    rows.append('<row r="3">' + cells({"A": "ID"}, 3) + "</row>")
    for index, ident in enumerate(ids, 4):
        values = dict(
            A=ident,
            B="45",
            C="0",
            D="1",
            E="0",
            F="-1.5",
            G="80",
            H="0",
            I="18",
            J="",
            K="540",
            L="23.4",
            M="-3.2",
        )
        values.update(replace or {})
        body = cells(values, index)
        if formula:
            body = body.replace("<is><t>45</t></is>", "<f>1+44</f><v>45</v>", 1)
        rows.append(f'<row r="{index}">{body}</row>')
    raw = io.BytesIO()
    with ZipFile(raw, "w", ZIP_DEFLATED) as z:
        z.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            + "".join(rows)
            + "</sheetData></worksheet>",
        )
        if extra:
            z.writestr("xl/externalLinks/externalLink1.xml", "x")
    return raw.getvalue()


def dataset(tmp_path):
    (tmp_path / "README.md").write_text("# PAPILAv1.1\n")
    (tmp_path / "ClinicalData").mkdir()
    (tmp_path / "FundusImages").mkdir()
    jpg = b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03" + b"\x00" * 10 + b"\xff\xd9"
    for eye in ("OD", "OS"):
        (tmp_path / f"ClinicalData/patient_data_{eye.lower()}.xlsx").write_bytes(xlsx())
        for ident in ("002", "003", "004"):
            (tmp_path / f"FundusImages/RET{ident}{eye}.jpg").write_bytes(jpg)
    return PapilaClinicalPreparation(tmp_path)


def test_projection_profiles_missing_and_negative(tmp_path):
    owner = dataset(tmp_path)
    a = owner.prepare("RET003OD", profile="image_demographics")
    b = owner.prepare("RET003OD", profile="image_demographics_clinical")
    assert not a.clinical and len(b.clinical) == 9
    assert a.image == b.image and a.patient_group == "PAPILA:RET003"
    assert b.sex == "male" and b.clinical["dioptre_1"].value == 0
    assert b.clinical["vf_md"].value == -3.2
    assert b.clinical["iop_perkins"].value is None
    assert b.clinical["iop_perkins"].missing_reason == "not_recorded"
    assert b.clinical["iop_pneumatic"].source_cell == "I5"
    assert b.clinical["astigmatism"].unit is None
    assert not b.runtime_admission and not b.final_test_unlocked
    assert "diagnosis" not in b.model_dump_json().lower()
    assert "contour" not in b.model_dump_json().lower()
    assert owner.restricted_diagnosis("RET003OD")["diagnosis_code"] == 1


def test_inventory_group_and_counts(tmp_path):
    value = dataset(tmp_path).inventory()
    assert value["patient_count"] == 3 and value["image_count"] == 6
    assert value["acquired_counts"]["OS"]["iop_perkins"] == 0
    assert "diagnosis" not in json.dumps(value)


@pytest.mark.parametrize(
    "kwargs",
    [dict(ids=("#002", "#002")), dict(formula=True), dict(extra=True), dict(ids=("../../x",))],
)
def test_workbook_invalid(kwargs):
    with pytest.raises(AdapterError, match="papila_workbook"):
        workbook(xlsx(**kwargs))


@pytest.mark.parametrize("value", ["NaN", "Inf", "SECRET", "-Inf"])
def test_nonfinite_and_free_text_rejected(value):
    with pytest.raises(AdapterError, match="papila_nonfinite"):
        number(value)


@pytest.mark.parametrize(
    "change",
    [
        dict(laterality="LEFT"),
        dict(patient_group="PAPILA:RET999"),
        dict(runtime_admission=True),
        dict(final_test_unlocked=True),
        dict(clinical={}),
        dict(diagnosis=1),
    ],
)
def test_sealed_semantics(tmp_path, change):
    value = dataset(tmp_path).prepare("RET003OD", profile="image_demographics_clinical")
    with pytest.raises(ValueError):
        PapilaClinicalInput.model_validate_json(json.dumps(value.model_dump(mode="json") | change))


def test_cross_eye_demographics_conflict(tmp_path):
    owner = dataset(tmp_path)
    (tmp_path / "ClinicalData/patient_data_os.xlsx").write_bytes(xlsx(replace={"C": "1"}))
    with pytest.raises(AdapterError, match="bilateral_demographic_conflict"):
        owner.inventory()


def test_missing_image_and_changed_source(tmp_path):
    owner = dataset(tmp_path)
    old = owner.prepare("RET003OD", profile="image_demographics")
    (tmp_path / "ClinicalData/patient_data_od.xlsx").write_bytes(xlsx(replace={"B": "46"}))
    current = owner.prepare("RET003OD", profile="image_demographics")
    assert old.clinical_source.checksum != current.clinical_source.checksum
    (tmp_path / "FundusImages/RET004OS.jpg").unlink()
    with pytest.raises(AdapterError, match="images_clinical_mismatch"):
        owner.inventory()


def test_symlink_rejected(tmp_path):
    owner = dataset(tmp_path)
    p = tmp_path / "FundusImages/RET003OD.jpg"
    p.unlink()
    p.symlink_to(tmp_path / "FundusImages/RET002OD.jpg")
    with pytest.raises(AdapterError, match="source_missing"):
        owner.prepare("RET003OD", profile="image_demographics")


def test_unknown_profile_rejected(tmp_path):
    with pytest.raises(ValueError):
        dataset(tmp_path).prepare("RET003OD", profile="anything")


def test_owner_recheck_rejects_forgery_and_source_drift(tmp_path):
    owner = dataset(tmp_path)
    original = owner.prepare("RET003OD", profile="image_demographics_clinical")
    assert owner.recheck(original) == original
    forged = original.model_copy(update={"age_years": 70})
    with pytest.raises(AdapterError, match="source_or_projection_changed"):
        owner.recheck(forged)
    (tmp_path / "ClinicalData/patient_data_od.xlsx").write_bytes(xlsx(replace={"B": "46"}))
    with pytest.raises(AdapterError, match="source_or_projection_changed"):
        owner.recheck(original)


def test_diagnosis_change_never_changes_visible_measurements(tmp_path):
    owner = dataset(tmp_path)
    before = owner.prepare("RET003OD", profile="image_demographics_clinical")
    (tmp_path / "ClinicalData/patient_data_od.xlsx").write_bytes(xlsx(replace={"D": "2"}))
    after = owner.prepare("RET003OD", profile="image_demographics_clinical")
    assert before.age_years == after.age_years and before.sex == after.sex
    assert {k: v.value for k, v in before.clinical.items()} == {
        k: v.value for k, v in after.clinical.items()
    }
    assert before.clinical_source.checksum != after.clinical_source.checksum
    assert owner.restricted_diagnosis("RET003OD")["diagnosis_code"] == 2
