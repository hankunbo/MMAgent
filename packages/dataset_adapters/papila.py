"""Bounded PAPILA clinical preparation in the dataset-adapter owner.

No diagnosis, contours or free text enters the clinical projection. This port does
not register a CaseBundle or change the old 06E benchmark dataset declaration.
"""

import io
import math
import re
from pathlib import Path
from zipfile import ZipFile

from defusedxml import ElementTree as XML

from packages.tool_benchmark.formats import jpeg_dimensions

from .papila_models import ClinicalValue, PapilaClinicalInput
from .reading import AdapterError, read_source

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
HEADERS = dict(
    zip(
        "BCDEFGHIJKLM",
        (
            "Age",
            "Gender",
            "Diagnosis",
            "dioptre_1",
            "dioptre_2",
            "astigmatism",
            "Phakic/Pseudophakic",
            "Pneumatic",
            "Perkins",
            "Pachymetry",
            "Axial_Length",
            "VF_MD",
        ),
        strict=True,
    )
)
# Raw field names are retained where the workbook does not disambiguate semantics.
FIELDS = {
    "E": ("dioptre_1", "D", "convention_pending_confirmation", None),
    "F": ("dioptre_2", "D", "convention_pending_confirmation", None),
    "G": ("astigmatism", None, "unresolved", None),
    "H": ("lens_removed", "0=natural_lens;1=removed", "coding", None),
    "I": ("iop_pneumatic", "mmHg", "paper", "Pneumatic"),
    "J": ("iop_perkins", "mmHg", "paper", "Perkins"),
    "K": ("pachymetry", "um", "paper", None),
    "L": ("axial_length", "mm", "convention_pending_confirmation", None),
    "M": ("vf_md", "dB", "convention_pending_confirmation", None),
}


def workbook(raw):
    """Known single-sheet layout, finite scalar values, formulas/external links rejected."""
    try:
        if not 0 < len(raw) <= 4194304:
            raise ValueError
        with ZipFile(io.BytesIO(raw)) as z:
            entries = z.infolist()
            names = [e.filename for e in entries]
            if len(names) != len(set(names)) or len(names) > 100:
                raise ValueError
            if sum(e.file_size for e in entries) > 8388608:
                raise ValueError
            if any(e.flag_bits & 1 or e.compress_type not in (0, 8) for e in entries):
                raise ValueError
            if any("externalLinks" in n or "vbaProject" in n for n in names):
                raise ValueError
            sheets = [n for n in names if re.fullmatch(r"xl/worksheets/sheet[0-9]+.xml", n)]
            if sheets != ["xl/worksheets/sheet1.xml"]:
                raise ValueError
            strings = []
            if "xl/sharedStrings.xml" in names:
                strings = [
                    "".join(v.itertext()) for v in XML.fromstring(z.read("xl/sharedStrings.xml"))
                ]
            document = XML.fromstring(z.read(sheets[0]))
        rows = {}
        for row in document.findall(f"{NS}sheetData/{NS}row"):
            index = int(row.attrib["r"])
            if index in rows or not 1 <= index <= 1003:
                raise ValueError
            cells = {}
            for cell in row:
                address = cell.attrib["r"]
                match = re.fullmatch(r"([A-M])([1-9][0-9]*)", address)
                if not match or int(match[2]) != index or match[1] in cells:
                    raise ValueError
                if cell.find(NS + "f") is not None:
                    raise ValueError
                value = cell.find(NS + "v")
                kind = cell.attrib.get("t", "n")
                if kind == "inlineStr":
                    value = "".join(cell.find(NS + "is").itertext())
                elif value is None:
                    value = ""
                elif kind == "s":
                    number = int(value.text)
                    if not 0 <= number < len(strings):
                        raise ValueError
                    value = strings[number]
                elif kind == "n":
                    value = value.text or ""
                else:
                    raise ValueError
                cells[match[1]] = value.strip()
            rows[index] = cells
        if {k: v for k, v in rows[2].items() if v} != HEADERS or rows[3].get("A") != "ID":
            raise ValueError
        result = {}
        for index, cells in rows.items():
            if index <= 3 or not any(cells.values()):
                continue
            ident = cells.get("A", "")
            if not re.fullmatch(r"#[0-9]{3}", ident) or ident in result:
                raise ValueError
            result[ident] = (index, cells)
        if not result:
            raise ValueError
        return result
    except Exception:
        raise AdapterError("papila_workbook_layout_or_content_invalid") from None


def number(value):
    if value == "":
        return None
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError
        return result
    except (TypeError, ValueError):
        raise AdapterError("papila_nonfinite_or_nonnumeric_field") from None


class PapilaClinicalPreparation:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_absolute() or self.root.resolve() != self.root:
            raise AdapterError("papila_root_invalid")
        version, _ = read_source(self.root, "README.md")
        if version.decode().splitlines()[0] != "# PAPILAv1.1":
            raise AdapterError("papila_version_unrecognized")

    def table(self, eye):
        if eye not in ("OD", "OS"):
            raise AdapterError("papila_eye_invalid")
        raw, binding = read_source(self.root, f"ClinicalData/patient_data_{eye.lower()}.xlsx")
        return workbook(raw), binding

    def inventory(self):
        """No diagnosis values returned; counts are acquisition, not interpretability."""
        tables = {eye: self.table(eye) for eye in ("OD", "OS")}
        if set(tables["OD"][0]) != set(tables["OS"][0]):
            raise AdapterError("papila_patient_sets_differ")
        expected = {f"RET{ident[1:]}{eye}" for ident in tables["OD"][0] for eye in tables}
        actual = {p.stem for p in (self.root / "FundusImages").glob("*.jpg")}
        if actual != expected:
            raise AdapterError("papila_images_clinical_mismatch")
        for ident in tables["OD"][0]:
            for col in ("B", "C"):
                if number(tables["OD"][0][ident][1].get(col, "")) != number(
                    tables["OS"][0][ident][1].get(col, "")
                ):
                    raise AdapterError("papila_bilateral_demographic_conflict")
        counts = {}
        for eye, (rows, _) in tables.items():
            counts[eye] = {
                name: sum(number(row.get(col, "")) is not None for _, row in rows.values())
                for col, (name, _, _, _) in FIELDS.items()
            }
        return dict(
            patient_count=len(tables["OD"][0]),
            image_count=len(actual),
            acquired_counts=counts,
            source_bindings={e: b.model_dump(mode="json") for e, (_, b) in tables.items()},
            sample_ids=sorted(actual),
            label_values_exported=False,
        )

    def prepare(self, sample_id, *, profile):
        if not re.fullmatch(r"RET[0-9]{3}(OD|OS)", sample_id):
            raise AdapterError("papila_sample_invalid")
        rows, binding = self.table(sample_id[-2:])
        ident = "#" + sample_id[3:6]
        if ident not in rows:
            raise AdapterError("papila_sample_missing")
        index, row = rows[ident]
        age, sex = number(row.get("B", "")), number(row.get("C", ""))
        if sex not in (None, 0, 1):
            raise AdapterError("papila_sex_code_invalid")
        clinical = {}
        if profile == "image_demographics_clinical":
            for col, (name, unit, basis, method) in FIELDS.items():
                value = number(row.get(col, ""))
                if col == "H" and value not in (None, 0, 1):
                    raise AdapterError("papila_lens_code_invalid")
                clinical[name] = ClinicalValue(
                    value=value,
                    unit=unit,
                    unit_basis=basis,
                    method=method,
                    source_field=HEADERS[col],
                    source_cell=f"{col}{index}",
                    source_checksum=binding.checksum,
                    acquired=value is not None,
                    missing_reason="not_recorded" if value is None else None,
                )
        raw, image = read_source(self.root, f"FundusImages/{sample_id}.jpg")
        return PapilaClinicalInput(
            sample_id=sample_id,
            patient_group="PAPILA:" + sample_id[:6],
            laterality="RIGHT" if sample_id.endswith("OD") else "LEFT",
            profile=profile,
            image=image,
            dimensions=jpeg_dimensions(raw),
            age_years=age,
            sex=None if sex is None else {0: "male", 1: "female"}[sex],
            clinical=clinical,
            clinical_source=binding,
        )

    def restricted_diagnosis(self, sample_id):
        if not re.fullmatch(r"RET[0-9]{3}(OD|OS)", sample_id):
            raise AdapterError("papila_sample_invalid")
        rows, binding = self.table(sample_id[-2:])
        index, row = rows["#" + sample_id[3:6]]
        label = number(row.get("D", ""))
        if label not in (0, 1, 2):
            raise AdapterError("papila_diagnosis_code_invalid")
        return dict(
            sample_id=sample_id,
            diagnosis_code=int(label),
            cell=f"D{index}",
            mapping={"0": "healthy", "1": "glaucoma", "2": "suspect"},
            source=binding.model_dump(mode="json"),
            purpose="restricted_evaluation_only",
        )

    def recheck(self, candidate):
        """Rebuild from current source, not caller-owned values or an old checksum claim."""
        value = PapilaClinicalInput.model_validate_json(candidate.model_dump_json())
        current = self.prepare(value.sample_id, profile=value.profile)
        if current != value:
            raise AdapterError("papila_preparation_source_or_projection_changed")
        return current
