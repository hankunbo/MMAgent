"""Reuse registered file readers; collect all explicit eye/visit records, no inference."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from .adapters import odir_index
from .multimodal.adapters import Builder, prepare
from .multimodal.formats import sheets, source
from .multimodal.models import SourceFile
from .papila import PapilaClinicalPreparation
from .reading import AdapterError

ADAPTERS = {
    "fairvlmed": "training",
    "gamma": "training",
    "harvard_gdp": "development_preparation",
    "grape": "development_preparation",
    "papila": "development_preparation",
    "odir": "training",
}


@dataclass
class Decoded:
    identity_basis: str = "sample_only_unproven_patient"
    units: list = field(default_factory=list)
    background: list = field(default_factory=list)
    restricted: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def append(self, value):
        m = value.manifest
        self.units.append(
            dict(
                components=m.components,
                sources=m.sources,
                payloads=m.payloads,
                contents=value.payload_bytes,
                owner_version=m.adapter_version,
            )
        )
        # References only: no diagnosis, label or raw note is copied into the packet.
        self.restricted.append(
            dict(
                record=m.sample,
                fields=list(m.excluded_fields) + list(m.quarantined_fields),
                reason="original_owner_excluded_or_unmapped_not_analysis_input",
            )
        )

    def builder(self, b, version):
        self.units.append(
            dict(
                components=tuple(b.components),
                sources=tuple(b.files.values()),
                payloads=tuple(b.payloads),
                contents=tuple(b.contents),
                owner_version=version,
            )
        )


def _grape_rows(root):
    raw, _ = source(root, "VF_and_clinical_information.xlsx")
    return dict(sheets(raw))


def candidates(adapter, root, *, limit=3, selection=None):
    root = Path(root).absolute()
    check_selection(adapter, selection)
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise AdapterError("patient_packet_candidate_limit")
    if adapter == "grape":
        rows = _grape_rows(root)["Baseline"]
        values = sorted({c["A"] for n, c in rows if n > 2}, key=int)
    elif adapter == "papila":
        owner = PapilaClinicalPreparation(root)
        values = sorted({k[1:] for eye in ("OD", "OS") for k in owner.table(eye)[0]})
    elif adapter == "odir":
        values = sorted(odir_index(root)[0], key=int)
    elif adapter == "fairvlmed":
        from .runtime_input import factory

        values = sorted(factory(adapter, root)._index()[0])
        values = [v for v in values if (root / "Dataset/Training" / v).is_file()]
    elif adapter == "gamma":
        values = sorted(
            p.name
            for p in (root / "training/multi-modality_images").iterdir()
            if p.is_dir() and re.fullmatch(r"[0-9]{4}", p.name)
        )
    else:
        values = sorted(
            {p.name for folder in ("RNFLT", "Bscan") for p in (root / folder).glob("data_*.npz")}
        )
    return values[:limit]


def check_selection(adapter, selection):
    if adapter not in ADAPTERS:
        raise AdapterError("patient_packet_unknown_adapter")
    if selection is not None and selection != ADAPTERS[adapter]:
        raise AdapterError("patient_packet_selection_not_supported")


def decode(adapter, root, subject, *, selection=None):
    check_selection(adapter, selection)
    root = Path(root).absolute()
    result = Decoded()
    if adapter == "grape":
        if not re.fullmatch(r"[1-9][0-9]{0,4}", subject):
            raise AdapterError("patient_packet_subject_expected_not_eye_selector")
        tables = _grape_rows(root)
        rows = [
            (name, n, c)
            for name, rs in tables.items()
            for n, c in rs
            if n > 2 and c.get("A") == subject
        ]
        eyes = sorted({c.get("B") for _, _, c in rows})
        if not eyes or any(e not in ("OD", "OS") for e in eyes):
            raise AdapterError("patient_packet_eye_or_subject_invalid")
        for eye in eyes:
            # Derived from recorded data, not a caller-imposed task cutoff.
            last = max(
                0 if name == "Baseline" else int(c["C"]) for name, _, c in rows if c["B"] == eye
            )
            result.append(
                prepare(
                    "grape",
                    root,
                    f"{subject}:{eye}",
                    cutoff_visit=last,
                    semantic_profile="grape-v1",
                )
            )
            # The archive selected the last recorded visit, not a prediction cutoff.
            reference = result.restricted[-1]
            if "visits_after_cutoff" in reference["fields"]:
                reference["fields"].remove("visits_after_cutoff")
                reference["not_applied_in_full_archive"] = ["visits_after_cutoff"]
        result.identity_basis = "explicit_source_subject"
        result.notes.append("all_recorded_visits_per_eye; visit_numbers_not_cross_eye_date_proof")
        for name, n, cells in rows:
            headers = dict(tables[name])[1]
            fields = (
                {
                    "D": ("Gender", "gender"),
                    "R": ("Acquisition Device", "acquisition_device"),
                    "S": ("Resolusion", "source_resolution"),
                }
                if name == "Baseline"
                else {
                    "G": ("Acquisition Device", "acquisition_device"),
                    "H": ("Resolusion", "source_resolution"),
                }
            )
            for col, (header, concept) in fields.items():
                if headers.get(col) == header:
                    _background(
                        result,
                        concept,
                        None if cells.get(col) == "/" else cells.get(col),
                        "VF_and_clinical_information.xlsx",
                        f"{name}!{col}{n}",
                        eye="RIGHT" if cells["B"] == "OD" else "LEFT",
                        visit="0" if name == "Baseline" else cells["C"],
                        meaning="code_mapping_unqualified"
                        if concept == "gender"
                        else "source_declared",
                    )
        # Preserve every selected cell's location; cells outside the original
        # clinical projection remain restricted/unmapped, never silently dropped.
        result.restricted.append(
            dict(
                source="VF_and_clinical_information.xlsx",
                selected_rows=[dict(sheet=name, row=n, fields=sorted(c)) for name, n, c in rows],
                reason="raw_rows_include_labels; only_projected_cells_are_analysis_material",
            )
        )
    elif adapter == "papila":
        _papila(result, root, subject)
    elif adapter == "odir":
        _odir(result, root, subject)
    else:
        result.append(
            prepare(
                adapter,
                root,
                subject,
                selection=ADAPTERS[adapter],
                semantic_profile="harvard-gdp-v1" if adapter == "harvard_gdp" else None,
            )
        )
        if adapter == "fairvlmed":
            _fair_background(result, root, subject)
    return result


def _background(
    result,
    concept,
    value,
    source_path,
    locator,
    *,
    eye="UNKNOWN",
    visit="unknown",
    meaning="source_declared",
):
    result.background.append(
        dict(
            concept=concept,
            value=None if value in (None, "") else str(value),
            status="missing" if value in (None, "") else "obtained",
            laterality=eye,
            visit=visit,
            source_path=source_path,
            locator=locator,
            meaning_status=meaning,
        )
    )


def _fair_background(result, root, subject):
    from .runtime_input import factory

    selected = factory("fairvlmed", root).prepare(subject)
    for origin, fields in (("csv", selected.fields["csv"]), ("npz", selected.fields["npz"])):
        path = "ReadMe/data_summary.csv" if origin == "csv" else "Dataset/Training/" + subject
        for key in ("gender", "race", "ethnicity", "language", "maritalstatus"):
            if key in fields:
                _background(
                    result,
                    key,
                    fields[key],
                    path,
                    f"{subject}#{key}",
                    meaning="code_mapping_unqualified",
                )
    result.notes.append(
        "CSV_and_NPZ_demographics_kept_as_separate_source_assertions_not_reconciled"
    )


def _papila(result, root, subject):
    if not re.fullmatch(r"[0-9]{3}", subject):
        raise AdapterError("patient_packet_papila_subject_three_digits")
    owner = PapilaClinicalPreparation(root)
    b = Builder(root)
    b.read("README.md")
    records = []
    for eye in ("OD", "OS"):
        rows, _ = owner.table(eye)
        if "#" + subject not in rows:
            continue
        p = owner.prepare(f"RET{subject}{eye}", profile="image_demographics_clinical")
        records.append(p)
        path = p.clinical_source.relative_path
        b.read(path)
        prefix = eye + ":"
        b.image(prefix + "fundus", p.image.relative_path, laterality=p.laterality)
        index = rows["#" + subject][0]
        b.scalar(
            prefix + "age",
            "age",
            p.age_years,
            path,
            f"B{index}",
            unit="years",
            basis="original_clinical_owner",
            laterality=p.laterality,
        )
        _background(result, "sex", p.sex, path, f"C{index}", eye=p.laterality)
        for key, field_value in p.clinical.items():
            b.scalar(
                prefix + key,
                key,
                field_value.value,
                path,
                field_value.source_cell,
                unit=field_value.unit,
                basis=field_value.unit_basis if field_value.unit else None,
                laterality=p.laterality,
                semantic_limits=(field_value.unit_basis, "visit_time_not_provided"),
            )
        result.restricted.append(
            dict(source=path, locator=f"D{index}", reason="diagnosis_evaluation_only")
        )
    if not records:
        raise AdapterError("patient_packet_subject_missing")
    if len(records) == 2 and any(
        getattr(records[0], k) != getattr(records[1], k) for k in ("age_years", "sex")
    ):
        result.notes.append("bilateral_demographic_conflict_preserved; do_not_merge_values")
    result.identity_basis = "explicit_source_subject"
    result.builder(b, "papila-clinical-preparation/1.0.0")


def _odir(result, root, subject):
    groups, binding = odir_index(root)
    if subject not in groups:
        raise AdapterError("patient_packet_subject_missing")
    row = groups[subject][0]
    b = Builder(root)
    b.files["full_df.csv"] = SourceFile(
        path="full_df.csv", checksum=binding.checksum, byte_size=binding.byte_size
    )
    for key, eye in (("Left-Fundus", "LEFT"), ("Right-Fundus", "RIGHT")):
        path = "ODIR-5K/ODIR-5K/Training Images/" + row[key]
        if (root / path).exists():
            b.image(eye, path, laterality=eye)
        else:
            b.missing(
                eye,
                "image",
                "color_fundus",
                "color_fundus",
                path,
                reason="referenced_image_missing",
                laterality=eye,
            )
    b.scalar(
        "age",
        "age",
        row["Patient Age"],
        "full_df.csv",
        f"ID={subject}:Patient Age",
        unit="years",
        basis="source_table_column",
    )
    _background(result, "sex", row["Patient Sex"], "full_df.csv", f"ID={subject}:Patient Sex")
    result.restricted.append(
        dict(
            source="full_df.csv",
            locator=f"ID={subject}",
            fields=list(row),
            reason="diagnostic_and_derived_label_fields_isolated",
        )
    )
    result.identity_basis = "explicit_source_subject"
    result.builder(b, "odir-original-index/1.0.0")
