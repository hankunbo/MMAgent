"""Canonical, bounded preparation values. This is consistency checking, not admission.

Only immutable bytes/tuples/scalars survive capture. No caller-owned nested object is
used after capture, including declaration/field dictionaries. Source CSV attestations
remain the trusted adapter's responsibility; raw NPZ/JPEG bindings are checked here.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from packages.disease_packs import DiseasePackCatalog
from packages.disease_packs.models import DatasetReference, parse_pack
from packages.object_store import checksum
from packages.tool_benchmark.formats import jpeg_dimensions
from packages.tool_execution.qc import grayscale_png

from .adapters import FAIR_FIELDS, ODIR_FIELDS, wire
from .models import FileBinding
from .reading import AdapterError, basename, read_npz

PREPARATION_VERSION = "prepared-input/2.0.0"
FINGERPRINT_VERSION = "registration-fingerprint/2.0.0"
UNKNOWN_PATIENT = checksum(b"patient_identity_unknown_not_a_group")


@dataclass(frozen=True, repr=False)
class PreparedSnapshot:
    declaration_bytes: tuple[bytes, bytes]
    file_bytes: tuple[bytes, ...]
    images: tuple
    fields_bytes: bytes
    note: str | None
    identity_basis: str
    patient_hash: str

    @property
    def runtime_profile(self):
        return self.fields.get("selection_version") == "training_selected_input/1.0.0"

    @property
    def preparation_version(self):
        return "prepared-input/3.0.0" if self.runtime_profile else PREPARATION_VERSION

    @property
    def fingerprint_version(self):
        return "registration-fingerprint/3.0.0" if self.runtime_profile else FINGERPRINT_VERSION

    @property
    def dataset(self):
        return DatasetReference.model_validate_json(self.declaration_bytes[0])

    @property
    def pack(self):
        return parse_pack(self.declaration_bytes[1])

    @property
    def files(self):
        return tuple(FileBinding.model_validate_json(raw) for raw in self.file_bytes)

    @property
    def fields(self):
        return json.loads(self.fields_bytes)

    def fingerprint(self):
        return checksum(
            wire(
                dict(
                    preparation=self.preparation_version,
                    fingerprint=self.fingerprint_version,
                    adapter="dataset-adapters/2.0.0"
                    if self.runtime_profile
                    else "dataset-adapters/1.0.0",
                    projection="conservative-note/1.0.0",
                    declarations=[json.loads(raw) for raw in self.declaration_bytes],
                    files=[json.loads(raw) for raw in self.file_bytes],
                    source_fields=json.loads(self.fields_bytes),
                    note=self.note,
                    identity_basis=self.identity_basis,
                    patient_hash=self.patient_hash,
                    images=[
                        dict(
                            checksum=checksum(raw),
                            byte_size=len(raw),
                            dimensions=shape,
                            laterality=eye,
                            modality=modality,
                            media_type="image/png" if modality == "SLO" else "image/jpeg",
                        )
                        for raw, shape, eye, modality in self.images
                    ],
                )
            )
        )


def require(condition):
    if not condition:
        raise ValueError


def strings(row, fields):
    require(type(row) is dict and set(row) == set(fields))
    require(all(type(v) is str for v in row.values()))


def capture(sample, catalog=None):
    """Return one validated snapshot before any idempotency access or object staging.

    No fallback for an old PreparedSample lacking original NPZ binding evidence.
    Re-prepare through the adapter. Errors intentionally expose only one stable code.
    """
    try:
        return _capture(sample, catalog)
    except Exception:
        raise AdapterError("prepared_input_inconsistent") from None


def _capture(sample, catalog):
    require(type(sample.files) in (tuple, list) and 2 <= len(sample.files) <= 3)
    require(type(sample.images) in (tuple, list) and 1 <= len(sample.images) <= 2)
    require(type(sample.source_images) in (tuple, list) and len(sample.source_images) <= 1)
    require(type(sample.fields) is dict)
    fields_bytes = wire(sample.fields)
    require(0 < len(fields_bytes) <= 4194304)
    files = tuple(
        FileBinding.model_validate_json(f.model_dump_json(warnings="error")) for f in sample.files
    )
    declarations = (
        sample.dataset.model_dump_json(warnings="error").encode(),
        sample.pack.model_dump_json(warnings="error").encode(),
    )
    images = []
    for raw, shape, eye, modality in sample.images:
        require(type(raw) is bytes and 0 < len(raw) <= 4194304)
        require(type(shape) in (tuple, list) and len(shape) == 2)
        require(all(type(n) is int and n > 0 for n in shape))
        require(type(eye) is str and type(modality) is str)
        images.append((raw, tuple(shape), eye, modality))
    require(sample.note is None or type(sample.note) is str)
    require(type(sample.identity_basis) is str and type(sample.patient_hash) is str)
    snap = PreparedSnapshot(
        declarations,
        tuple(f.model_dump_json(warnings="error").encode() for f in files),
        tuple(images),
        fields_bytes,
        sample.note,
        sample.identity_basis,
        sample.patient_hash,
    )
    # Capture source evidence once too; never consult the mutable sample during build.
    source_images = tuple(sample.source_images)
    dataset, pack, fields = snap.dataset, snap.pack, snap.fields
    if catalog is None:
        catalog = DiseasePackCatalog.from_directory(
            Path(__file__).resolve().parents[2] / "configs/disease_packs"
        )
    require(catalog.dataset(dataset.reference()) == dataset)
    require(catalog.pack(pack.reference()) == pack)
    require(dataset.reference() in pack.definition.dataset_refs)
    require(dataset.owner == "07A" and not dataset.synthetic and not pack.definition.synthetic)
    require(len({f.relative_path for f in files}) == len(files))
    if dataset.code == "SOURCE:DATASET:FAIRVLMED":
        require(dataset.role == "case_evidence" and "SLO" in dataset.modalities)
        require(pack.definition.goal_level == "evidence_planning")
        require(set(fields) == {"csv", "npz", "npz_metadata", "selection_version"})
        row = fields["csv"]
        if snap.runtime_profile:
            import re

            require(type(row) is dict and {"filename", "use"} <= set(row))
            require(2 <= len(row) <= 64)
            require(
                all(type(k) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", k) for k in row)
            )
            require(all(type(v) is str for v in row.values()))
        else:
            strings(row, FAIR_FIELDS)
        require(
            row["use"] == "training"
            and fields["selection_version"]
            in ("training_filename_order/1.0.0", "training_selected_input/1.0.0")
        )
        name = basename(row["filename"], ".npz")
        require(
            tuple(f.relative_path for f in files)
            == ("ReadMe/data_summary.csv", "Dataset/Training/" + name)
        )
        require(snap.note == row.get("note"))
        if not snap.runtime_profile:
            require(type(snap.note) is str)
        require(snap.identity_basis == "unknown" and snap.patient_hash == UNKNOWN_PATIENT)
        require(len(images) == 1 and images[0][2:] == ("UNKNOWN", "SLO"))
        actual_shape, pixels = grayscale_png(images[0][0], 1048576)
        require(actual_shape == images[0][1])
        require(len(source_images) == 1 and type(source_images[0]) is bytes)
        original = source_images[0]
        require(len(original) == files[1].byte_size and checksum(original) == files[1].checksum)
        arrays, metadata = read_npz(original, runtime_profile=snap.runtime_profile)
        source_shape, source_pixels = arrays.pop("slo_fundus")
        require(actual_shape == source_shape and pixels == source_pixels)
        require(
            wire(fields["npz"]) == wire(arrays) and wire(fields["npz_metadata"]) == wire(metadata)
        )
    elif dataset.code == "SOURCE:DATASET:ODIR":
        require(dataset.role == "coarse_triage" and "bilateral_color_fundus" in dataset.modalities)
        require(pack.definition.goal_level == "coarse_multilabel_triage")
        require(
            set(fields) == {"source_rows", "patient_labels", "label_scope", "selection_version"}
        )
        require(fields["selection_version"] == "training_explicit_id_order/1.0.0")
        require(
            fields["label_scope"] == "patient_coarse; keywords_eye; labels_target_derived_image_row"
        )
        rows = fields["source_rows"]
        require(type(rows) is list and 1 <= len(rows) <= 2)
        for row in rows:
            strings(row, ODIR_FIELDS)
        row = rows[0]
        ident = row["ID"]
        require(ident.isascii() and ident.isdigit() and len(ident) <= 12)
        keys = set(ODIR_FIELDS) - {"filename", "filepath", "labels", "target"}
        require(all(all(r[k] == row[k] for k in keys) for r in rows))
        names = tuple(basename(row[k], ".jpg") for k in ("Left-Fundus", "Right-Fundus"))
        require(names[0] != names[1] and len({r["filename"] for r in rows}) == len(rows))
        require(all(r["filename"] in names for r in rows))
        require(all(row[k] in ("0", "1") for k in "NDGCAHMO"))
        require(fields["patient_labels"] == {k: row[k] for k in "NDGCAHMO"})
        require(snap.note is None and not source_images)
        require(snap.identity_basis == "dataset_local_explicit_id")
        require(snap.patient_hash == checksum(("ODIR:explicit_id:" + ident).encode()))
        require(
            tuple(f.relative_path for f in files)
            == ("full_df.csv", *("ODIR-5K/ODIR-5K/Training Images/" + n for n in names))
        )
        require(len(images) == 2)
        for index, (raw, shape, eye, modality) in enumerate(images):
            require((eye, modality) == (("LEFT", "RIGHT")[index], "color_fundus"))
            require(jpeg_dimensions(raw) == shape)
            require(
                files[index + 1].byte_size == len(raw)
                and files[index + 1].checksum == checksum(raw)
            )
    else:
        raise ValueError
    artifact_code = (
        "ARTIFACT:SLO_IMAGE" if dataset.code.endswith("FAIRVLMED") else "ARTIFACT:FUNDUS_IMAGE"
    )
    require(
        any(
            s.input_artifact_code == artifact_code and s.origin == "image_tool"
            for s in pack.definition.evidence_slots
        )
    )
    return snap
