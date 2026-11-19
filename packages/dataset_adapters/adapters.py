"""Read-only preparation APIs; deployment owns roots, callers select training identities."""

import json
from dataclasses import dataclass, field

from packages.disease_packs import DiseasePackCatalog
from packages.object_store import checksum
from packages.tool_benchmark.formats import jpeg_dimensions

from .projection import project_note
from .reading import AdapterError, basename, csv_rows, grayscale_image, read_npz, read_source

FAIR_FIELDS = (
    "filename",
    "age",
    "gender",
    "race",
    "ethnicity",
    "language",
    "maritalstatus",
    "note",
    "gpt4_summary",
    "glaucoma",
    "use",
)
ODIR_FIELDS = (
    "ID",
    "Patient Age",
    "Patient Sex",
    "Left-Fundus",
    "Right-Fundus",
    "Left-Diagnostic Keywords",
    "Right-Diagnostic Keywords",
    *"NDGCAHMO",
    "filepath",
    "labels",
    "target",
    "filename",
)


def wire(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


@dataclass(repr=False)
class PreparedSample:
    dataset: object
    pack: object
    files: tuple
    images: tuple  # (bytes, shape, laterality, modality)
    fields: dict = field(repr=False)
    note: str | None = field(default=None, repr=False)
    identity_basis: str = "unknown"
    patient_hash: str = checksum(b"patient_identity_unknown_not_a_group")

    # Original selected NPZ bytes, retained only until canonical capture. No dataset copy.
    source_images: tuple = field(default=(), repr=False)

    def fingerprint(self):
        from .prepared import capture

        return capture(self).fingerprint()


class BaseAdapter:
    def __init__(self, root, catalog: DiseasePackCatalog, dataset_ref, pack_ref):
        from pathlib import Path

        self.root = Path(root)
        self.dataset = catalog.dataset(dataset_ref)
        self.pack = catalog.pack(pack_ref)
        if self.dataset.reference() not in self.pack.definition.dataset_refs:
            raise AdapterError("dataset_pack_mismatch")
        if self.dataset.owner != "07A" or self.dataset.synthetic or self.pack.definition.synthetic:
            raise AdapterError("dataset_owner_invalid")
        self.catalog = catalog

    def split(self, split):
        if split != "training":
            raise AdapterError("validation_test_locked")

    def relate(self, other):
        self.catalog.validate_patient_relation(self.dataset.reference(), other.dataset.reference())
        raise AdapterError("patient_linkage_requires_explicit_record_evidence")


class FairVLMedAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dataset.code != "SOURCE:DATASET:FAIRVLMED":
            raise AdapterError("dataset_mismatch")

    def _index(self):
        rows, binding = csv_rows(self.root, "ReadMe/data_summary.csv", FAIR_FIELDS)
        seen = set()
        selected = {}
        for row in rows:
            name = basename(row["filename"], ".npz")
            if name in seen:
                raise AdapterError("csv_duplicate_filename")
            seen.add(name)
            if row["use"] not in ("training", "validation", "test"):
                raise AdapterError("csv_split_invalid")
            if row["use"] == "training":
                selected[name] = row
        actual = {p.name for p in (self.root / "Dataset/Training").glob("*.npz")}
        if actual != set(selected):
            raise AdapterError("training_npz_csv_missing_or_extra")
        return selected, binding

    def candidates(self, *, count=20, split="training"):
        self.split(split)
        if type(count) is not int or not 1 <= count <= 50:
            raise AdapterError("candidate_count_invalid")
        rows, _ = self._index()
        return tuple(sorted(rows)[:count])

    def prepare(self, filename, *, split="training"):
        self.split(split)
        basename(filename, ".npz")
        rows, csv_binding = self._index()
        if filename not in rows:
            raise AdapterError("training_sample_not_found")
        row = rows[filename]
        raw, binding = read_source(self.root, "Dataset/Training/" + filename)
        arrays, metadata = read_npz(raw)
        shape, pixels = arrays.pop("slo_fundus")
        # CSV labels and NPZ labels remain separate even if they disagree.
        fields = dict(
            csv=row,
            npz=arrays,
            npz_metadata=metadata,
            selection_version="training_filename_order/1.0.0",
        )
        return PreparedSample(
            self.dataset,
            self.pack,
            (csv_binding, binding),
            ((grayscale_image(shape, pixels), shape, "UNKNOWN", "SLO"),),
            fields,
            row["note"],
            source_images=(raw,),
        )


class ODIRAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dataset.code != "SOURCE:DATASET:ODIR":
            raise AdapterError("dataset_mismatch")

    def _index(self):
        return odir_index(self.root)

    def candidates(self, *, count=20, split="training"):
        self.split(split)
        if type(count) is not int or not 1 <= count <= 50:
            raise AdapterError("candidate_count_invalid")
        rows, _ = self._index()
        return tuple(sorted(rows, key=int)[:count])

    def prepare(self, ident, *, split="training"):
        self.split(split)
        groups, csv_binding = self._index()
        if ident not in groups:
            raise AdapterError("training_sample_not_found")
        rows = groups[ident]
        images = []
        files = [csv_binding]
        for column, eye in (("Left-Fundus", "LEFT"), ("Right-Fundus", "RIGHT")):
            raw, binding = read_source(
                self.root, "ODIR-5K/ODIR-5K/Training Images/" + rows[0][column]
            )
            try:
                shape = jpeg_dimensions(raw)
            except Exception:
                raise AdapterError("odir_image_corrupt") from None
            if shape[0] * shape[1] > 16777216:
                raise AdapterError("odir_image_dimensions_limit")
            files.append(binding)
            images.append((raw, shape, eye, "color_fundus"))
        return PreparedSample(
            self.dataset,
            self.pack,
            tuple(files),
            tuple(images),
            dict(
                source_rows=rows,
                patient_labels={k: rows[0][k] for k in "NDGCAHMO"},
                label_scope="patient_coarse; keywords_eye; labels_target_derived_image_row",
                selection_version="training_explicit_id_order/1.0.0",
            ),
            identity_basis="dataset_local_explicit_id",
            patient_hash=checksum(("ODIR:explicit_id:" + ident).encode()),
        )


def projection(sample):
    return project_note(sample.note) if sample.note is not None else None


def odir_index(root):
    rows, binding = csv_rows(root, "full_df.csv", ODIR_FIELDS)
    groups = {}
    seen = set()
    image_owner = {}
    for row in rows:
        ident = row["ID"]
        if not ident.isascii() or not ident.isdigit() or len(ident) > 12:
            raise AdapterError("odir_id_invalid")
        names = [basename(row[k], ".jpg") for k in ("Left-Fundus", "Right-Fundus", "filename")]
        if names[0] == names[1] or names[2] not in names[:2]:
            raise AdapterError("odir_eye_alignment_invalid")
        if (ident, names[2]) in seen:
            raise AdapterError("odir_duplicate_image_row")
        seen.add((ident, names[2]))
        for name in names[:2]:
            if name in image_owner and image_owner[name] != ident:
                raise AdapterError("odir_image_patient_conflict")
            image_owner[name] = ident
        if any(row[k] not in ("0", "1") for k in "NDGCAHMO"):
            raise AdapterError("odir_multilabel_invalid")
        group = groups.setdefault(ident, [])
        keys = (
            "Left-Fundus",
            "Right-Fundus",
            "Patient Age",
            "Patient Sex",
            "Left-Diagnostic Keywords",
            "Right-Diagnostic Keywords",
            *"NDGCAHMO",
        )
        if group and any(group[0][k] != row[k] for k in keys):
            raise AdapterError("odir_patient_rows_conflict")
        group.append(row)
    return groups, binding
