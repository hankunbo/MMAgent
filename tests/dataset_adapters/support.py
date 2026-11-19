"""Small synthetic data only; no actual dataset roots are accessed by pytest."""

import csv
import io
import struct
import zipfile
from pathlib import Path

from packages.dataset_adapters import FairVLMedAdapter, ODIRAdapter
from packages.dataset_adapters.adapters import FAIR_FIELDS, ODIR_FIELDS
from packages.dataset_adapters.reading import NPZ_KEYS
from packages.disease_packs import DiseasePackCatalog
from packages.disease_packs.models import DatasetReference, DiseasePack

ROOT = Path(__file__).resolve().parents[2]


def adapter(root, name="fairvlmed"):
    configs = ROOT / "configs/disease_packs"
    catalog = DiseasePackCatalog.from_directory(configs)
    dataset = DatasetReference.model_validate_json(
        (configs / "datasets" / f"{name}_1.0.0.json").read_bytes()
    )
    pack_name = (
        "fairvlmed_glaucoma_image_note_evidence_v1"
        if name == "fairvlmed"
        else "fundus_multidisease_triage_v1"
    )
    pack = DiseasePack.model_validate_json((configs / f"{pack_name}_1.0.0.json").read_bytes())
    return (FairVLMedAdapter if name == "fairvlmed" else ODIRAdapter)(
        root, catalog, dataset.reference(), pack.reference()
    )


def npy(shape, dtype, data):
    header = repr(dict(descr=dtype, fortran_order=False, shape=shape)).encode() + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def npz(*, replacement=None, extra=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for key in sorted(NPZ_KEYS):
            data = (
                npy((2, 3), "|u1", bytes(range(6)))
                if key == "slo_fundus"
                else npy((), "<U2", "42".encode("utf-32-le"))
                if key == "age"
                else npy((), "<i8", struct.pack("<q", 1))
            )
            z.writestr(key + ".npy", replacement if key == "slo_fundus" and replacement else data)
        if extra:
            z.writestr(*extra)
    return output.getvalue()


def csv_write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fair(root, *, note="IOP OD current: 18 mmHg\nAssessment: SECRET_DIAGNOSIS", count=1):
    rows = []
    for i in range(count):
        name = f"data_{i:05}.npz"
        path = root / "Dataset/Training" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(npz())
        rows.append(
            dict.fromkeys(FAIR_FIELDS, "1")
            | dict(filename=name, use="training", note=note, gpt4_summary="SECRET_SUMMARY")
        )
    csv_write(root / "ReadMe/data_summary.csv", FAIR_FIELDS, rows)
    return adapter(root), rows


def odir(root):
    # Header-only image fixture intentionally covers only metadata; production reader
    # reports UNASSESSED, never a full JPEG decode/QC claim.
    jpg = b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03" + b"\x00" * 10 + b"\xff\xd9"
    directory = root / "ODIR-5K/ODIR-5K/Training Images"
    directory.mkdir(parents=True)
    row = dict.fromkeys(ODIR_FIELDS, "1") | {
        "ID": "11",
        "Left-Fundus": "11_left.jpg",
        "Right-Fundus": "11_right.jpg",
        "Left-Diagnostic Keywords": "SECRET_LEFT",
        "Right-Diagnostic Keywords": "SECRET_RIGHT",
        "filepath": "untrusted/never_follow.jpg",
    }
    rows = []
    for side in ("left", "right"):
        filename = f"11_{side}.jpg"
        (directory / filename).write_bytes(jpg)
        rows.append(row | dict(filename=filename))
    csv_write(root / "full_df.csv", ODIR_FIELDS, rows)
    return adapter(root, "odir"), rows
