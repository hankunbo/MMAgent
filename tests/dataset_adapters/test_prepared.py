"""Only synthetic inputs. Invalid preparation must fail before the repository is called."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from packages.dataset_adapters.adapters import FAIR_FIELDS
from packages.dataset_adapters.prepared import capture
from packages.dataset_adapters.reading import AdapterError, grayscale_image
from packages.dataset_adapters.service import DatasetPreparationService
from packages.object_store import LocalObjectStore, checksum
from tests.dataset_adapters.support import csv_write, fair, odir


@pytest.mark.parametrize(
    "kind",
    [
        "note",
        "none",
        "identity",
        "patient",
        "eye",
        "modality",
        "shape",
        "bool_shape",
        "missing",
        "duplicate",
        "corrupt",
        "pixels",
        "source",
        "source_missing",
        "file_order",
        "file_duplicate",
        "fields",
        "split",
        "declaration",
        "cross_pack",
        "oversized",
    ],
)
def test_invalid_preparation_before_repository_and_stage(tmp_path, kind, caplog, capsys):
    a, _ = fair(tmp_path / "source")
    s = a.prepare("data_00000.npz")
    raw, shape, eye, modality = s.images[0]
    if kind == "note":
        s.note = "SECRET_CHANGED_NOTE"
    elif kind == "none":
        s.note = None
    elif kind == "identity":
        s.identity_basis = "dataset_local_explicit_id"
    elif kind == "patient":
        s.patient_hash = checksum(b"SECRET_PATIENT")
    elif kind == "eye":
        s.images = ((raw, shape, "LEFT", modality),)
    elif kind == "modality":
        s.images = ((raw, shape, eye, "SECRET_MODALITY"),)
    elif kind == "shape":
        s.images = ((raw, (3, 2), eye, modality),)
    elif kind == "bool_shape":
        s.images = ((raw, (True, 3), eye, modality),)
    elif kind == "missing":
        s.images = ()
    elif kind == "duplicate":
        s.images *= 2
    elif kind == "corrupt":
        s.images = ((b"SECRET_BAD_IMAGE", shape, eye, modality),)
    elif kind == "pixels":
        s.images = ((grayscale_image(shape, b"654321"), shape, eye, modality),)
    elif kind == "source":
        s.source_images = (b"SECRET_BAD_NPZ",)
    elif kind == "source_missing":
        s.source_images = ()
    elif kind == "file_order":
        s.files = tuple(reversed(s.files))
    elif kind == "file_duplicate":
        s.files = (s.files[0], s.files[0])
    elif kind == "fields":
        s.fields["extra"] = "SECRET_FIELD"
    elif kind == "split":
        s.fields["csv"]["use"] = "test"
    elif kind == "declaration":
        s.dataset = s.dataset.model_copy(update={"version": "9.0.0"})
    elif kind == "cross_pack":
        s.pack = odir(tmp_path / "odir")[0].pack
    elif kind == "oversized":
        s.images = ((b"x" * 4194305, shape, eye, modality),)
    repo = Mock()
    stores = [LocalObjectStore(tmp_path / name) for name in ("visible", "restricted")]
    svc = DatasetPreparationService(repo, *stores)
    with pytest.raises(AdapterError, match="^prepared_input_inconsistent$") as error:
        svc.register(s, key="invalid")
    repo.register.assert_not_called()
    assert all(not list(store.root.glob("objects/*")) for store in stores)
    assert "SECRET" not in str(error.value) + caplog.text + capsys.readouterr().out


@pytest.mark.parametrize(
    "kind",
    [
        "hash",
        "basis",
        "id",
        "note",
        "eyes",
        "modality",
        "shape",
        "files",
        "labels",
        "duplicate_rows",
        "missing_eye",
        "bytes",
    ],
)
def test_odir_identity_and_order_consistency(tmp_path, kind):
    a, _ = odir(tmp_path)
    s = a.prepare("11")
    raw, shape, eye, modality = s.images[0]
    if kind == "hash":
        s.patient_hash = checksum(b"SECRET_OTHER_ID")
    elif kind == "basis":
        s.identity_basis = "unknown"
    elif kind == "id":
        s.fields["source_rows"][0]["ID"] = "12"
    elif kind == "note":
        s.note = ""
    elif kind == "eyes":
        s.images = tuple(reversed(s.images))
    elif kind == "modality":
        s.images = ((raw, shape, eye, "SLO"), s.images[1])
    elif kind == "shape":
        s.images = ((raw, (3, 2), eye, modality), s.images[1])
    elif kind == "files":
        s.files = (s.files[0], s.files[2], s.files[1])
    elif kind == "labels":
        s.fields["patient_labels"]["G"] = "0"
    elif kind == "duplicate_rows":
        s.fields["source_rows"][1] = s.fields["source_rows"][0]
    elif kind == "missing_eye":
        s.images = s.images[:1]
    elif kind == "bytes":
        s.images = ((raw[:-2] + b"changed\xff\xd9", shape, eye, modality), s.images[1])
    with pytest.raises(AdapterError, match="^prepared_input_inconsistent$"):
        capture(s)


def test_canonical_equivalence_and_complete_snapshot(tmp_path):
    a, rows = fair(tmp_path, note="")
    s = a.prepare("data_00000.npz")
    snap = capture(s)
    assert snap.note == "" and snap.patient_hash == s.patient_hash
    equivalent = replace(
        s,
        fields=dict(reversed(list(s.fields.items()))),
        images=tuple((raw, list(shape), eye, mode) for raw, shape, eye, mode in s.images),
    )
    assert equivalent.fingerprint() == snap.fingerprint()
    # Nested mutation of the original or a returned projection never changes the snapshot.
    before = snap.fingerprint()
    s.fields["csv"]["note"] = "SECRET_MUTATION"
    snap.fields["csv"]["note"] = "OTHER"
    assert snap.fingerprint() == before and snap.fields["csv"]["note"] == ""
    rows[0]["note"] = "合法变化"
    csv_write(tmp_path / "ReadMe/data_summary.csv", FAIR_FIELDS, rows)
    assert a.prepare("data_00000.npz").fingerprint() != before
    with pytest.raises(AdapterError):
        replace(a.prepare("data_00000.npz"), note=None).fingerprint()
